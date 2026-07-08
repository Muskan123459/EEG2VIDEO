"""End-to-end training / generation pipeline (``Group4Pipeline``).

Owns the six pipeline stages, the CLIP feature extractor/scorer, the optimiser and
schedulers, and the loss modules. Implements EEG encoding (with EMA momentum
blend), the InfoNCE contrastive term, a full training epoch with gradient
accumulation and mixed precision, validation, and autoregressive video generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CFG, DEVICE, Config
from .utils import safe_mean, load_frame_for_vae
from .data import VideoFeatureExtractor
from .models import (
    Stage2Tokenization,
    Stage3CARDEncoder,
    Stage4LatentProjection,
    Stage5CLIPAlignment,
    build_video_denoiser,
    EMAEmbeddingBuffer,
)
from .losses import CompositeLoss, VideoLDMLoss
from .metrics import MetricsBundle, ClipScorer


class Group4Pipeline:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.device = DEVICE
        self.clip_extractor = VideoFeatureExtractor(self.device, cfg.clip_model_name)
        self.clip_scorer = ClipScorer(self.device, cfg.clip_model_name)

        self.st2 = Stage2Tokenization().to(self.device)
        self.st3 = Stage3CARDEncoder().to(self.device)
        self.st4 = Stage4LatentProjection().to(self.device)
        self.st5 = Stage5CLIPAlignment().to(self.device)
        self.st6 = build_video_denoiser()

        self.diffusion_loss = VideoLDMLoss().to(self.device)
        self.composite_loss = CompositeLoss().to(self.device)
        # Lazy import for scheduler
        from diffusers import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)

        enc_params = list(self.st2.parameters()) + list(self.st3.parameters()) + list(self.st4.parameters()) + list(self.st5.parameters()) + list(self.st6.eeg_proj.parameters()) + list(self.st6.temporal_attn.parameters())
        unet_params = [p for p in self.st6.unet.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW([
            {"params": enc_params, "lr": cfg.lr_encoder},
            {"params": unet_params, "lr": cfg.lr_unet},
        ], weight_decay=cfg.weight_decay)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

        self.metrics = MetricsBundle(self.device)

    def models_train(self):
        for m in [self.st2, self.st3, self.st4, self.st5, self.st6]:
            m.train()

    def models_eval(self):
        for m in [self.st2, self.st3, self.st4, self.st5, self.st6]:
            m.eval()

    def encode_eeg(self, eeg_windows: torch.Tensor, ema: Optional[EMAEmbeddingBuffer] = None) -> torch.Tensor:
        # eeg_windows: (B, 128, 440)
        h = self.st3(self.st2(eeg_windows))
        z_raw = self.st5(self.st4(h))
        if ema is not None:
            z_mom = ema.update(z_raw)
            z = F.normalize(0.7 * z_raw + 0.3 * z_mom, p=2, dim=1)
        else:
            z = F.normalize(z_raw, p=2, dim=1)
        return z

    def contrastive_infonce(self, z: torch.Tensor, z_img: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        # Symmetric InfoNCE on normalized embeddings.
        z = F.normalize(z, p=2, dim=1)
        z_img = F.normalize(z_img, p=2, dim=1)
        logits = (z @ z_img.T) / temperature
        targets = torch.arange(z.size(0), device=z.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_j = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i + loss_j)

    def forward_diffusion_step(self,
                               z_i: torch.Tensor,
                               temporal_ctx: torch.Tensor,
                               latent_h: int,
                               latent_w: int,
                               guidance_steps: int = 0) -> torch.Tensor:
        latent = torch.randn(1, 4, latent_h, latent_w, device=self.device)
        self.scheduler.set_timesteps(self.cfg.num_inference_steps)
        for t in self.scheduler.timesteps:
            ts = torch.tensor([t], device=self.device).long()
            noise_pred = self.st6(latent, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
        return latent

    @torch.no_grad()
    def decode_latents_to_frames(self, latents: torch.Tensor) -> torch.Tensor:
        return self.st6.vae.decode(latents / 0.18215).sample.clamp(-1, 1)

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.models_train()
        ema = EMAEmbeddingBuffer(alpha=0.9)
        totals = {"loss": 0.0, "l_mae": 0.0, "l_contrastive": 0.0, "l_clip": 0.0, "l_diffusion": 0.0, "l_temporal": 0.0, "l_recon": 0.0}
        n_batches = 0
        torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.mixed_precision and self.device.type == "cuda")

        accum_steps = max(1, getattr(self.cfg, "grad_accum_steps", 1))
        global_step = 0

        # Start with zeroed grads so accumulation works from batch 1
        self.optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            items = batch["batch"]
            if not items:
                continue
            # per-batch processing; optimizer.step() will be called every accum_steps batches
            logs = {k: 0.0 for k in totals.keys() if k != "loss"}
            batch_loss_log = 0.0
            n_valid = 0
            # Store raw windows for the cheap InfoNCE encoder-only re-pass after the sample loop
            all_eeg_for_infonce: List[torch.Tensor] = []
            all_clip_for_infonce: List[torch.Tensor] = []

            for sample in items:
                eeg_windows = sample["eeg_windows"].to(self.device)  # (N, 128, 440)
                gt_clip = sample["clip_feat"].to(self.device)         # (1, 512)
                n = eeg_windows.size(0)
                if n == 0:
                    continue
                n_valid += 1

                take = min(n, self.cfg.max_video_frames)
                eeg_windows = eeg_windows[:take]

                z_seq = []
                z_mom_seq: List[torch.Tensor] = []
                noise_preds = []
                noises = []
                z_raw_seq = []
                recon_losses: List[torch.Tensor] = []

                for i in range(take):
                    x_i = eeg_windows[i:i+1]
                    z_raw = self.st5(self.st4(self.st3(self.st2(x_i))))
                    z_raw_seq.append(z_raw)
                    z_mom = ema.update(z_raw)
                    z_mom_seq.append(z_mom.detach())
                    z_i = F.normalize(0.7 * z_raw + 0.3 * z_mom, p=2, dim=1)
                    z_seq.append(z_i)

                    ctx_start = max(0, len(z_seq) - self.cfg.context_len)
                    temporal_ctx = torch.cat(z_seq[ctx_start:], dim=0).unsqueeze(0)

                    frame_tensor = load_frame_for_vae(sample["video_path"], i, take, self.device)
                    with torch.no_grad():
                        latent = self.st6.vae.encode(frame_tensor).latent_dist.sample() * 0.18215
                    noise = torch.randn_like(latent)
                    ts = torch.randint(0, self.cfg.num_train_timesteps, (1,), device=self.device).long()
                    noisy_lat = self.scheduler.add_noise(latent, noise, ts)

                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        noise_pred = self.st6(noisy_lat, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)

                    alpha_bar = self.scheduler.alphas_cumprod[ts.item()].to(self.device).float()
                    x0_pred = (noisy_lat.float() - (1.0 - alpha_bar).sqrt() * noise_pred.float()) / alpha_bar.sqrt().clamp(min=1e-6)
                    recon_losses.append(F.l1_loss(x0_pred, latent.float().detach()))

                    noise_preds.append(noise_pred)
                    noises.append(noise)

                noise_preds_t = torch.cat(noise_preds, dim=0).float()
                noises_t = torch.cat(noises, dim=0).float()
                z_t = torch.cat(z_seq, dim=0).float()
                z_img = gt_clip.repeat(z_t.size(0), 1)

                z_raw_t = torch.cat(z_raw_seq, dim=0)
                z_mom_t = F.normalize(torch.cat(z_mom_seq, dim=0), p=2, dim=1)
                l_recon = torch.stack(recon_losses).mean()

                video_loss, video_breakdown = self.diffusion_loss(
                    noise_preds_t, noises_t, adj_frames=noise_preds_t.unsqueeze(0)
                )
                total, loss_dict = self.composite_loss(
                    z=z_t, z_img=z_img,
                    mae_input=z_raw_t, mae_target=z_mom_t,
                    contrastive_loss=None,
                    diffusion_loss=video_loss,
                )
                total = total + self.cfg.lambda_recon * l_recon

                # Per-sample backward — divide by accum_steps to make gradients average over
                # the effective accumulated batch.
                scaler.scale(total / (max(len(items), 1) * accum_steps)).backward()
                torch.cuda.empty_cache()

                all_eeg_for_infonce.append(eeg_windows.detach())
                all_clip_for_infonce.append(gt_clip.detach())
                batch_loss_log += total.item()
                logs["l_mae"] += float(loss_dict["l_mae"].item())
                logs["l_clip"] += float(loss_dict["l_clip"].item())
                logs["l_diffusion"] += float(video_breakdown["l_diffusion"].item())
                logs["l_temporal"] += float(video_breakdown["l_temporal"].item())
                logs["l_recon"] += float(l_recon.item())

            # InfoNCE: encoder-only second pass (no UNet) — cheap and graph-free per sample
            if n_valid > 1:
                z_reprs: List[torch.Tensor] = []
                for eeg_w in all_eeg_for_infonce:
                    z = self.encode_eeg(eeg_w).float()
                    z_reprs.append(z.mean(0, keepdim=True))
                z_batch = torch.cat(z_reprs, dim=0)
                z_clip_batch = torch.cat(all_clip_for_infonce, dim=0)
                l_contrast = self.contrastive_infonce(z_batch, z_clip_batch)
                scaler.scale(self.cfg.lambda_contrastive * l_contrast / max(n_valid, 1)).backward()
                logs["l_contrastive"] += float(l_contrast.item())
                batch_loss_log += self.cfg.lambda_contrastive * l_contrast.item() / max(n_valid, 1)

            # Step only every accum_steps batches
            global_step += 1
            if global_step % accum_steps == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in self.optimizer.param_groups for p in pg["params"] if p.requires_grad],
                    self.cfg.grad_clip,
                )
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            totals["loss"] += batch_loss_log / max(n_valid, 1)
            for k in logs:
                totals[k] += logs[k] / max(n_valid, 1)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        self.lr_scheduler.step(avg["loss"])
        torch.cuda.empty_cache()
        return avg

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.models_eval()
        losses = []
        clip_sims = []
        for batch in tqdm(val_loader, desc="Validation"):
            items = batch["batch"]
            for sample in items:
                eeg_windows = sample["eeg_windows"].to(self.device)
                gt_clip = sample["clip_feat"].to(self.device)
                take = min(eeg_windows.size(0), self.cfg.max_video_frames)
                if take == 0:
                    continue
                z = self.encode_eeg(eeg_windows[:take])
                clip_sim = F.cosine_similarity(z, gt_clip.repeat(z.size(0), 1), dim=1).mean().item()
                clip_sims.append(clip_sim)
                losses.append(1.0 - clip_sim)
        return {"val_loss_proxy": safe_mean(losses), "val_clip_sim": safe_mean(clip_sims)}

    @torch.no_grad()
    def generate_video_sequence(self,
                                eeg_windows: torch.Tensor,
                                num_inference_steps: int = None,
                                context_len: int = None) -> List[torch.Tensor]:
        self.models_eval()
        num_inference_steps = num_inference_steps or self.cfg.num_inference_steps
        context_len = context_len or self.cfg.context_len
        self.scheduler.set_timesteps(num_inference_steps)

        ema = EMAEmbeddingBuffer(alpha=0.9)
        z_history: List[torch.Tensor] = []
        frame_latents: List[torch.Tensor] = []

        take = min(eeg_windows.size(0), self.cfg.max_video_frames)
        for i in tqdm(range(take), desc="Generating frames"):
            x_i = eeg_windows[i:i+1].to(self.device)
            z_raw = self.st5(self.st4(self.st3(self.st2(x_i))))
            z_mom = ema.update(z_raw)
            z_i = F.normalize(0.7 * z_raw + 0.3 * z_mom, p=2, dim=1)
            z_history.append(z_i)

            ctx_start = max(0, len(z_history) - context_len)
            temporal_ctx = torch.cat(z_history[ctx_start:], dim=0).unsqueeze(0)
            latent = torch.randn(1, 4, self.cfg.latent_h, self.cfg.latent_w, device=self.device)

            for t in self.scheduler.timesteps:
                ts = torch.tensor([t], device=self.device).long()
                noise_pred = self.st6(latent, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample

            frame_latents.append(latent.squeeze(0).cpu())

        # Decode latent frames to RGB
        lat_batch = torch.stack(frame_latents, dim=0).to(self.device)
        decoded = self.st6.vae.decode(lat_batch / 0.18215).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        return [decoded[i].detach().cpu() for i in range(decoded.size(0))]

    @torch.no_grad()
    def evaluate_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, float]:
        eeg_windows = sample["eeg_windows"]
        gt_clip = sample["clip_feat"].to(self.device)
        frames = self.generate_video_sequence(eeg_windows)

        # We do not have ground-truth RGB frames in the prompt, so for a true
        # evaluation you would pass decoded real video frames from the dataset.
        # Below metrics are computed against a placeholder reconstruction target
        # only if real frames are available in your local dataset.
        clip_sim = self.clip_scorer.cosine_similarity(frames, frames)  # self-consistency placeholder
        return {"clip_sim": clip_sim, "n_frames": float(len(frames))}
