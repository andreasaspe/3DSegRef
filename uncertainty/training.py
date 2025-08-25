

# =========================
# 3D Segmentation + SWAG + Predictive Covariance
# =========================
import math
import random
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# ---------- Utils ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Simple mean Dice across classes (including background by default)
def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    # pred, target: [B,D,H,W] (class indices)
    dice_per_class = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = (2 * inter + eps) / (denom + eps)
        dice_per_class.append(dice)
    return torch.stack(dice_per_class).mean()

# ---------- Basic UNet Blocks ----------
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch)
        )
    def forward(self, x): return self.block(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ---------- Low-rank covariance head ----------
class LowRankLogitCovHead3D(nn.Module):
    def __init__(self, in_ch, n_classes, rank=4, hidden=0, eps=1e-4):
        super().__init__()
        self.C = n_classes
        self.r = rank
        self.eps = eps

        if hidden > 0:
            self.pre = nn.Sequential(
                nn.Conv3d(in_ch, hidden, 1, bias=True),
                nn.GroupNorm(8, hidden),
                nn.SiLU(),
            )
            feat_ch = hidden
        else:
            self.pre = nn.Identity()
            feat_ch = in_ch

        self.mu_head   = nn.Conv3d(feat_ch, n_classes, 1)   # μ(x)
        self.coeff_head= nn.Conv3d(feat_ch, rank, 1)        # a(x)
        self.diag_head = nn.Conv3d(feat_ch, n_classes, 1)   # s(x) -> σ=softplus(s)+eps

        B_init = torch.randn(n_classes, rank) * 0.02 if rank > 0 else torch.zeros(n_classes, 0)
        self.B = nn.Parameter(B_init, requires_grad=(rank > 0))

    def forward(self, feats):
        feats = self.pre(feats)
        mu = self.mu_head(feats)                       # [B,C,D,H,W]
        if self.r > 0:
            a  = self.coeff_head(feats)               # [B,r,D,H,W]
        else:
            a = torch.zeros(mu.size(0), 0, *mu.shape[2:], device=mu.device, dtype=mu.dtype)
        s  = self.diag_head(feats)                     # [B,C,D,H,W]
        sigma = F.softplus(s) + self.eps
        return mu, a, sigma, self.B

# ---------- Full UNet + uncertainty ----------
class UNet3D_Uncertainty(nn.Module):
    def __init__(self, n_channels, n_classes, rank=4, base_ch=32):
        super().__init__()
        self.inc   = DoubleConv3D(n_channels, base_ch)
        self.down1 = Down3D(base_ch, base_ch*2)
        self.down2 = Down3D(base_ch*2, base_ch*4)
        self.down3 = Down3D(base_ch*4, base_ch*8)
        self.down4 = Down3D(base_ch*8, base_ch*8)

        self.up1 = Up3D(base_ch*16, base_ch*4)
        self.up2 = Up3D(base_ch*8, base_ch*2)
        self.up3 = Up3D(base_ch*4, base_ch)
        self.up4 = Up3D(base_ch*2, base_ch)

        self.outc = LowRankLogitCovHead3D(base_ch, n_classes, rank=rank)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        mu, a, sigma, B = self.outc(x)
        return mu, a, sigma, B

# ---------- Logit sampling + losses ----------
def sample_logits(mu, a, sigma, B, K: int):
    """
    Sample logits ~ N(mu, Σ), Σ = (B diag(a))(B diag(a))^T + diag(σ^2)
    Shapes:
      mu:    [B,C,D,H,W]
      a:     [B,r,D,H,W]
      sigma: [B,C,D,H,W]
      B:     [C,r]
    Returns:
      logits_mc: [K,B,C,D,H,W]
    """
    Bsz, C, D, H, W = mu.shape
    r = a.shape[1]
    if K <= 0:
        return mu.unsqueeze(0)

    device = mu.device
    dtype = mu.dtype

    eps = torch.randn(K, Bsz, C, D, H, W, device=device, dtype=dtype)
    diagpart = sigma.unsqueeze(0) * eps

    if r > 0:
        z = torch.randn(K, Bsz, r, D, H, W, device=device, dtype=dtype)
        # Expand B to broadcast over spatial dims: [1,C,r,1,1,1]
        B_exp = B.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lowrank = (B_exp * (a.unsqueeze(0).unsqueeze(2) * z.unsqueeze(2))).sum(dim=3)  # [K,B,C,D,H,W]
        logits = mu.unsqueeze(0) + lowrank + diagpart
    else:
        logits = mu.unsqueeze(0) + diagpart

    return logits

# ---------- SWAG ----------
class SWAG(nn.Module):
    def __init__(self, model, subset_filter: Optional[Callable[[str, nn.Parameter], bool]]=None,
                 max_rank: int = 20, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.max_rank = max_rank
        self.n_models = 0
        self.mean: Dict[str, torch.Tensor] = {}
        self.sq_mean: Dict[str, torch.Tensor] = {}
        self.deltas: list[Dict[str, torch.Tensor]] = []
        self.params = []
        for name, p in model.named_parameters():
            if p.requires_grad and (subset_filter is None or subset_filter(name, p)):
                self.params.append(name)

    @torch.no_grad()
    def collect_model(self, model):
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if self.n_models == 0:
            for k in self.params:
                w = sd[k]
                self.mean[k] = w.clone()
                self.sq_mean[k] = w.clone()**2
            self.n_models = 1
        else:
            n = self.n_models + 1
            for k in self.params:
                w = sd[k]
                self.mean[k] = self.mean[k] + (w - self.mean[k]) / n
                self.sq_mean[k] = self.sq_mean[k] + (w**2 - self.sq_mean[k]) / n
            self.n_models = n

        delta = {k: (sd[k] - self.mean[k]) for k in self.params}
        self.deltas.append(delta)
        if len(self.deltas) > self.max_rank:
            self.deltas.pop(0)

    @torch.no_grad()
    def sample_model(self, model, scale: float = 1.0):
        current = model.state_dict()
        new_state = {}
        for k, v in current.items():
            if k not in self.params:
                new_state[k] = v
                continue
            mean = self.mean[k].to(self.device, dtype=v.dtype)
            var = (self.sq_mean[k] - self.mean[k]**2).clamp_min(1e-30).to(self.device, dtype=v.dtype)

            # diagonal sample
            eps = torch.randn_like(var)
            w = mean + scale * torch.sqrt(var) * eps

            # low-rank sample
            if self.deltas:
                Z = torch.randn(len(self.deltas), device=self.device, dtype=v.dtype)
                D = torch.stack([d[k].to(self.device, dtype=v.dtype) for d in self.deltas], dim=0)  # [R, ...]
                lowrank = torch.sum(Z[:, None] * D.reshape(len(self.deltas), -1), dim=0)
                lowrank = lowrank.view_as(mean) / math.sqrt(len(self.deltas))
                w = w + scale * lowrank

            new_state[k] = w
        model.load_state_dict(new_state)

# ---------- Inference helpers ----------
@torch.no_grad()
def infer_cov_only(model, x, K_logits=16, T=1.0):
    mu, a, sigma, B = model(x)
    logits_mc = sample_logits(mu, a, sigma, B, K=K_logits)
    probs = torch.softmax(logits_mc / T, dim=2).mean(0)              # [B,C,D,H,W]
    pred = probs.argmax(1)
    return pred, probs

@torch.no_grad()
def infer_swag_only(model, swag: SWAG, x, n_weight_samples=4, T=1.0):
    # Uses μ only; no logit sampling
    probs_list = []
    for _ in range(n_weight_samples):
        swag.sample_model(model)
        mu, _, _, _ = model(x)
        probs_list.append(torch.softmax(mu / T, dim=1))
    probs = torch.stack(probs_list, dim=0).mean(0)
    pred = probs.argmax(1)
    return pred, probs

@torch.no_grad()
def infer_both(model, swag: SWAG, x, n_weight_samples=4, K_logits=8, T=1.0):
    probs_list = []
    for _ in range(n_weight_samples):
        swag.sample_model(model)
        mu, a, sigma, B = model(x)
        logits_mc = sample_logits(mu, a, sigma, B, K=K_logits)
        probs_list.append(torch.softmax(logits_mc / T, dim=2).mean(0))
    probs = torch.stack(probs_list, dim=0).mean(0)
    pred = probs.argmax(1)
    return pred, probs

# ---------- Training loops ----------
def train_one_epoch(model, loader, optimizer, loss_fn, device, amp=True):
    model.train()
    scaler = GradScaler(enabled=amp)
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            output = model(x, target = y)
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += output.loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, device, num_classes, mode: str,
               swag: Optional[SWAG]=None, T=1.0, n_weight_samples=4, K_logits=8):
    model.eval()
    losses = []
    dices = []
    ce_loss = StandardSegLoss(T=T)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        if mode == "cov_only":
            pred, probs = infer_cov_only(model, x, K_logits=K_logits, T=T)
            mu, _, _, _ = model(x)  # for reporting CE on μ
            loss = ce_loss(mu, y)
        elif mode == "swag_only":
            assert swag is not None
            pred, probs = infer_swag_only(model, swag, x, n_weight_samples=n_weight_samples, T=T)
            # evaluate CE on the *current* μ (deterministic snapshot)
            mu, _, _, _ = model(x)
            loss = ce_loss(mu, y)
        elif mode == "both":
            assert swag is not None
            pred, probs = infer_both(model, swag, x, n_weight_samples=n_weight_samples, K_logits=K_logits, T=T)
            mu, _, _, _ = model(x)
            loss = ce_loss(mu, y)
        else:
            raise ValueError("Unknown mode")

        dice = dice_score(pred, y, num_classes=num_classes)
        losses.append(loss.item())
        dices.append(dice.item())
    return sum(losses)/max(1,len(losses)), sum(dices)/max(1,len(dices))

def swag_collect_epoch(model, loader, optimizer, loss_fn, swag: SWAG, device, amp=True):
    """One SWAG collection epoch on your trained model, continuing training."""
    model.train()
    scaler = GradScaler(enabled=amp)
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            out = model(x, target = y)
        scaler.scale(out.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # collect after each optimizer step
        swag.collect_model(model)
        running += loss.item()
    return running / max(1, len(loader))

# ---------- Experiment harness ----------
@dataclass
class ExperimentConfig:
    n_channels: int
    n_classes: int
    rank: int = 4                     # low-rank for predictive covariance head
    base_ch: int = 32
    device: str = "cuda"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    amp: bool = True
    # Loss / sampling
    T: float = 1.0
    K_train: int = 2                  # MC samples during training (cov modes)
    K_eval: int = 16                  # MC samples during eval
    # SWAG
    swag_epoch: int = 1               # extra epoch for SWAG collection
    swag_max_rank: int = 20
    swag_weight_samples_eval: int = 4
    swag_subset_filter: Optional[Callable[[str, nn.Parameter], bool]] = None  # e.g., lambda n,p: n.startswith("outc")
    swag_scale: float = 1.0



def run_cov_experiment()
def run_experiment(mode: str,
                   train_loader, val_loader,
                   cfg: ExperimentConfig):
    """
    mode in {"swag_only", "cov_only", "both"}
    train_loader / val_loader yield (volume, labels)
    """
    set_seed(42)
    device = cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu"

    # Model
    if mode == "swag_only":
        # no predictive covariance effect in training: use rank=0 and ignore sigma in loss
        model = UNet3D_Uncertainty(cfg.n_channels, cfg.n_classes, rank=0, base_ch=cfg.base_ch).to(device)
        loss_fn = StandardSegLoss(T=cfg.T)
    elif mode == "cov_only":
        model = UNet3D_Uncertainty(cfg.n_channels, cfg.n_classes, rank=cfg.rank, base_ch=cfg.base_ch).to(device)
        loss_fn = LogisticNormalSegLoss(T=cfg.T, K=cfg.K_train)
    elif mode == "both":
        model = UNet3D_Uncertainty(cfg.n_channels, cfg.n_classes, rank=cfg.rank, base_ch=cfg.base_ch).to(device)
        loss_fn = LogisticNormalSegLoss(T=cfg.T, K=cfg.K_train)
    else:
        raise ValueError("mode must be one of {'swag_only','cov_only','both'}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- Stage A: Train base model
    for ep in range(cfg.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp=cfg.amp)
        val_loss, val_dice = eval_epoch(
            model, val_loader, device, cfg.n_classes,
            mode=("cov_only" if mode != "swag_only" else "cov_only"),  # use cov_only eval for stable metric on μ
            swag=None, T=cfg.T, K_logits=cfg.K_eval
        )
        print(f"[StageA][{ep+1}/{cfg.epochs}] train_loss={tr_loss:.4f} val_ce_on_mu={val_loss:.4f} val_dice={val_dice:.4f}")

    # ---- Stage B: Optional SWAG collection epoch
    swag = None
    if mode in ("swag_only", "both") and cfg.swag_epoch > 0:
        swag = SWAG(model,
                    subset_filter=cfg.swag_subset_filter,  # e.g., lambda n,p: n.startswith("outc")
                    max_rank=cfg.swag_max_rank,
                    device=device)
        # Keep training for 1 epoch while collecting
        tr_loss_swag = swag_collect_epoch(model, train_loader, optimizer, loss_fn, swag, device, amp=cfg.amp)
        print(f"[SWAG collect] train_loss={tr_loss_swag:.4f} snapshots={swag.n_models} rank_used={len(swag.deltas)}")

    # ---- Final evaluation according to requested mode
    if mode == "swag_only":
        vloss, vdice = eval_epoch(
            model, val_loader, device, cfg.n_classes,
            mode="swag_only", swag=swag, T=cfg.T,
            n_weight_samples=cfg.swag_weight_samples_eval
        )
        print(f"[Final][SWAG-only] val_ce_on_mu={vloss:.4f} val_dice={vdice:.4f}")
    elif mode == "cov_only":
        vloss, vdice = eval_epoch(
            model, val_loader, device, cfg.n_classes,
            mode="cov_only", swag=None, T=cfg.T, K_logits=cfg.K_eval
        )
        print(f"[Final][Cov-only] val_ce_on_mu={vloss:.4f} val_dice={vdice:.4f}")
    elif mode == "both":
        vloss, vdice = eval_epoch(
            model, val_loader, device, cfg.n_classes,
            mode="both", swag=swag, T=cfg.T,
            n_weight_samples=cfg.swag_weight_samples_eval, K_logits=cfg.K_eval
        )
        print(f"[Final][Both] val_ce_on_mu={vloss:.4f} val_dice={vdice:.4f}")

    return model, swag



# =========================
# Example usage (pseudo):
# =========================
# train_loader = ...
# val_loader = ...
# cfg = ExperimentConfig(
#     n_channels=1, n_classes=3, rank=4, base_ch=32,
#     epochs=50, lr=1e-3, weight_decay=1e-5,
#     K_train=2, K_eval=16,
#     swag_epoch=1, swag_max_rank=20, swag_weight_samples_eval=4,
#     swag_subset_filter=lambda n,p: n.startswith("outc"),  # SWAG only on the uncertainty head
#     device="cuda", amp=True
# )
# model, swag = run_experiment("both", train_loader, val_loader, cfg)
