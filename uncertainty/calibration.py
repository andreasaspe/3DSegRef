import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryCalibrationError
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from trainer import get_trainer, dice_score

class Calibration:
    def __init__(self, dataloader, model, device="cuda"):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.device = device

        self.probs = None
        self.labels = None
        self.ece = None
        self.mce = None

    def handle_various_input(self, elem):
        
        input_volume, target = elem['data'], elem['target']
        
        target = target[0].squeeze(1)
        if isinstance(input_volume, list):
            if not all(vol.shape[0] == 1 for vol in input_volume):
                input_volume = [vol.unsqueeze(0) for vol in input_volume]
            input_volume = torch.cat(input_volume, dim = 0)
        
        if isinstance(target, list):
            if not all(tar.shape[0] == 1 for tar in target):
                target = [tar.unsqueeze(0) for tar in target]
            target = torch.cat(target, dim = 0)
        
        return input_volume, target

    @torch.no_grad()
    def predict(self, max_idx = 10):
        self.model.eval()
        all_probs, all_labels, dices = [], [], []
        max_index = max_idx
        for i in tqdm(range(len(self.dataloader)), desc = 'predicting'):
            x, y = self.handle_various_input(self.dataloader[i])
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()

            # model outputs probs with 2 channels: [B, 2, D, H, W]
            probs = self.model(x, targets = y, reduction = 'mean').mu
            dices.append(dice_score(probs, y))
            all_probs.append(probs.detach().cpu()[:, 1].squeeze(0))
            all_labels.append(y.detach().cpu().squeeze(0))
            
            if i == max_index:
                break 

        self.probs = torch.cat(all_probs).flatten()
        self.labels = torch.cat(all_labels).flatten()

        # Compute calibration errors
        ece_metric = BinaryCalibrationError(n_bins=50, norm='l2')
        mce_metric = BinaryCalibrationError(n_bins=50, norm='max')

        self.ece = ece_metric(self.probs, self.labels).item()
        self.mce = mce_metric(self.probs, self.labels).item()
        self.dices = dices
        return self.ece, self.mce, dices

    def visualize(
        self,
        ax=None,
        n_bins=15,
        label=None,
        binning="uniform",
        accuracy_mode="empirical",
        threshold=0.5,
        show_counts=False,
        return_numbers = True,
        ):
        """
        Plot calibration (reliability) curve.

        Args:
            ax (matplotlib axis): existing axis to draw on, or None to create new.
            n_bins (int): number of bins requested.
            label (str): legend label for this curve.
            binning (str): 'uniform' or 'quantile'.
            accuracy_mode (str): 'empirical' -> plot mean(label) per bin (empirical positive frequency),
                                 'accuracy'   -> plot fraction correct in each bin using threshold.
            threshold (float): threshold used to define predicted class when accuracy_mode='accuracy'.
            show_counts (bool): whether to annotate each plotted point with the number of samples in that bin.
        Returns:
            ax: matplotlib axis containing the plot.
        """
        if self.probs is None or self.labels is None:
            raise RuntimeError("Call .predict() before visualize().")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        probs = self.probs.numpy()
        labels = self.labels.numpy()

        # --- compute bin edges ---
        if binning == "uniform":
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif binning == "quantile":
            # quantiles may create duplicate edges if many identical probs; handle that.
            edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
            edges[0], edges[-1] = 0.0, 1.0
            unique_edges = np.unique(edges)
            if unique_edges.size - 1 < 1:
                raise ValueError("Not enough unique quantile edges to form bins.")
            if unique_edges.size - 1 != n_bins:
                warnings.warn(
                    f"Quantile bin edges collapsed; using {unique_edges.size - 1} bins instead of {n_bins}."
                )
                n_bins = unique_edges.size - 1
                edges = unique_edges
        else:
            raise ValueError("binning must be 'uniform' or 'quantile'")

        # digitize (bins define edges of length n_bins+1)
        binids = np.digitize(probs, edges) - 1  # indices 0..n_bins-1 normally
        # clamp to valid range
        binids = np.clip(binids, 0, max(0, edges.size - 2))

        # prepare bin statistics
        bin_conf = np.full(n_bins, np.nan)   # mean predicted prob in bin
        bin_pos = np.full(n_bins, np.nan)    # mean label in bin (empirical positive freq)
        bin_acc = np.full(n_bins, np.nan)    # fraction correct in bin (using threshold)
        bin_count = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = binids == i
            if not np.any(mask):
                continue
            bin_count[i] = int(mask.sum())
            bin_conf[i] = probs[mask].mean()
            bin_pos[i] = labels[mask].mean()
            bin_acc[i] = (labels[mask] == (probs[mask] > threshold)).mean()

        valid = ~np.isnan(bin_conf)
        if not valid.any():
            raise RuntimeError("No bins contain data (check n_bins / binning).")

        x = bin_conf[valid]
        if accuracy_mode == "empirical":
            y = bin_pos[valid]
            y_label = "Empirical positive freq."
            legend_label = label or "empirical (pos freq)"
        elif accuracy_mode == "accuracy":
            y = bin_acc[valid]
            y_label = f"Accuracy (threshold={threshold})"
            legend_label = label or f"accuracy (th={threshold})"
        else:
            raise ValueError("accuracy_mode must be 'empirical' or 'accuracy'")

        # --- plot ---
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="gray", label="_nolegend_")
        ax.plot(x, y, marker="o", linestyle="-", label=legend_label)
        if show_counts:
            for xi, yi, c in zip(x, y, bin_count[valid]):
                ax.text(xi, yi, f"{c}", fontsize=7, ha="center", va="bottom", alpha=0.8)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Confidence (mean predicted probability)")
        ax.set_ylabel(y_label)
        ax.set_title("Calibration Curve")
        ax.grid(alpha=0.3)
        ax.legend()

        if return_numbers:
            return ax, x, y
        return ax 
