import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchmetrics.classification import MulticlassCalibrationError



class NDUncertaintyCalibration:

    def __init__(self, posterior_probabilites, labels=None, kernel_size=5, threshold = 0.85):

        self.posterior_probabilites = posterior_probabilites
        self.labels = labels
        self.kernel_size = kernel_size
        self.threshold = threshold

    def smooth_probs_with_log_conv(
            self,
            probs: torch.Tensor,
            labels: torch.Tensor,
            kernel_size: int = 3,
            mode: str = "geometric_mean",  # or "joint"
    ):
        """
        Smooth class probabilities with a log-space convolution.

        Args:
            probs: Tensor [C, D, H, W] or [B, C, D, H, W], class probabilities per voxel.
            labels: Tensor [D, H, W] or [B, D, H, W], ground truth (unused here, kept for API).
            kernel_size: Size of the cubic kernel (odd integer).
            mode: "geometric_mean" (normalized kernel) or "joint" (unnormalized).
        Returns:
            Smoothed probabilities, same shape as probs.
        """
        eps = 1e-8
        is_batched = (probs.ndim == 5)

        if not is_batched:
            probs = probs.unsqueeze(0)  # add batch dim
            labels = labels.unsqueeze(0)

        B, C, D, H, W = probs.shape

        # 1) Log transform
        logp = torch.log(probs.clamp_min(eps))

        # 2) Build kernel
        kernel = torch.ones((C, 1, kernel_size, kernel_size, kernel_size), device=probs.device)
        if mode == "geometric_mean":
            kernel = kernel / kernel.numel()  # normalize for mean
        elif mode == "joint":
            pass  # leave unnormalized
        else:
            raise ValueError("mode must be 'geometric_mean' or 'joint'")

        # 3) Depthwise 3D convolution (per class)
        logp_smooth = F.conv3d(
            logp, kernel, padding=kernel_size // 2, groups=C
        )

        # 4) Back to probabilities
        probs_smooth = torch.exp(logp_smooth)

        # Normalize so that per-voxel probabilities sum to 1
        probs_smooth = probs_smooth / probs_smooth.sum(dim=1, keepdim=True)

        if not is_batched:
            probs_smooth = probs_smooth.squeeze(0)
            labels = labels.squeeze(0)

        return probs_smooth

    def find_uncertain_regions(self, probs_joint: torch.Tensor, threshold: float = 0.6):
        """
        Given joint probabilities [C, D, H, W], find connected regions of low confidence.

        Args:
            probs_joint: torch.Tensor, shape [C, D, H, W], probabilities per class
            threshold: confidence threshold, voxels below this are uncertain
        Returns:
            components: list of dicts with keys:
                - 'label_id': connected component id
                - 'size': number of voxels
                - 'bbox': bounding box (z1,z2,y1,y2,x1,x2)
                - 'mask': boolean mask for that component
                - 'mean_conf': mean confidence in that component
        """
        # 1) confidence map = max probability per voxel
        conf = probs_joint.max(dim=0).values  # [D, H, W]

        # 2) threshold for low confidence
        uncertain_mask = conf < threshold
        uncertain_np = uncertain_mask.cpu().numpy()

        # 3) connected components (3D, 26-connectivity)
        labeled, ncomp = ndimage.label(uncertain_np, structure=np.ones((3, 3, 3)))

        components = []
        for comp_id in range(1, ncomp + 1):
            mask = (labeled == comp_id)

            if mask.sum() == 0:
                continue
            # bounding box
            coords = np.where(mask)
            z1, z2 = coords[0].min(), coords[0].max()
            y1, y2 = coords[1].min(), coords[1].max()
            x1, x2 = coords[2].min(), coords[2].max()

            # mean confidence inside component
            mean_conf = conf.cpu().numpy()[mask].mean()

            components.append({
                "label_id": comp_id,
                "size": int(mask.sum()),
                "bbox": (z1, z2, y1, y2, x1, x2),
                "mask": mask,
                "mean_conf": float(mean_conf),
            })

        return components

    def compute_calibration_errors_torchmetrics(
            self, probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, num_classes: int = None
    ):
        """
        Compute calibration errors using torchmetrics.

        Args:
            probs: [C, D, H, W] tensor of probabilities
            labels: [D, H, W] tensor of ground truth labels
            n_bins: number of bins for calibration
            num_classes: number of classes (optional, inferred from probs)
        Returns:
            dict with {"ECE", "MCE"}
        """
        if num_classes is None:
            num_classes = probs.shape[0]

        # flatten
        # probs = probs.permute(1, 2, 3, 0).reshape(-1, num_classes)  # [N, C]
        labels = labels.reshape(-1)  # [N]
        probs = probs.permute(1, 0)
        # torchmetrics ECE
        ece_metric = MulticlassCalibrationError(
            num_classes=num_classes, n_bins=n_bins, norm="l1"
        )
        mce_metric = MulticlassCalibrationError(
            num_classes=num_classes, n_bins=n_bins, norm='max'
        )

        ece = ece_metric(probs, labels).item()
        mce = mce_metric(probs, labels).item()
        return {"ECE": ece, "MCE": mce}

    def component_wise_calibration_torchmetrics(
            self, components, probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
    ):
        """
        Compute calibration metrics per connected component using torchmetrics.
        """
        results = []
        C = probs.shape[0]

        for comp in components:
            mask = torch.from_numpy(comp["mask"]).to(labels.device)

            probs_comp = probs[:, mask]  # [C, Nvox]
            labels_comp = labels[mask]  # [Nvox]

            if labels_comp.numel() < 3:
                continue

            metrics = self.compute_calibration_errors_torchmetrics(
                probs_comp, labels_comp, n_bins=n_bins, num_classes=C
            )

            results.append({
                "label_id": comp["label_id"],
                "size": comp["size"],
                "mean_conf": comp["mean_conf"],
                "ECE": metrics["ECE"],
                "MCE": metrics["MCE"],
            })

        return results

    def do_analysis(self, ):
        regions = self.find_uncertain_regions(self.posterior_probabilites, threshold=self.threshold)
        results = self.component_wise_calibration_torchmetrics(
            regions, self.posterior_probabilites, self.labels,
        )

        full_results = {
            'full_region': self.compute_calibration_errors_torchmetrics(self.posterior_probabilites, self.labels),
            'component_wise': results
        }

        return full_results


