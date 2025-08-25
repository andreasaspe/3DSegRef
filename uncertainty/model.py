import pickle

import torch.nn as nn
import torch
import os
import numpy as np
import torch.nn.functional as F
from attr.converters import optional
from pandocfilters import attributes
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.architectures import unet
from dynamic_network_architectures.building_blocks import unet_decoder as unet_dec
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dataclasses import dataclass
from typing import Union, Type, List, Tuple, Optional
from collections import OrderedDict
from transformers.utils import ModelOutput


@dataclass
class UncertaintyModelOutput(ModelOutput):
    mu: Optional[torch.FloatTensor] = None
    cov_mat: Optional[torch.FloatTensor] = None
    sigma_diag: Optional[torch.FloatTensor] = None
    sample_logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_decomp: Optional[dict] = None


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




class UnetWithUncertainty(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        cov_rank: int = 16,
        num_samples: int = 5,
        return_samples: bool = True,
        **loss_kwargs
    ):
        super().__init__()

        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UnetDecoderWithUncertainty(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, cov_rank=cov_rank, num_samples=num_samples)

        self.loss_fn = VariationalLoss(**loss_kwargs)
        self.num_samples = num_samples
        self.return_samples = return_samples
        self.cov_rank = cov_rank

    def forward(self, x, targets = None):

        skips = self.encoder(x)
        mu, diag_var_out, cov_out = self.decoder(skips)

        if self.decoder.deep_supervision:
            mu = mu[0]

        logits = None
        loss, loss_attributes = None, None
        if self.return_samples:
            mu, loss, loss_attributes = self.sample_logits(
                mu, cov_out, diag_var_out,num_samples = self.num_samples, targets=targets
            )
        else:
            if targets is not None:
                loss, loss_attributes = self.loss_fn(targets, mu, diag_var_out)

        output = UncertaintyModelOutput(mu, cov_out, diag_var_out, logits, loss, loss_attributes)
        return output

    def sample_logits(self, mu, a, sigma, num_samples: int, targets=None):
        """
        Sample logits ~ N(mu, Σ), Σ = (B diag(a))(B diag(a))^T + diag(σ^2)
        Shapes:
          mu:    [B,C,D,H,W]
          a:     [B,r,D,H,W]
          sigma: [B,C,D,H,W]
          num_samples: int
          targets: [B,D,H,W]
        Returns:
          logits_mc: [K,B,C,D,H,W]
        """
        Bsz, C, D, H, W = mu.shape
        r = a.shape[1]
        if num_samples <= 0:
            return mu.unsqueeze(0)

        device = mu.device
        dtype = mu.dtype

        eps = torch.randn(num_samples, Bsz, C, D, H, W, device=device, dtype=dtype)
        diagpart = sigma.unsqueeze(0) * eps

        if self.cov_rank > 0:
            z = torch.randn(num_samples, Bsz, r, D, H, W, device=device, dtype=dtype)
            # Expand B to broadcast over spatial dims: [1,C,r,1,1,1]
            B_exp = self.decoder.cov_basis_mat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            lowrank = (B_exp * (a.unsqueeze(0).unsqueeze(2) * z.unsqueeze(2))).sum(dim=3)  # [K,B,C,D,H,W]
            logits = mu.unsqueeze(0) + lowrank + diagpart
        else:
            logits = mu.unsqueeze(0) + diagpart

        loss, attributes = 0., {}
        if targets is not None:
            for idx in range(num_samples):
                if idx == 0:
                    comb_loss, sample_attributes = self.loss_fn(logits[idx], targets, sigma)
                    attributes = {key: 0 for key in sample_attributes.keys()}
                else:
                    comb_loss, sample_attributes = self.loss_fn(logits[idx], targets)  # Only KL loss once

                loss += comb_loss
                attributes = {key: val + sample_attributes[key] for key, val in attributes.items()}

        return logits, loss, attributes


class UnetDecoderWithUncertainty(unet_dec.UNetDecoder):
    
    def __init__(self, 
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 cov_rank: int = 16,
                 num_samples: int = 5,
                 **loss_kwargs
                 ):

        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, conv_bias)

        self.cov_rank = cov_rank
        B_init = torch.randn(num_classes, cov_rank) * 0.02 if cov_rank > 0 else torch.zeros(num_classes, 0)
        self.cov_basis_mat = nn.Parameter(B_init, requires_grad=(cov_rank > 0))

        self.output_stage_meta_diag = {
            'in_channels': self.seg_layers[-1].in_channels,
            'out_channels': self.seg_layers[-1].out_channels,
            'kernel_size': self.seg_layers[-1].kernel_size,
            'stride': self.seg_layers[-1].stride,
            'padding': self.seg_layers[-1].padding
        }

        self.output_stage_meta_cov = self.output_stage_meta_diag.copy()
        self.output_stage_meta_cov['out_channels'] = cov_rank

        self.diag_head = self.encoder.conv_op(**self.output_stage_meta_diag)
        self.cov_head = self.encoder.conv_op(**self.output_stage_meta_cov)

        self.return_samples = True
        self.num_samples = num_samples


    def forward(self, skips, targets = None):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :param targets:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        diag_var_out, cov_out = None, None
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)

            is_last = s == len(self.stages) - 1
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
                if is_last:
                    diag_var_out = self.diag_head(x)
                    cov_out = self.cov_head(x)


            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
                diag_var_out = self.diag_head(x)
                cov_out = self.cov_head(x)

            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0], diag_var_out, cov_out
        else:
            return seg_outputs, diag_var_out, cov_out


class UncertaintySegLoss(nn.Module):
    def __init__(self,
                 lambda_ce=1.0,
                 lambda_dice=0.0,
                 lambda_nll=1.0,
                 lambda_kl=1e-4,
                 smooth=1e-6,
                 reduction="mean"):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_nll = lambda_nll
        self.lambda_kl = lambda_kl
        self.smooth = smooth
        self.reduction = reduction

    def dice_loss(self, probs, target_onehot):
        """
        probs: (B, C, D, H, W) softmax predictions
        target_onehot: (B, C, D, H, W) one-hot ground truth
        """
        dims = tuple(range(2, probs.ndim))  # spatial dims
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs**2 + target_onehot**2, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_score.mean()
        return dice_loss

    def forward(self, mu_logits, log_var, target, **kwargs):
        """
        mu_logits: (B, C, D, H, W) predicted mean logits
        log_var:   (B, C, D, H, W) predicted log-variances
        target:    (B, D, H, W) integer class labels
        """
        B, C, *spatial = mu_logits.shape
        N = B * torch.prod(torch.tensor(spatial, device=mu_logits.device))

        # ---- Cross Entropy Loss ----
        ce_loss = F.cross_entropy(mu_logits, target, reduction=self.reduction)

        # ---- Dice Loss ----
        probs = F.softmax(mu_logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=C).permute(0, -1, *range(1, len(spatial)+1)).float()
        dice_loss = self.dice_loss(probs, target_onehot)

        # ---- Negative Log Likelihood (Gaussian over logits) ----
        var = torch.exp(log_var)
        precision = 1.0 / var
        nll = 0.5 * torch.sum(
            (target_onehot - mu_logits)**2 * precision + log_var
        ) / N

        # ---- KL regularization on log-variance ----
        kl = 0.5 * torch.sum(var - 1.0 - log_var) / N

        # ---- Combine ----
        loss = (self.lambda_ce * ce_loss +
                self.lambda_dice * dice_loss +
                self.lambda_nll * nll +
                self.lambda_kl * kl)

        return loss, {
            "ce": ce_loss.item(),
            "dice": dice_loss.item(),
            "nll": nll.item(),
            "kl": kl.item()
        }


class VariationalLoss(nn.Module):
    def __init__(self,
                 lambda_ce=1.0,
                 lambda_dice=0.0,
                 lambda_nll=1.0,
                 lambda_kl=1e-4,
                 smooth=1e-6,
                 reduction="mean",
                 num_total = 100,
                 **kwargs):

        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_nll = lambda_nll
        self.lambda_kl = lambda_kl
        self.smooth = smooth
        self.reduction = reduction
        self.ce_forward = nn.CrossEntropyLoss(reduction=reduction)
        self.num_total = num_total

    def forward(self, logits, target, log_var = None):

        ce_loss = self.ce_forward(logits, target)
        dice_loss = self.dice_loss(F.softmax(logits, dim = 1), target, num_classes = 2)
        kl_loss = 0 if log_var is None else self.kl_loss(log_var, self.num_total)

        loss = self.lambda_ce * ce_loss + self.lambda_dice * dice_loss + self.lambda_kl * kl_loss
        return loss, {'cross_entropy': ce_loss.item(), 'dice': dice_loss.item(), 'kl_loss': kl_loss}


    @staticmethod
    def kl_loss(log_var, num_total = 1000):
        """
        Calculates kl div loss in approximate setting, log var is only diagonal to reduce computation
        :param log_var: predicted variances
        :param num_total: the total number of samples in training, so to act like a prior
        :return: KL Div
        """
        return 0.5 * torch.sum(torch.exp(log_var) - 1 - log_var) / num_total

    @staticmethod
    def dice_loss(probs, target, num_classes, eps=1e-6):
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersect = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice = (2. * intersect / (cardinality + eps)).mean()
        return 1. - dice


def get_model_from_base_kwargs(path_to_base, **kwargs):

    base_kwargs = pickle.load(open(path_to_base, 'rb'))
    base_kwargs['loss_kwargs'] = kwargs.pop('loss_kwargs', dict())
    for key, val in kwargs.items():
        base_kwargs[key] = val

    model = UnetWithUncertainty(**base_kwargs)
    return model



if __name__ == '__main__':

    path_to_base = r"C:\Users\pjtka\Downloads\info_dict.pkl"
    model = get_model_from_base_kwargs(path_to_base)
    model = model.to('cuda')
    data = torch.rand((1, 1, 128, 128, 128))
    # out = model(data.to('cuda'))
    targets = (torch.randn((1,128,128,128)) > 0).to('cuda').long()
    out = model(data.to('cuda'), targets)
    breakpoint()
















