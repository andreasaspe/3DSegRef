import torch.nn as nn
import torch 
import torch.nn.functional as F

class BaseSampler(nn.Module):

    def __init__(self,mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__()

        self.mu = mu 
        self.sigma = sigma
        self.low_rank_cov = low_rank_cov
        self.cov_basis_mat = cov_basis_mat
        self.weighting = weighting
    
    def sample(self, size = 1):
        
        if size > 1:
            samples = torch.cat([self.foward().unsqueeze(0) for _ in range(size)], dim = 0)
            return samples
        return self.forward()

    def forward(self,):
        pass


class SamplerUnweighted(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)

    
    def forward(self, ):

        """
        Sample MC logits from a Gaussian with low-rank + diagonal covariance.

        Distribution:
            logits ~ N(mu, Σ), with
            Σ = (B diag(a))(B diag(a))^T + diag(sigma^2)

        Args:
            mu:    [B,C,D,H,W]   mean logits
            a:     [B,r,D,H,W]   low-rank scales (voxel dependent)
            sigma: [B,C,D,H,W]   diagonal std dev
            B:     [C,r]         learned basis matrix (shared across voxels/batch)
            num_samples: int     number of Monte Carlo samples
            targets: [B,D,H,W]   (optional) target labels, unused here but kept for API compatibility

        Returns:
            logits_mc: [K,B,C,D,H,W]   sampled logits
        """
        Bsz, C, D, H, W = self.mu.shape
        r = self.low_rank_cov.shape[1]  # rank

        # -------------------------------------------------------
        # 1. Expand mean for sampling
        # -------------------------------------------------------
        mu_exp = self.mu.unsqueeze(0)  # [K,B,C,D,H,W]

        # -------------------------------------------------------
        # 2. Low-rank noise component
        # -------------------------------------------------------
        # eps_r ~ N(0, I), shape [K,B,r,D,H,W]
        eps_r = torch.randn(1, Bsz, r, D, H, W, device=self.mu.device, dtype=self.mu.dtype)

        # Scale by 'a': [K,B,r,D,H,W]
        scaled = eps_r * self.low_rank_cov.unsqueeze(0)

        # Reshape to [K,B,r,N] with N = D*H*W
        N = D * H * W
        scaled = scaled.view(1, Bsz, r, N)

        # Project into C with B:
        #   B: [C,r], scaled: [K,B,r,N]
        #   result: [K,B,C,N]
        low_rank = torch.matmul(self.cov_basis_mat, scaled)  

        # Reshape back to [K,B,C,D,H,W]
        low_rank_noise = low_rank.view(1, Bsz, C, D, H, W)

        # -------------------------------------------------------
        # 3. Diagonal noise component
        # -------------------------------------------------------
        eps_diag = torch.randn(1, Bsz, C, D, H, W, device=self.mu.device, dtype=self.mu.dtype)
        diag_noise = eps_diag * self.sigma.unsqueeze(0)

        # -------------------------------------------------------
        # 4. Combine all parts
        # -------------------------------------------------------
        logits_mc = mu_exp + low_rank_noise + diag_noise
        return logits_mc.squeeze(0)
    

class SamplerWeighted(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)
    

    def forward(self, ):
        """
        Sample MC logits from a Gaussian with low-rank + diagonal covariance.

            Distribution:
            logits ~ N(mu, Σ), with
            Σ = (B diag(a))(B diag(a))^T + diag(sigma^2)

        Args:
            mu:    [B,C,D,H,W]   mean logits
            a:     [B,r,D,H,W]   low-rank scales (voxel dependent)
            sigma: [B,C,D,H,W]   diagonal std dev
            B:     [C,r]         learned basis matrix (shared across voxels/batch)
            num_samples: int     number of Monte Carlo samples
            targets: [B,D,H,W]   (optional) target labels, unused here but kept for API compatibility

        Returns:
            logits_mc: [K,B,C,D,H,W]   sampled logits
        """

        num_samples = 1
        Bsz, C, D, H, W = self.mu.shape
        r = self.low_rank_cov.shape[1]  # rank

        weighting = self.weighting.unsqueeze(-1).unsqueeze(-1)

        cov_bases_selected = (weighting * self.cov_basis_mat.unsqueeze(0)).sum(dim = 1) # Here we sum over the different bases, for hard softmax only one value is none zero so silly to sum

        # -------------------------------------------------------
        # 1. Expand mean for sampling
        # -------------------------------------------------------
        mu_exp = self.mu.unsqueeze(0).expand(num_samples, -1, -1, -1, -1, -1)  # [K,B,C,D,H,W]

        # -------------------------------------------------------
        # 2. Low-rank noise component
        # -------------------------------------------------------
        # eps_r ~ N(0, I), shape [K,B,r,D,H,W]
        eps_r = torch.randn(num_samples, Bsz, r, D, H, W, device=self.mu.device, dtype=self.mu.dtype)

        # Scale by 'a': [K,B,r,D,H,W]
        scaled = eps_r * self.low_rank_cov.unsqueeze(0)

        # Reshape to [K,B,r,N] with N = D*H*W
        N = D * H * W
        scaled = scaled.view(num_samples, Bsz, r, N)

        # Project into C with B:
        #   B: [C,r], scaled: [K,B,r,N]
        #   result: [K,B,C,N]
        low_rank = torch.matmul(cov_bases_selected, scaled)  

        # Reshape back to [K,B,C,D,H,W]
        low_rank_noise = low_rank.view(num_samples, Bsz, C, D, H, W)

        # -------------------------------------------------------
        # 3. Diagonal noise component
        # -------------------------------------------------------
        eps_diag = torch.randn(num_samples, Bsz, C, D, H, W, device=self.mu.device, dtype=self.mu.dtype)
        diag_noise = eps_diag * self.sigma.unsqueeze(0)

        # -------------------------------------------------------
        # 4. Combine all parts
        # -------------------------------------------------------
        logits_mc = mu_exp + low_rank_noise + diag_noise
        return logits_mc
    



class ConvexSamplerWeighted(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)
    
        self.eps = 1e-8 
        self.per_basis_a = False

    def forward(self, ):
        """
        Sample Monte-Carlo logits from
        mu: [B, C, D, H, W]
        log_var: [B, C, D, H, W]  (diagonal log-variance)
        cov_bases: [N, C, r]       (N basis matrices)
        weights: either [B, N] (per-volume) or [B, N, D, H, W] (voxel-wise)
        a: either None, or
            - if per_basis_a=True: [B, N, r, D, H, W]  (a_n(x) per basis)
            - otherwise: [B, r, D, H, W]              (shared across bases)
        num_samples: int, number of MC samples to produce
        Returns:
        samples: tensor of shape [num_samples, B, C, D, H, W]
        """
        device = self.mu.device
        batch, C, D, H, W = self.mu.shape
        N, _, r = self.cov_basis_mat.shape  # cov_bases: [N, C, r]
     
        # Prepare weights -> make them voxel-wise [B, N, D, H, W]
        if self.weighting.dim() == 2:  # [B, N]
            w = self.weighting.view(batch, N, 1, 1, 1).expand(-1, -1, D, H, W)
        else:
            # assume already [B, N, D, H, W]
            w = self.weighting
        # small floor to avoid zero sqrt
        w = w.clamp(min=0.0)
        w_sqrt = torch.sqrt(w + self.eps)  # [B, N, D, H, W]

        # Pre-expand cov_bases for broadcast: [1, N, C, r, 1, 1, 1]
        cov_bases_exp = self.cov_basis_mat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # diag noise term
        eps_diag = torch.randn_like(self.mu, device=device)  # [B,C,D,H,W]
        diag_term = self.sigma * eps_diag

        # ----- low-rank mixture term -----
        # Sample independent z_n for each basis
        # eps_low: [B, N, r, D, H, W]
        eps_low = torch.randn(batch, N, r, D, H, W, device=device)

        
        if self.per_basis_a:
            # a is [B, N, r, D, H, W]; multiply elementwise
            assert self.low_rank_cov.shape == (batch, N, r, D, H, W)
            scaled = self.low_rank_cov * eps_low
        else:
            # a is [B, r, D, H, W] -> broadcast to [B, N, r, D, H, W]
            assert self.low_rank_cov.shape == (batch, r, D, H, W)
            scaled = eps_low * self.low_rank_cov.unsqueeze(1)  # broadcast in N

        # Now compute B_n @ (a_n * z_n) for every n, batch, voxel.
        # Do this via broadcasting (no einsum):
        # - cov_bases_exp: [1, N, C, r, 1,1,1]
        # - scaled_unsq  : [B, N, 1, r, D,H,W]
        scaled_unsq = scaled.unsqueeze(2)  # [B, N, 1, r, D, H, W]
        # multiply -> [B, N, C, r, D, H, W]
        mult = (cov_bases_exp * scaled_unsq)  # broadcasting
        # sum over r -> [B, N, C, D, H, W]
        per_basis_lowrank = mult.sum(dim=3)

        # weight each basis contribution by sqrt(p_n(x)) and sum over N:
        # w_sqrt: [B, N, D, H, W] -> unsqueeze class dim to multiply
        w_sqrt_unsq = w_sqrt.unsqueeze(2)  # [B, N, 1, D, H, W]
        # multiply and sum over N -> [B, C, D, H, W]
        low_rank_sum = (w_sqrt_unsq * per_basis_lowrank).sum(dim=1)

        # final sample
        sample = self.mu + diag_term + low_rank_sum  # [B, C, D, H, W]
        return sample







