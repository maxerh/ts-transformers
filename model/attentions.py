import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops import rearrange
#from flash_attn.modules.mha import MHA


# Standard Multihead Attention
class StandardAttention(nn.Module):
    # https://arxiv.org/pdf/1706.03762
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

# Chunked Attention (Vectorized using unfold)
class ChunkedAttention(nn.Module):
    # https://arxiv.org/pdf/1712.05382
    def __init__(self, dim, window_size=128, overlap=32, heads=8):
        """
        Applies attention on overlapping chunks extracted from the sequence.
        Args:
            dim: Input feature dimension.
            window_size: Length of each chunk (equals chunk_size.
            overlap: Amount of overlapping time steps between consecutive chunks.
            heads: Number of attention heads.
        """
        super().__init__()
        assert window_size > overlap, "Chunk size must be larger than the overlap."
        self.dim = dim
        self.chunk_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            x_recon: [B, T, D] after processing with chunked attention.
        """
        B, T, D = x.shape
        # Calculate padded sequence length so that sliding windows cover the sequence.
        num_chunks = math.ceil((T - self.overlap) / self.stride)
        T_pad = num_chunks * self.stride + self.overlap
        pad_len = T_pad - T
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Rearrange x for unfolding: treat time as spatial dimension.
        # x_t shape: [B, D, T_pad, 1]
        x_t = x.transpose(1, 2).unsqueeze(-1)
        # Extract overlapping patches (chunks) along the time axis.
        chunks = F.unfold(
            x_t,
            kernel_size=(self.chunk_size, 1),
            stride=(self.stride, 1)
        )  # shape: [B, D*chunk_size, L]
        L = chunks.shape[-1]
        # Reshape to get individual chunks.
        chunks = chunks.view(B, D, self.chunk_size, L).permute(0, 3, 2, 1)  # [B, L, chunk_size, D]
        chunks = chunks.reshape(B * L, self.chunk_size, D)  # [B*L, chunk_size, D]

        # Apply attention to each chunk.
        attn_out, _ = self.attn(chunks, chunks, chunks)  # [B*L, chunk_size, D]
        attn_out = attn_out.view(B, L, self.chunk_size, D).permute(0, 3, 2, 1)  # [B, D, chunk_size, L]
        attn_out = attn_out.reshape(B, D * self.chunk_size, L)  # [B, D*chunk_size, L]

        # Reconstruct the sequence using fold.
        x_recon = F.fold(
            attn_out,
            output_size=(T_pad, 1),
            kernel_size=(self.chunk_size, 1),
            stride=(self.stride, 1)
        )  # [B, D, T_pad, 1]
        x_recon = x_recon.squeeze(-1).transpose(1, 2)  # [B, T_pad, D]

        # Compute divisor map to average overlapping regions.
        ones = torch.ones(B, self.chunk_size, L, device=x.device)  # shape: [B, chunk_size, L]
        divisor = F.fold(
            ones,
            output_size=(T_pad, 1),
            kernel_size=(self.chunk_size, 1),
            stride=(self.stride, 1)
        )  # shape: [B, 1, T_pad, 1]
        divisor = divisor.squeeze(1)  # now shape: [B, T_pad, 1]

        # Divide to average overlapping regions.
        x_recon = x_recon / (divisor + 1e-6)
        return x_recon[:, :T, :]


# Swin Attention (Windowed local attention)
class SwinAttention(nn.Module):
    # https://arxiv.org/pdf/2103.14030
    def __init__(self, dim, window_size=8, shift_size=0, heads=8):
        """
        Args:
          dim: Dimension of the input features.
          window_size: Number of time steps per window.
          shift_size: How much to shift windows (typically set to half the window size).
          heads: Number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.heads = heads

        # Linear projections for Q, K, V and output
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Learnable relative positional bias.
        # For 1D windows, the relative distance index spans [-(window_size-1), window_size-1]
        self.relative_bias = nn.Parameter(torch.zeros(2 * window_size - 1, heads))
        self._init_relative_bias()

    def _init_relative_bias(self):
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def window_partition(self, x):
        """
        Partition sequence into non-overlapping windows.
        Args:
          x: [B, T, D]
        Returns:
          windows: [B * n_windows, window_size, D] and padded length
        """
        B, T, D = x.shape
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        T_pad = T + pad_len

        # If shift is applied, shift the sequence:
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        # Partition: reshape to windows.
        n_windows = T_pad // self.window_size
        x = x.view(B, n_windows, self.window_size, D)
        windows = x.reshape(-1, self.window_size, D)  # [B * n_windows, window_size, D]
        return windows, T_pad

    def window_reverse(self, windows, B, T_pad):
        """
        Reverse the window partition into the original sequence shape.
        Args:
          windows: [B * n_windows, window_size, D]
          B, T_pad: original batch and padded length
        Returns:
          x: [B, T_pad, D]
        """
        n_windows = T_pad // self.window_size
        x = windows.view(B, n_windows, self.window_size, -1)
        x = x.view(B, T_pad, -1)
        # Reverse shift if needed.
        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        return x

    def forward(self, x):
        """
        Args:
          x: [B, T, D]
        Returns:
          out: [B, T, D]
        """
        B, T, D = x.shape
        windows, T_pad = self.window_partition(x)  # [B*n_windows, window_size, D]
        n_windows = windows.shape[0]

        # Compute Q, K, V
        qkv = self.qkv(windows)  # [B*n_windows, window_size, 3*D]
        qkv = qkv.reshape(n_windows, self.window_size, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # shape: [3, B*n_windows, heads, window_size, D//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B*n_windows, heads, window_size, D//heads]

        scale = (D // self.heads) ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B*n_windows, heads, window_size, window_size]

        # Create relative position bias
        coords = torch.arange(self.window_size, device=x.device)
        relative_coords = coords[None, :] - coords[:, None]  # [window_size, window_size]
        relative_coords += self.window_size - 1  # shift range to [0, 2*window_size-2]
        bias = self.relative_bias[relative_coords.view(-1)]
        bias = bias.view(self.window_size, self.window_size, self.heads).permute(2, 0, 1)  # [heads, window_size, window_size]
        attn_scores = attn_scores + bias.unsqueeze(0)  # add bias to each window

        attn_probs = F.softmax(attn_scores, dim=-1)  # [B*n_windows, heads, window_size, window_size]

        attn_out = attn_probs @ v  # [B*n_windows, heads, window_size, D//heads]
        attn_out = attn_out.transpose(1, 2).reshape(n_windows, self.window_size, D)
        out_windows = self.proj(attn_out)  # [B*n_windows, window_size, D]

        # Merge windows back to sequence
        out = self.window_reverse(out_windows, B, T_pad)
        # Remove any extra padding
        out = out[:, :T, :]
        return out

# Focal Attention (Efficient Masking)
class FocalAttention(nn.Module):
    # https://arxiv.org/pdf/2107.00641
    def __init__(self, dim, radius=4, heads=8):
        """
        A version of focal attention that focuses only on a local window.
        Args:
            dim: Feature dimension.
            radius: The local neighborhood radius.
            heads: Number of attention heads.
        """
        super().__init__()
        self.heads = heads
        self.radius = radius
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads

        qkv = self.qkv(x)  # [B, T, 3*D]
        qkv = qkv.reshape(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, H, T, D/H]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Create mask for local neighborhood (distance > radius: -inf)
        idx = torch.arange(T, device=x.device)
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.radius
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class LinearAttention(nn.Module):
    # https://arxiv.org/pdf/2006.16236
    def __init__(self, dim: int, heads: int = 8, eps: float = 1e-6):
        """
        Linear attention module using kernel-based self-attention.

        Args:
            dim: The input feature dimension.
            heads: Number of attention heads.
            eps: A small value to avoid division by zero.
        """
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.eps = eps
        self.head_dim = dim // heads
        assert self.heads * self.head_dim == dim, "dim must be divisible by heads"

        # Learnable linear projections for Q, K, V.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear feature map: here we use ELU + 1 as suggested in linear attention literature.
        """
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, dim].
        Returns:
            Tensor of shape [B, T, dim] after applying linear attention.
        """
        B, T, D = x.shape

        # Compute linear projections.
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)  # [B, T, D]
        v = self.v_proj(x)  # [B, T, D]

        # Apply the nonlinear feature map.
        q = self.phi(q)  # [B, T, D]
        k = self.phi(k)  # [B, T, D]

        # Reshape for multi-head attention.
        # New shape: [B, heads, T, head_dim]
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Compute kv as a weighted sum over time for each head.
        # kv shape: [B, heads, head_dim, head_dim]
        kv = torch.einsum("bhtd,bhtv->bhdv", k, v)

        # Compute the normalization factor.
        # First, sum K over time -> shape: [B, heads, head_dim]
        k_sum = k.sum(dim=2)
        # Denom: for each query in each head, compute dot product with K_sum.
        # Shape: [B, heads, T]
        denom = torch.einsum("bhtd,bhd->bht", q, k_sum)

        # Compute the attention output.
        # Numerator: Q times KV -> shape: [B, heads, T, head_dim]
        out = torch.einsum("bhtd,bhdv->bhtv", q, kv)
        # Normalize: divide each output by the appropriate denominator.
        out = out / (denom.unsqueeze(-1) + self.eps)

        # Reshape back to [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class SparseAttention(nn.Module):
    # https://arxiv.org/abs/2502.11089
    def __init__(self, dim, heads=8, window_size=16, global_stride=16):
        """
        SparseAttention:
          - Local window attention of size window_size
          - Strided global attention every global_stride tokens
        Complexity: O(B·T·(window_size + T/global_stride)·dim) :contentReference[oaicite:0]{index=0}
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window = window_size
        self.stride = global_stride

        # fused QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads
        hd = D // H

        # 1) compute Q, K, V
        qkv = self.qkv(x)  # [B, T, 3D]
        qkv = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)  # [3, B, H, T, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, T, hd]

        # 2) LOCAL attention via unfold
        # reshape so time is spatial
        k_t = k.permute(0, 1, 3, 2).reshape(B * H, hd, T).unsqueeze(-1)  # [BH,hd,T,1]
        v_t = v.permute(0, 1, 3, 2).reshape(B * H, hd, T).unsqueeze(-1)

        # pad so that unfold yields L = T
        pad = self.window - 1
        k_t = F.pad(k_t, (0, 0, pad, 0))  # pad time dim on left
        v_t = F.pad(v_t, (0, 0, pad, 0))

        # unfold into local windows
        patches_k = F.unfold(k_t, (self.window, 1), stride=(1, 1))  # [BH, hd*W, T]
        patches_v = F.unfold(v_t, (self.window, 1), stride=(1, 1))

        # now L = T
        L = patches_k.shape[-1]

        # reshape into [B,H,T,W,hd]
        patches_k = patches_k.view(B, H, hd * self.window, L)
        patches_k = patches_k.view(B, H, self.window, hd, L).permute(0, 1, 4, 2, 3)
        patches_v = patches_v.view(B, H, hd * self.window, L)
        patches_v = patches_v.view(B, H, self.window, hd, L).permute(0, 1, 4, 2, 3)

        # proceed with attention
        q_exp = q.unsqueeze(-2)  # [B,H,T,1,hd]
        attn_local = (q_exp * patches_k).sum(-1) / (hd ** 0.5)  # now shapes match [B,H,T,W]
        attn_local = F.softmax(attn_local, dim=-1)  # [B, H, T, W]
        out_local = (attn_local.unsqueeze(-1) * patches_v).sum(-2)  # [B, H, T, hd]

        # 3) GLOBAL strided attention
        idx = torch.arange(0, T, self.stride, device=x.device)
        k_global = k[:, :, idx, :]  # [B, H, Tg, hd]
        v_global = v[:, :, idx, :]  # [B, H, Tg, hd]
        attn_global = torch.einsum("bhtd,bhgd->bhtg", q, k_global) / (hd ** 0.5)  # [B, H, T, Tg]
        attn_global = F.softmax(attn_global, dim=-1)  # [B, H, T, Tg]
        out_global = torch.einsum("bhtg,bhgd->bhtd", attn_global, v_global)  # [B, H, T, hd]

        # combine and project
        out = out_local + out_global  # [B, H, T, hd]
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)  # [B, T, D]
        return self.proj(out)


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: pads input on left so output at t
    depends only on inputs ≤ t.
    """

    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # No padding here; we will pad manually
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        # x: [B, C, T]
        pad = self.kernel_size - 1
        # pad = (left, right)
        x = F.pad(x, (pad, 0))
        return self.conv(x)

class LogSparseAttention(nn.Module):
    # https://arxiv.org/abs/1907.00235
    def __init__(self, dim, heads=8, local_window=16, num_exps=8):
        """
        LogSparseAttention (Li et al., NeurIPS 2019):
          - Local window of size W (pad to get T windows)
          - Exponential jumps: attend to positions i-2^k for k=0…num_exps-1
        Complexity: O(B·T·(W + num_exps)·dim)
        """
        super().__init__()
        assert local_window > 0, "local_window must be positive"
        self.dim = dim
        self.heads = heads
        self.W = local_window
        self.E = num_exps
        self.hd = dim // heads
        assert self.hd * heads == dim, "dim must be divisible by heads"

        # causal convolutions for Q and K
        self.q_conv = CausalConv1d(dim, dim, kernel_size=self.W)
        self.k_conv = CausalConv1d(dim, dim, kernel_size=self.W)
        # pointwise projection for V
        self.v_proj = nn.Linear(dim, dim)
        # output projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        B, T, D = x.shape
        H, W, E, hd = self.heads, self.W, self.E, self.hd

        # 1) QKV via conv + linear
        # convert to [B, D, T] for Conv1d
        x_t = x.transpose(1, 2)
        q = self.q_conv(x_t).transpose(1, 2)  # [B, T, D]
        k = self.k_conv(x_t).transpose(1, 2)  # [B, T, D]
        v = self.v_proj(x)                   # [B, T, D]

        # split into heads: [B, H, T, hd]
        def split_heads(z):
            return z.view(B, T, H, hd).transpose(1, 2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # 2) Local window attention via unfold (pad left)
        pad = W - 1
        k_t = k.permute(0,1,3,2).reshape(B*H, hd, T).unsqueeze(-1)  # [B*H, hd, T, 1]
        v_t = v.permute(0,1,3,2).reshape(B*H, hd, T).unsqueeze(-1)
        k_t = F.pad(k_t, (0,0, pad, 0))
        v_t = F.pad(v_t, (0,0, pad, 0))
        patches_k = F.unfold(k_t, (W,1), stride=(1,1))  # [B*H, hd*W, T]
        patches_v = F.unfold(v_t, (W,1), stride=(1,1))
        patches_k = patches_k.view(B, H, hd*W, T).view(B, H, W, hd, T).permute(0,1,4,2,3)
        patches_v = patches_v.view(B, H, hd*W, T).view(B, H, W, hd, T).permute(0,1,4,2,3)

        q_exp = q.unsqueeze(-2)  # [B, H, T, 1, hd]
        attn_l = (q_exp * patches_k).sum(-1) / math.sqrt(hd)  # [B, H, T, W]
        attn_l = F.softmax(attn_l, dim=-1)
        out_l = (attn_l.unsqueeze(-1) * patches_v).sum(-2)    # [B, H, T, hd]

        # 3) Exponential jumps
        out_e = torch.zeros_like(out_l)
        for e in range(E):
            shift = 2 ** e
            if shift >= T:
                break
            k_s = k.roll(-shift, dims=2)
            v_s = v.roll(-shift, dims=2)
            score = (q * k_s).sum(-1) / math.sqrt(hd)  # [B, H, T]
            alpha = F.softmax(score, dim=-1).unsqueeze(-1)
            out_e = out_e + alpha * v_s                # [B, H, T, hd]

        # 4) Combine heads and project
        out = out_l + out_e                            # [B, H, T, hd]
        out = out.transpose(1, 2).reshape(B, T, D)      # [B, T, D]
        return self.proj(out)

# FlashAttention (Requires flash-attn installed)
#class FlashAttention(nn.Module):
#   # https://arxiv.org/pdf/2205.14135
#    def __init__(self, dim, heads=8):
#        super().__init__()
#        self.attn = MHA(embed_dim=dim, num_heads=heads, bias=False, dropout=0.0)
#
#    def forward(self, x):
#        return self.attn(x)


