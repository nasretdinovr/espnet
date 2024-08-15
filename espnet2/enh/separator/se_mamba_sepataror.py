import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.get_layer_from_string import get_layer

from functools import partial
from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class SEMambaSeparator(AbsSeparator):
    """Offline SEMambaSeparator.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.
    Specifically, use:
        std_ = std(mix)
        mix = mix / std_
        tgt = tgt / std_

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNetV3 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_attn_qk_output_channel: output channels of point-wise conv2d for getting
            key and query
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNetV3 model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        input_dim,
        n_srcs=2,
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_qk_output_channel=4,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert self.n_imics == 1, self.n_imics

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        """
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)
        """
        # SEMamba
        self.num_tscblocks = 4
        self.dense_encoder = DenseEncoder(in_channel=2)

        self.TSMamba = nn.ModuleList([])
        for i in range(self.num_tscblocks):
            self.TSMamba.append(TSMambaBlock())
        
        self.mask_decoder = MaskDecoder(out_channel=1)
        self.phase_decoder = PhaseDecoder(out_channel=1)

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, T, F]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """

        # B, 2, T, (C,) F
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=1)
        else:
            assert input.size(-1) == 2, input.shape
            feature = input.moveaxis(-1, 1)

        assert feature.ndim == 4, "Only single-channel mixture is supported now"

        n_batch, _, n_frames, n_freqs = feature.shape

        """
        batch = self.conv(feature)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(input, (batch[:, :, 0], batch[:, :, 1]))

        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()
        """

        #spectrum = rearrange(input, 'B T F -> B F T')

        #noisy_mag = torch.abs(spectrum)
        #noisy_pha = torch.angle(spectrum)
        #noisy_mag = torch.pow(noisy_mag, 0.3)

        real_part = input.real
        imag_part = input.imag
        noisy_mag = torch.sqrt( real_part.pow(2) + imag_part.pow(2))
        noisy_pha = torch.atan2( imag_part, real_part )
        noisy_mag = torch.pow( noisy_mag, 0.3 )
        
        #print(noisy_mag.shape) # [B, T, F]

        #noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        #noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(1)  # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]

        B, C, T, F = x.shape
        if F % 2 == 0:
            zeros = torch.zeros(B, C, T, 1, device=x.device)
            x = torch.cat((x, zeros), dim=-1)

        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSMamba[i](x)
        
        #denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_mag = (self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        denoised_mag = torch.pow(denoised_mag, (1.0/0.3))

        if F % 2 == 0:
            denoised_mag = denoised_mag[:, :F, :]
            denoised_pha = denoised_pha[:, :F, :]

        denoised_com = rearrange(torch.complex( denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha) ), 'B F T -> B T F')

        return [denoised_com], ilens, OrderedDict()

    @property
    def num_spk(self):
        return self.n_srcs


class GridNetV3Block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"

        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        # use constant E not to be dependent on the number of frequency bins
        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetV2Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        batch = inter_rnn

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](batch))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](batch))  # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x


# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False, 
    ):
    d_state = 16
    d_conv = 4
    expand = 4
    norm_epsilon = 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self, in_channels):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels) for i in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels) for i in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)


"""
class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        
        return residual
"""
    

class DenseBlock(nn.Module):
    def __init__(self, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(64*(i+1), 64, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(64, affine=True),
                nn.PReLU(64)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, in_channel):
        super(DenseEncoder, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, (1, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64))

        self.dense_block = DenseBlock(depth=4) # [b, 64, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 3), (1, 2)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (1, 3), (1, 2)),
            nn.Conv2d(64, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        #self.lsigmoid = LearnableSigmoid_2d_SFI(h.n_fft//2+1, beta=h.beta)
        self.lsigmoid = nn.Softplus()

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return torch.clamp(x, min=1e-10)


class PhaseDecoder(nn.Module):
    def __init__(self, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (1, 3), (1, 2)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        self.phase_conv_r = nn.Conv2d(64, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(64, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class TSMambaBlock(nn.Module):
    def __init__(self):
        super(TSMambaBlock, self).__init__()

        self.time_mamba = MambaBlock(in_channels=64)
        self.freq_mamba = MambaBlock(in_channels=64)

        self.tlinear = nn.ConvTranspose1d(
            64 * 2, 64, 1, stride=1
        )

        self.flinear = nn.ConvTranspose1d(
            64 * 2, 64, 1, stride=1
        )

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        #fesfesf
        return x


class SEMamba(nn.Module):
    def __init__(self, h):
        super(SEMamba, self).__init__()
        self.num_tscblocks = 4
        self.dense_encoder = DenseEncoder(in_channel=2)

        self.TSMamba = nn.ModuleList([])
        for i in range(self.num_tscblocks):
            self.TSMamba.append(TSMambaBlock(h))
        
        self.mask_decoder = MaskDecoder(out_channel=1)
        self.phase_decoder = PhaseDecoder(out_channel=1)


    #def forward(self, noisy_mag, noisy_pha): # [B, F, T]
    def forward(self, x): # [B, F, T]
        mix_std_ = torch.std(x, dim=(1), keepdim=True)  # [B, 1]
        x = x / mix_std_  # RMS normalization

        n_sample = x.shape[1]
        ilens = torch.ones(x.shape[0], dtype=torch.long, device=x.device) * n_sample
        #spectrum, flens = self.enc(x, ilens)
        spectrum, flens = self.enc(x, ilens, 48000)
        spectrum = rearrange(spectrum, 'B T F -> B F T')

        noisy_mag = torch.abs(spectrum)
        noisy_pha = torch.angle(spectrum)
        noisy_mag = torch.pow(noisy_mag, 0.3)

        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]

        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSMamba[i](x)
        
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        
        new_mag = torch.pow(denoised_mag, (1.0/0.3))
        denoised_tcom = rearrange(torch.complex( new_mag * torch.cos(denoised_pha), new_mag * torch.sin(denoised_pha) ), 'B F T -> B T F')
        #denoised_audio, wav_lens = self.dec( denoised_tcom, ilens )
        denoised_audio, wav_lens = self.dec( denoised_tcom, ilens, 48000 )
        denoised_audio = denoised_audio * mix_std_  # reverse the RMS normalization

        return denoised_mag, denoised_pha, denoised_com, denoised_audio


def phase_losses(phase_r, phase_g, h):
    B, F, T = phase_r.size()

    #dim_freq = h.n_fft // 2 + 1
    #dim_time = phase_r.size(-1)

    dim_freq, dim_time = F, T

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdims=True)
    return norm

def si_snr_loss(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return -1 * torch.mean(snr)

def freq_delta_spec_loss(s1, s2):

    s1_delta_spec = s1[..., 1:] - s1[..., :-1]
    s2_delta_spec = s2[..., 1:] - s2[..., :-1]

    delta_spec_loss = torch.mean(torch.abs(s1_delta_spec - s2_delta_spec))
    return delta_spec_loss

def combined_losses(s1, s2, alpha=0.5, eps=1e-8):
    return si_snr_loss(s1, s2, eps) + freq_delta_spec_loss(s1, s2)

def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class LearnableSigmoid_2d_SFI(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.default_in_features = in_features
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        B, C, L = x.size()

        if C != self.default_in_features:
            slope_interpolated = nn.functional.interpolate(self.slope.unsqueeze(0).permute(0, 2, 1), size=C, mode='linear', align_corners=False).squeeze(0).permute(1, 0)
        else:
            slope_interpolated = self.slope

        return self.beta * torch.sigmoid(slope_interpolated * x)
