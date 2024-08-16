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

    On top of TFGridNetV2, TFGridNetV3 slightly modifies the internal architecture
    to make the model sampling-frequency-independent (SFI). This is achieved by
    making all network layers independent of the input time and frequency dimensions.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in ICASSP, 2023.

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
        mamba_blocks=6,
        mamba_layers=6,
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
        #self.num_tscblocks = 4
        #self.num_tscblocks = 14
        #self.num_tscblocks = 10
        #self.num_tscblocks = 20
        self.num_tscblocks = mamba_blocks
        self.dense_encoder = DenseEncoder(in_channel=2)

        self.TSMamba = nn.ModuleList([])
        for i in range(self.num_tscblocks):
            self.TSMamba.append(TSMambaBlock(mamba_layers))

        #self.laynorms = nn.ModuleList([])
        #for i in range(self.num_tscblocks):
        #    self.laynorms.append(RMSNorm(64))
        
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

        noisy_mag = torch.abs(input)
        noisy_pha = torch.angle(input)
        noisy_mag = torch.pow(noisy_mag, 0.3)


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
    #d_state = 32
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
    def __init__(self, in_channels, n_layer):
        super(MambaBlock, self).__init__()
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
    def __init__(self, n_layer):
        super(TSMambaBlock, self).__init__()

        self.time_mamba = MambaBlock(in_channels=64, n_layer=n_layer)
        self.freq_mamba = MambaBlock(in_channels=64, n_layer=n_layer)

        self.tlinear = nn.ConvTranspose1d(
            64 * 2, 64, 1, stride=1
        )

        self.flinear = nn.ConvTranspose1d(
            64 * 2, 64, 1, stride=1
        )
        #LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),

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
        #self.num_tscblocks = 4
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
