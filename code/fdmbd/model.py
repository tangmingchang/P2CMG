"""New codec-conditioned multiband rectified-flow decoder, not recovered S0.

STFT -> four frequency slices -> shared temporal U-Net -> explicit cross-band
attention -> concatenation -> ISTFT. EnCodec is a frozen conditional prior.
The deployed decoder needs only codes, a codec, trained weights, and a seed.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    sample_rate: int = 32000
    n_fft: int = 1024
    hop: int = 256
    bands: int = 4
    latent_channels: int = 128
    width: int = 48
    levels: int = 4
    heads: int = 4
    prior_noise: float = 0.05
    cross_band: bool = True
    energy_balance: bool = True


class SpectralTransform:
    def __init__(self, config=Config()):
        self.config = config
        self.bins = config.n_fft // 2 + 1
        self.per_band = math.ceil(self.bins / config.bands)
        self.channels = 2 * self.per_band

    def analysis(self, wave):
        c = self.config
        return torch.stft(wave.float(), c.n_fft, c.hop,
                          window=torch.hann_window(c.n_fft, device=wave.device),
                          center=True, return_complex=True)

    def synthesis(self, spectrum, length):
        c = self.config
        return torch.istft(spectrum, c.n_fft, c.hop,
                           window=torch.hann_window(c.n_fft, device=spectrum.device),
                           center=True, length=length)

    def pack(self, spectrum):
        # [B,F,T] complex -> [B,K,2*ceil(F/K),T]; real/imag interleaved.
        real = torch.view_as_real(spectrum).permute(0, 1, 3, 2)
        real = F.pad(real, (0, 0, 0, 0, 0, self.per_band * self.config.bands - self.bins))
        return real.reshape(spectrum.shape[0], self.config.bands, self.channels, spectrum.shape[-1])

    def unpack(self, packed):
        b, _, _, t = packed.shape
        real = packed.reshape(b, self.config.bands * self.per_band, 2, t)[:, :self.bins]
        return torch.view_as_complex(real.permute(0, 1, 3, 2).contiguous())

    def mask(self, device):
        real = torch.arange(self.config.bands * self.per_band, device=device) < self.bins
        return real.reshape(1, self.config.bands, self.per_band, 1).repeat_interleave(2, dim=2).float()


class ResBlock(nn.Module):
    def __init__(self, channels, embedding):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.emb = nn.Linear(embedding, channels)

    def forward(self, x, emb):
        y = self.conv1(F.silu(self.norm1(x))) + self.emb(F.silu(emb))[:, :, None]
        return x + self.conv2(F.silu(self.norm2(y)))


class CrossBandAttention(nn.Module):
    """At each downsampled time, Q/K/V attend across the four band slots."""
    def __init__(self, channels, heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)

    def forward(self, x, batch, bands):
        _, channels, frames = x.shape
        slots = x.reshape(batch, bands, channels, frames).permute(0, 3, 1, 2).reshape(batch * frames, bands, channels)
        q = self.norm(slots)
        slots = slots + self.attn(q, q, q, need_weights=False)[0]
        return slots.reshape(batch, frames, bands, channels).permute(0, 2, 3, 1).reshape(batch * bands, channels, frames)


class MultibandFlow(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        transform = SpectralTransform(config)
        self.transform = transform
        widths = [config.width * (i + 1) for i in range(config.levels)]
        embedding = widths[-1]
        self.time_mlp = nn.Sequential(nn.Linear(32, embedding), nn.SiLU(), nn.Linear(embedding, embedding))
        self.band_embedding = nn.Embedding(config.bands, embedding)
        self.input = nn.Conv1d(2 * transform.channels, widths[0], 1)
        self.condition = nn.Conv1d(config.latent_channels, widths[0], 1)
        self.encoders = nn.ModuleList([ResBlock(w, embedding) for w in widths])
        self.down = nn.ModuleList([nn.Conv1d(a, b, 4, stride=2, padding=1) for a, b in zip(widths[:-1], widths[1:])])
        self.band_attention = CrossBandAttention(widths[-1], config.heads)
        self.up = nn.ModuleList([nn.Conv1d(widths[i+1] + widths[i], widths[i], 1) for i in reversed(range(config.levels-1))])
        self.decoders = nn.ModuleList([ResBlock(widths[i], embedding) for i in reversed(range(config.levels-1))])
        self.output = nn.Conv1d(widths[0], transform.channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, t, codec_spectrum, latent):
        b, k, channels, frames = state.shape
        assert k == self.config.bands and channels == self.transform.channels
        frequency = torch.exp(torch.linspace(0, math.log(10000), 16, device=t.device))
        phase = t[:, None] * frequency[None]
        temporal = self.time_mlp(torch.cat([phase.sin(), phase.cos()], dim=1))
        band_ids = torch.arange(k, device=state.device)
        emb = (temporal[:, None] + self.band_embedding(band_ids)[None]).reshape(b*k, -1)
        x = self.input(torch.cat([state, codec_spectrum], dim=2).reshape(b*k, 2*channels, frames))
        condition = self.condition(latent).unsqueeze(1).expand(-1, k, -1, -1).reshape(b*k, -1, frames)
        x = x + condition
        skips = []
        for i, block in enumerate(self.encoders):
            x = block(x, emb)
            skips.append(x)
            if i < len(self.down):
                x = self.down[i](x)
        if self.config.cross_band:
            x = self.band_attention(x, b, k)
        for projection, block, skip in zip(self.up, self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = block(projection(torch.cat([x, skip], dim=1)), emb)
        return self.output(F.silu(x)).reshape(b, k, channels, frames) * self.transform.mask(state.device)


def flow_target(clean, prior, noise, t, sigma):
    start = prior + sigma * noise
    velocity = clean - start
    state = start + t[:, None, None, None] * velocity
    return state, velocity


@torch.no_grad()
def sample_spectrum(model, prior, latent, scale, steps, generator):
    c = model.config
    mask = model.transform.mask(prior.device)
    z = prior + c.prior_noise * torch.randn(prior.shape, device=prior.device, generator=generator) * mask
    for i in range(steps):
        t = torch.full((len(z),), i / steps, device=z.device)
        z = (z + model(z, t, prior, latent) / steps) * mask
    return model.transform.unpack(z * scale)


@torch.no_grad()
def decode_codes(codec, model, codes, scale, length, steps=10, seed=0):
    """Token-only deployment API. No target waveform or painting label input."""
    device = next(model.parameters()).device
    codes = codes.to(device).long()
    baseline = codec.decode(codes, None).mean(1)[:, :length]
    latent = codec.decode_latent(codes)
    raw = model.transform.analysis(baseline)
    prior = model.transform.pack(raw) / scale
    latent = F.interpolate(latent, size=prior.shape[-1], mode='linear', align_corners=False)
    generator = torch.Generator(device=device).manual_seed(seed)
    predicted = sample_spectrum(model, prior, latent, scale, steps, generator)
    return model.transform.synthesis(predicted, length), baseline


def config_dict(model):
    return asdict(model.config)
