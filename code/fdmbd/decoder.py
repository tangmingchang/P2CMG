"""Token-only deployment of the NEW decoder; no claim of recovered S0 identity."""
from pathlib import Path

import torch

from model import Config, decode_codes
from model_v2 import EndpointFlow


class NewFDMBDDecoder:
    def __init__(self, checkpoint, codec, device='cuda', allow_untrained_for_diagnostics=False):
        self.path = Path(checkpoint)
        # Only load locally authored/trusted training artifacts.
        artifact = torch.load(self.path, map_location='cpu', weights_only=False)
        step = artifact.get('selected_step', artifact.get('step', 0))
        if step <= 0 and not allow_untrained_for_diagnostics:
            raise ValueError('EnCodec baseline was selected; this artifact is not an improved trained FD-MBD.')
        if artifact.get('parameterization') != 'bounded_polar_endpoint':
            raise ValueError('Checkpoint parameterization is not the polar-endpoint model.')
        self.device = torch.device(device)
        self.model = EndpointFlow(Config(**artifact['architecture'])).to(self.device).eval()
        self.model.load_state_dict(artifact['state'], strict=True)
        self.scale = artifact['scale'].to(self.device).float()
        if self.scale.shape != (1, 4, 1, 1) or not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError('Invalid train-only normalization.')
        if self.model.config.sample_rate != 32000 or getattr(codec, 'sample_rate', None) != 32000:
            raise ValueError('This decoder requires the matching mono MusicGen 32-kHz codec.')
        self.codec = codec.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)
        self.metadata = {key: artifact.get(key) for key in ('seed', 'selected_step', 'step', 'meaning', 'protocol_sha256', 'parameterization')}
        self.metadata['identity'] = 'new_fdmbd_codec_conditioned_polar_endpoint; not historical S0'

    @torch.no_grad()
    def decode(self, codes, length=None, steps=10, seed=0):
        if codes.ndim != 3 or codes.shape[1] != 4 or codes.shape[-1] < 2:
            raise ValueError('Expected [batch,4,frames] RVQ codes with at least two frames.')
        if codes.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('RVQ token ids must have integer dtype.')
        if int(codes.min()) < 0 or int(codes.max()) >= 2048:
            raise ValueError('RVQ token id outside the matching 2048-entry codebook.')
        expected_length = codes.shape[-1] * 640
        length = expected_length if length is None else length
        if not isinstance(length, int) or length < 1024 or length > expected_length:
            raise ValueError('Length must be a valid 32-kHz sample count within the token duration.')
        if not isinstance(steps, int) or not 1 <= steps <= 256:
            raise ValueError('Euler steps must be an integer in [1,256].')
        wave, baseline = decode_codes(self.codec, self.model, codes, self.scale, length, steps, seed)
        if wave.shape != (codes.shape[0], length) or not torch.isfinite(wave).all():
            raise RuntimeError('Invalid decoder output; do not silently normalize or clip.')
        return wave, baseline
