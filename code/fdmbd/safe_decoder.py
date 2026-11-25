"""Explicit clip-free interpolation with full raw-output and fallback reporting.

This is an engineering safety layer, not learned FD-MBD quality evidence.
It does not remove clipping already present in the frozen codec baseline.
"""
from dataclasses import dataclass
import torch
from decoder import NewFDMBDDecoder


@dataclass
class SafeDecodeResult:
    waveform: torch.Tensor
    codec_waveform: torch.Tensor
    raw_refinement: torch.Tensor
    safety: dict
    model_identity: dict


@torch.no_grad()
def safe_mix(raw,codec):
    """Largest feasible fraction on fixed 1,15/16,...,0 grid, per batch item.

    Chosen only from raw/codec waveforms, no target, text or painting input.
    No sample limiter, loudness normalization or hidden baseline substitution.
    Revalidate integer counts on the ACTUAL returned waveform.
    """
    if raw.shape!=codec.shape or raw.ndim!=2 or raw.shape[1]<1:
        raise ValueError('Expected matching nonempty batch-by-sample waveforms')
    if raw.dtype!=torch.float32 or codec.dtype!=torch.float32 or raw.device!=codec.device:
        raise ValueError('Safety layer requires matching float32/device waveforms')
    if not torch.isfinite(raw).all() or not torch.isfinite(codec).all():
        raise ValueError('Nonfinite input cannot be repaired by interpolation')
    batch,samples=raw.shape
    grid=torch.arange(16,-1,-1,device=raw.device,dtype=torch.float32)/16
    candidate=codec[None]+grid[:,None,None]*(raw-codec)[None]
    candidate[0]=raw;candidate[-1]=codec
    codec_count=(codec.abs()>1).sum(1)
    limit=codec_count+int(samples*.001+1e-9)
    counts=(candidate.abs()>1).sum(2)
    peaks=candidate.abs().amax(2)
    feasible=(counts<=limit[None])&(peaks<=2)
    if not feasible.any(0).all():
        raise RuntimeError('No safe raw/codec mixture; refuse output, do not normalize or clip')
    index=feasible.to(torch.int32).argmax(0)
    chosen=candidate[index,torch.arange(batch,device=raw.device)]
    alpha=grid[index]
    count=(chosen.abs()>1).sum(1);peak=chosen.abs().amax(1)
    assert (count<=limit).all() and (peak<=2).all() and torch.isfinite(chosen).all()
    report={'method':'explicit global convex mixture; fixed grid1/16; no pointwise limiter',
        'alpha':alpha.cpu().tolist(),'adjusted':(alpha<1).cpu().tolist(),
        'fallback_to_codec':(alpha==0).cpu().tolist(),
        'raw_overrange_samples':(raw.abs()>1).sum(1).cpu().tolist(),
        'codec_overrange_samples':codec_count.cpu().tolist(),
        'output_overrange_samples':count.cpu().tolist(),'allowed_overrange_samples':limit.cpu().tolist(),
        'output_peak':peak.cpu().tolist(),'samples':samples,
        'raw_passed':feasible[0].cpu().tolist(),
        'not_a_learned_quality_or_historical_reproduction_claim':True}
    return chosen,report


class SafeFDMBDDecoder(NewFDMBDDecoder):
    def __init__(self,checkpoint,codec,device='cuda'):
        super().__init__(checkpoint,codec,device=device)
        artifact=torch.load(self.path,map_location='cpu',weights_only=False)
        if artifact.get('inference_window') is not None:
            raise ValueError('This adopted API uses raw full-context inference, not rejected tiled candidates')

    @torch.no_grad()
    def decode(self,*args,**kwargs):
        raw,codec=super().decode(*args,**kwargs)
        chosen,report=safe_mix(raw.float(),codec.float())
        return SafeDecodeResult(chosen,codec,raw,report,dict(self.metadata))
