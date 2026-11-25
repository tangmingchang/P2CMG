"""Offline CLI for a trusted local RVQ token tensor; preserves all output identities."""
import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from safe_decoder import SafeFDMBDDecoder


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',required=True,help='Trusted local tensor [batch=1,4,frames]; relative to the working directory')
    parser.add_argument('--output-dir',required=True,help='New directory; existing outputs are never overwritten')
    parser.add_argument('--weights', type=Path, required=True, help='Trusted trained refiner checkpoint')
    parser.add_argument('--codec', type=Path, required=True, help='Trusted local MusicGen 32-kHz EnCodec file or directory')
    parser.add_argument('--noise-seed',type=int,default=42)
    args=parser.parse_args();out=Path(args.output_dir).resolve()
    if out.exists():raise FileExistsError('Choose a new output directory; existing results are protected')
    tokens,weights=Path(args.tokens).resolve(), args.weights.resolve()
    if not tokens.is_file() or not weights.is_file():raise FileNotFoundError('Missing trusted local token/checkpoint')
    codec_path = args.codec.expanduser().resolve()
    if not codec_path.exists():
        raise FileNotFoundError(codec_path)
    out.mkdir(parents=True);start=time.monotonic()
    try:
        codes=torch.load(tokens,map_location='cpu',weights_only=True)
        if not isinstance(codes,torch.Tensor) or codes.ndim!=3 or codes.shape[0]!=1:
            raise ValueError('CLI supports one batch item [1,4,frames]')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        from audiocraft.models.loaders import load_compression_model
        codec = load_compression_model(codec_path, device='cuda')
        decoder = SafeFDMBDDecoder(weights, codec)
        torch.cuda.synchronize();a=time.perf_counter()
        result=decoder.decode(codes,steps=10,seed=args.noise_seed)
        torch.cuda.synchronize();decode_seconds=time.perf_counter()-a
        files={}
        for name,wave in [('audio',result.waveform),('raw',result.raw_refinement),('codec',result.codec_waveform)]:
            p=out/(name+'.wav');sf.write(p,wave[0].cpu().numpy(),32000,subtype='FLOAT')
            files[name]={'path':str(p),'sha256':sha256(p)}
        write_json(out/'receipt.json',{'status':'completed','files':files,'safety':result.safety,
                   'tokens_sha256':sha256(tokens),'weights_sha256':sha256(weights),'model':result.model_identity,
                   'noise_seed':args.noise_seed,'steps':10,'sample_rate':32000,
                   'reference_audio_input':False,'pointwise_clipping_or_loudness_normalization':False,
                   'decode_seconds_with_codec_and_safety':decode_seconds,'runtime_seconds':time.monotonic()-start})
        print('Saved audio.wav, raw.wav, codec.wav and receipt.json to',out,flush=True)
    except BaseException as e:
        write_json(out/'failure.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    if hasattr(signal,'SIGALRM'):
        def timeout(signum,frame):raise TimeoutError('Local decode hard timeout300seconds')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    main()
