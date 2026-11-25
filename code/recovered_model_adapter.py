"""Actual archived LM adapter; requires a trained Dora experiment in the repaired runtime.

Configure P2CMG_DORA_DIR, P2CMG_SIG, P2CMG_CHECKPOINT_FILE; decoder defaults to
encodec. P2CMG_DECODER=mbd uses public MBD, not custom FDG. No VLM regeneration.
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

_solver = None
_identity = None


def generate(request):
    global _solver, _identity
    checkpoint = Path(os.environ['P2CMG_CHECKPOINT_FILE']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    hasher = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest.lower() != request['checkpoint_sha256'].lower():
        raise ValueError('Checkpoint hash does not match request')
    dora = Path(os.environ['P2CMG_DORA_DIR']).resolve()
    sig = os.environ['P2CMG_SIG']
    expected = (dora/'xps'/sig).resolve()
    if not checkpoint.is_relative_to(expected):
        raise ValueError('Checkpoint must belong to the requested Dora experiment directory')
    os.environ['AUDIOCRAFT_CLUSTER'] = 'default'
    os.environ['AUDIOCRAFT_DORA_DIR'] = str(dora)
    os.environ['AUDIOCRAFT_REFERENCE_DIR'] = str(dora/'reference')
    sys.path.insert(0, str(Path(__file__).resolve().parent/'audiocraft'))
    import numpy as np
    import torch
    from audiocraft.train import get_solver_from_sig
    from audiocraft.modules.conditioners import ConditioningAttributes
    from audiocraft.data.audio import audio_write
    from audiocraft.models import MultiBandDiffusion, DiffusionCompressionWrapper
    identity = (str(dora), sig, digest)
    if _solver is None:
        _solver = get_solver_from_sig(sig, load_best=True, disable_fsdp=True,
                                    ignore_state_keys=['optimizer'])
        _identity = identity
    if _identity != identity:
        raise ValueError('Checkpoint switch requires a fresh adapter process')
    solver = _solver
    if solver.checkpoint_path(use_fsdp=False).resolve() != checkpoint:
        raise ValueError('Hashed file is not the checkpoint path used by this solver')
    solver.model.eval()
    if int(solver.cfg.sample_rate) != request['sample_rate_hz']:
        raise ValueError('Requested sample rate differs from model')
    if 'image_clip' not in solver.model.condition_provider.conditioners:
        raise ValueError('Checkpoint has no image conditioner; image controls would be invalid')
    decoder = os.environ.get('P2CMG_DECODER', 'encodec')
    n = int(request['decoding_steps'])
    if decoder == 'mbd':
        if not 1 <= n <= 999:
            raise ValueError('MBD NFE must be in 1..999')
        # N network evaluations: sampler consumes all except final schedule state.
        schedule = [round(999*(n-i)/n) for i in range(n+1)]
        if len(set(schedule)) != n+1:
            raise ValueError('Nonunique diffusion states')
        if not isinstance(solver.compression_model, DiffusionCompressionWrapper):
            diffusion = MultiBandDiffusion.get_mbd_musicgen(device=next(solver.model.parameters()).device)
            solver.compression_model = DiffusionCompressionWrapper(solver.compression_model, diffusion, step_list=schedule)
        else:
            solver.compression_model.set_step_list(schedule)
    elif decoder != 'encodec':
        raise ValueError('Supported recovered decoders: encodec, mbd. Custom FDG absent.')
    seed = int(request['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    description = request['schema_condition']
    if not isinstance(description, str):
        raise TypeError('Recovered model consumes a paragraph; do not invent fieldwise serialization')
    cond = ConditioningAttributes(text={'description': description}, image={'image_clip': request['image_condition']})
    output = Path(request['output_audio'])
    if output.exists():
        raise FileExistsError(output)
    with torch.no_grad():
        length = round(request['duration_seconds']*solver.compression_model.frame_rate)
        tokens = solver.model.generate(prompt=None, conditions=[cond], max_gen_len=length,
                                       **solver.generation_params)
        audio = solver.compression_model.decode(tokens, None)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_write(output, audio.cpu(), solver.cfg.sample_rate, format='wav', add_suffix=False)
    output.with_suffix('.provenance.json').write_text(json.dumps({
        'checkpoint_sha256': digest, 'sig': sig, 'decoder': decoder,
        'requested_nfe_per_band': n if decoder == 'mbd' else None,
        'image_condition': request['image_condition'], 'description': description, 'seed': seed,
        'duration_seconds': request['duration_seconds'], 'sample_rate_hz': request['sample_rate_hz']},
        ensure_ascii=False, indent=2), encoding='utf-8')
