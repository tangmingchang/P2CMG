"""New training launcher: validates three independent splits before importing GPU code."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def portable_path(value):
    # Existing Windows manifests can be reused from WSL without editing originals.
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', str(value))
    if os.name == 'posix' and match:
        return Path('/mnt')/match.group(1).lower()/match.group(2).replace('\\','/')
    return Path(value)


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty manifest: '+str(path))
    for row in rows:
        for key in ('path', 'image_path'):
            row[key] = str(portable_path(row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if not row.get('source_group_id'):
            raise ValueError('Source group IDs required')
    return rows


def main():
    p = argparse.ArgumentParser()
    for name in ('train', 'valid', 'test'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--updates-per-epoch', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--segment-seconds', type=float, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--conditioner', choices=['imagenclip2music', 'text2music'], default='imagenclip2music')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if min(a.epochs, a.updates_per_epoch, a.batch_size, a.segment_seconds) <= 0:
        p.error('Training budgets must be positive')
    manifests = {name: getattr(a, name).resolve() for name in ('train', 'valid', 'test')}
    rows = {name: read_manifest(path) for name, path in manifests.items()}
    for name, rs in rows.items():
        if any(r.get('split') != name for r in rs):
            raise ValueError('Manifest split label mismatch: '+name)
    for first, second in (('train','valid'), ('train','test'), ('valid','test')):
        for key in ('source_group_id', 'image_path', 'path'):
            if {r[key] for r in rows[first]} & {r[key] for r in rows[second]}:
                raise ValueError(f'Split overlap: {first}/{second}/{key}')
    package = Path(__file__).resolve().parent/'audiocraft'
    config = package/'config/conditioner'/f'{a.conditioner}.yaml'
    if not config.is_file():
        raise FileNotFoundError('Required conditioner missing; fallback disabled: '+str(config))
    run_dir = a.run_dir.resolve()
    # Hydra quoted strings preserve spaces and Windows drive letters.
    q = lambda path: json.dumps(path.as_posix())
    overrides = ['solver=musicgen/musicgen_base_32khz', 'conditioner='+a.conditioner,
        'device='+a.device, 'seed='+str(a.seed), 'mp_start_method=spawn',
        'datasource.max_sample_rate=32000', 'datasource.max_channels=1',
        'datasource.train='+q(manifests['train']), 'datasource.valid='+q(manifests['valid']),
        'datasource.evaluate='+q(manifests['test']), 'datasource.generate='+q(manifests['valid']),
        'dataset.batch_size='+str(a.batch_size), 'dataset.segment_duration='+str(a.segment_seconds),
        'dataset.num_workers=0', 'optim.epochs='+str(a.epochs),
        'optim.updates_per_epoch='+str(a.updates_per_epoch),
        'optim.optimizer=adamw', 'optim.lr=0.0001', 'optim.ema.device=cpu',
        'schedule.cosine.warmup=0', 'dora.dir='+q(run_dir), 'dora.git_save=false',
        'evaluate.every=0', 'generate.every=0', 'logging.log_tensorboard=false']
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = run_dir/'manifests'
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized = {}
    for name, rs in rows.items():
        normalized[name] = normalized_dir/(name+'.jsonl')
        normalized[name].write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rs),encoding='utf-8')
    # AudioCraft reads these normalized manifests, not the unconverted input paths.
    overrides = [s for s in overrides if not s.startswith(('datasource.train=', 'datasource.valid=',
                                                           'datasource.evaluate=', 'datasource.generate='))]
    overrides += ['datasource.train='+q(normalized['train']), 'datasource.valid='+q(normalized['valid']),
                  'datasource.evaluate='+q(normalized['test']), 'datasource.generate='+q(normalized['valid'])]
    record = {'status': 'prepared_not_trained', 'overrides': overrides,
        'manifest_sha256': {n: hashlib.sha256(v.read_bytes()).hexdigest() for n,v in manifests.items()},
        'normalized_manifest_sha256': {n: hashlib.sha256(v.read_bytes()).hexdigest() for n,v in normalized.items()},
        'row_counts': {n:len(rs) for n,rs in rows.items()},
        'note': 'New training from random LM initialization; not recovery of original best run. Explicit AdamW budget is a new choice. Test generation/evaluation disabled during model selection.'}
    (run_dir/'launch_plan.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if a.prepare_only:
        print(json.dumps({'status': record['status'], 'rows': record['row_counts']}))
        return
    os.environ['AUDIOCRAFT_CLUSTER'] = 'default'
    os.environ['AUDIOCRAFT_DORA_DIR'] = str(run_dir)
    os.environ['AUDIOCRAFT_REFERENCE_DIR'] = str(run_dir/'reference')
    sys.path.insert(0, str(package))
    from audiocraft import train
    sys.argv = [sys.argv[0]]+overrides
    train.main()


if __name__ == '__main__':
    main()
