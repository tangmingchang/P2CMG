"""Transparent CPU estimands. Chroma is never called timbre; absent data isn't zero."""
from __future__ import annotations
import numpy as np
from scipy.signal import stft, find_peaks


def finite_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite feature vectors required')
    return x


def frechet(x, y):
    """Collection Gaussian distance, low-rank exact formulation via nuclear norm.

    tr(sqrt(Cx Cy)) = || X_centered Y_centered.T ||_* / sqrt((n-1)(m-1)).
    No averaging clipwise distances, hidden ridge, complex sqrt or silent NaNs.
    Embedding identity/normalization is a REQUIRED label at the calling boundary.
    """
    x, y = finite_matrix(x), finite_matrix(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError('Feature dimensions differ')
    mx, my = x.mean(0), y.mean(0)
    a = (x - mx) / np.sqrt(len(x) - 1)
    b = (y - my) / np.sqrt(len(y) - 1)
    cross = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    value = float(np.sum((mx - my) ** 2) + np.sum(a * a) + np.sum(b * b) - 2 * cross)
    if value < -1e-7:
        raise ArithmeticError(f'Non-PSD numerical distance {value}')
    return max(0., value)


def paired_bootstrap(records, arm_a, arm_b, metric, repeats=5000, seed=20261018, resample_training_seeds=False):
    # One source's multiple clips/generation seeds are first averaged. Sources,
    # not 10-second chunks, are the sampling units. Keep training-seed pairing.
    by = {}; identities = {}
    for r in records:
        if r['arm'] not in (arm_a, arm_b):
            continue
        value = r.get(metric)
        if value is None:
            raise ValueError('Missing metric: report coverage, do not drop failed pairs')
        if not np.isfinite(value):
            raise ValueError('Nonfinite metric')
        k = (r['source_group_id'], r['training_seed'], r['arm'])
        by.setdefault(k, []).append(value)
        if 'generation_seed' in r:
            identity=(r.get('case_id'),r['generation_seed'],r.get('reference_audio'))
            if identity in identities.setdefault(k,set()):raise ValueError('Duplicated paired output identity')
            identities[k].add(identity)
    groups = sorted({g for g, _, _ in by})
    seeds = sorted({s for _, s, _ in by})
    if len(groups) < 2 or not seeds:
        raise ValueError('At least two paired sources required')
    delta = np.empty((len(groups), len(seeds)))
    for i, g in enumerate(groups):
        for j, s in enumerate(seeds):
            if any((g, s, a) not in by for a in (arm_a, arm_b)):
                raise ValueError('Unbalanced source/seed/arm table')
            if len(by[g, s, arm_a]) != len(by[g, s, arm_b]):
                raise ValueError('Unequal replicate counts')
            if identities and identities.get((g,s,arm_a))!=identities.get((g,s,arm_b)):
                raise ValueError('Generation seeds/cases do not match across arms')
            delta[i, j] = np.mean(by[g, s, arm_a]) - np.mean(by[g, s, arm_b])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        gs = rng.integers(len(groups), size=len(groups))
        ss = rng.integers(len(seeds), size=len(seeds)) if resample_training_seeds else np.arange(len(seeds))
        boot.append(float(delta[np.ix_(gs, ss)].mean()))
    return {'contrast': f'{arm_a} minus {arm_b}', 'metric': metric, 'mean_difference': float(delta.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(), 'source_groups': len(groups), 'training_seeds': seeds,
            'per_training_seed_difference': delta.mean(0).tolist(), 'bootstrap_repeats': repeats,
            'interval_scope': 'two-way source/seed exploratory bootstrap' if resample_training_seeds else 'source bootstrap conditional on fitted training seeds',
            'not_equivalence_test': True}


def frechet_sensitivity(x,y,groups,repeats=20,seed=20261018):
    """Same fixed source subsets for every arm; exploratory sample-size curve.

    Replicate generations within a source are retained together. No selection of
    a favorable subset or replacement of the full collection's primary value.
    """
    x,y=finite_matrix(x),finite_matrix(y);groups=np.asarray(groups)
    if len(x)!=len(y) or len(x)!=len(groups):raise ValueError('Matched collections required')
    unique=sorted(set(groups.tolist()));rng=np.random.default_rng(seed);result=[]
    for n in sorted(set([max(2,len(unique)//2),len(unique)])):
        values=[]
        for _ in range(1 if n==len(unique) else repeats):
            chosen=rng.choice(unique,n,replace=False);keep=np.isin(groups,chosen)
            values.append(frechet(x[keep],y[keep]))
        result.append({'source_groups':n,'generated_count':int(keep.sum()),'repetitions':len(values),
            'mean':float(np.mean(values)),'range':[float(min(values)),float(max(values))],
            'interpretation':'descriptive sample-size sensitivity, not a confidence interval'})
    return result


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid p values')
    order = np.argsort(p); result = np.empty(len(p)); last = 0.
    for rank, i in enumerate(order):
        last = max(last, (len(p) - rank) * p[i]); result[i] = min(1., last)
    return result.tolist()


def permutation_p(group_differences, repeats=20000, seed=20261018):
    x = np.asarray(group_differences, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Finite source-level paired differences required')
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1., 1.], (repeats, len(x))) * x).mean(1)
    return float((1 + (np.abs(null) >= abs(x.mean())).sum()) / (repeats + 1))


def tempo_diagnostic(x, sr):
    """Spectral-flux autocorrelation diagnostic, not validated beat ground truth."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < sr * 5:
        return {'status': 'invalid_or_too_short', 'bpm': None}
    if np.sqrt(np.mean(x * x)) < 1e-5:
        return {'status': 'silent', 'bpm': None}
    hop = max(1, round(sr / 100))
    _, _, z = stft(x, sr, nperseg=4 * hop, noverlap=3 * hop, boundary=None)
    mag = np.log1p(np.abs(z) * 100)
    onset = np.maximum(np.diff(mag, axis=1), 0).sum(0)
    onset -= onset.mean()
    ac = np.correlate(onset, onset, mode='full')[len(onset)-1:]
    # Divide by number of overlapping frames; compare 40--200 BPM.
    ac /= np.maximum(1, np.arange(len(onset), 0, -1))
    rate = sr / hop
    lo, hi = int(np.ceil(60 * rate / 200)), min(len(ac)-1, int(60 * rate / 40))
    peaks, _ = find_peaks(ac[lo:hi+1])
    if not len(peaks) or ac[0] <= 1e-12:
        return {'status': 'no_periodic_peak', 'bpm': None}
    candidates = sorted(((float(ac[i + lo] / ac[0]), i + lo) for i in peaks), reverse=True)
    strength, lag = candidates[0]
    return {'status': 'estimated', 'bpm': float(60 * rate / lag), 'periodicity': strength,
            'alternatives_bpm': [float(60 * rate / i) for _, i in candidates[:3]],
            'method': '100Hz spectral-flux autocorrelation; unvalidated diagnostic, octave ambiguous'}


def tempo_errors(estimated, target):
    if estimated is None:
        return {'strict_relative_error': None, 'octave_tolerant_relative_error': None, 'missing': True}
    if estimated <= 0 or target <= 0:
        raise ValueError('Positive tempo required')
    return {'strict_relative_error': abs(estimated / target - 1),
            'octave_tolerant_relative_error': min(abs(estimated * k / target - 1) for k in (.5, 1, 2)),
            'missing': False}


def pentatonic_note_coverage(midi, confidence, tonic_pc, threshold=.8):
    """Only for externally validated note transcription and independently set tonic.

    Does not infer notes from chroma or select a best-fitting key on test audio.
    Caller must record transcription checkpoint, calibration and coverage.
    """
    if tonic_pc is None:
        return {'status': 'missing_independent_tonic', 'coverage': None}
    if tonic_pc not in range(12):
        raise ValueError('Tonic pitch class must be 0..11')
    midi, confidence = np.asarray(midi), np.asarray(confidence)
    if midi.shape != confidence.shape or not np.isfinite(confidence).all():
        raise ValueError('Pitch/confidence shape or value error')
    keep = np.isfinite(midi) & (confidence >= threshold)
    if not keep.any():
        return {'status': 'no_confident_notes', 'coverage': None, 'retained_notes': 0}
    pcs = (np.rint(midi[keep]).astype(int) - tonic_pc) % 12
    return {'status': 'computed_from_supplied_notes', 'coverage': float(np.isin(pcs, [0, 2, 4, 7, 9]).mean()),
            'retained_notes': int(keep.sum()), 'total_notes': len(midi), 'confidence_threshold': threshold,
            'scale': 'major pentatonic relative to independent tonic; not cultural authenticity'}
