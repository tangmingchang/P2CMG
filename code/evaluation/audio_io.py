import os
import subprocess
import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _to_mono_tensor(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure waveform is shape [1, T] by averaging channels if needed."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.dim() == 2:
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)
    # Unexpected shape; flatten last dim and convert to mono
    return waveform.reshape(1, -1)


def _resample_torch(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio  # type: ignore
        return torchaudio.functional.resample(waveform, orig_sr, target_sr)
    except Exception as exc:
        logger.warning(f"torchaudio 重采样失败，将尝试 librosa: {exc}")
        return _resample_librosa(waveform, orig_sr, target_sr)


def _resample_librosa(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
        y = waveform.squeeze(0).cpu().numpy()
        y_rs = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return torch.from_numpy(np.ascontiguousarray(y_rs)).unsqueeze(0)
    except Exception as exc:
        logger.error(f"librosa 重采样失败: {exc}")
        raise


def _load_with_torchaudio(path: str) -> Tuple[torch.Tensor, int]:
    import torchaudio  # type: ignore
    waveform, sample_rate = torchaudio.load(path)
    return waveform, int(sample_rate)


def _load_with_librosa(path: str) -> Tuple[torch.Tensor, int]:
    import numpy as np  # type: ignore
    import librosa  # type: ignore
    # mono=False 保持通道，后续统一转 mono
    y, sr = librosa.load(path, sr=None, mono=False)
    if y is None:
        raise RuntimeError("librosa 返回空音频")
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2 and y.shape[0] < y.shape[1]:
        # librosa 通常返回 [C, T]，保持不变
        pass
    elif y.ndim == 2:
        # 罕见情况下可能是 [T, C]
        y = y.T
    waveform = torch.from_numpy(np.ascontiguousarray(y))
    return waveform, int(sr)


def _load_with_ffmpeg(path: str, target_sr: Optional[int]) -> Tuple[torch.Tensor, int]:
    import numpy as np  # type: ignore
    sr = int(target_sr) if target_sr else 16000
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1",  # 转换为单通道，便于后续处理
        "-ar", str(sr),
        "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as exc:
        raise RuntimeError(f"ffmpeg 解码失败: {exc}; stderr: {exc}")
    audio_bytes = proc.stdout
    if not audio_bytes:
        raise RuntimeError("ffmpeg 未输出音频数据")
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    return waveform, sr


def robust_load_audio(path: str, target_sr: Optional[int] = 16000, ensure_mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio robustly with fallbacks:
    1) torchaudio.load
    2) librosa.load
    3) ffmpeg raw PCM via subprocess

    Returns tensor [1, T] (mono) by default and sample_rate.
    """
    last_error: Optional[Exception] = None

    # 1) torchaudio
    try:
        waveform, sr = _load_with_torchaudio(path)
        if ensure_mono:
            waveform = _to_mono_tensor(waveform)
        if target_sr:
            waveform = _resample_torch(waveform, sr, int(target_sr))
            sr = int(target_sr)
        return waveform, sr
    except Exception as exc:
        last_error = exc
        logger.warning(f"torchaudio 加载失败，将尝试 librosa: {exc}")

    # 2) librosa
    try:
        waveform, sr = _load_with_librosa(path)
        if ensure_mono:
            waveform = _to_mono_tensor(waveform)
        if target_sr:
            waveform = _resample_librosa(waveform, sr, int(target_sr))
            sr = int(target_sr)
        return waveform, sr
    except Exception as exc:
        last_error = exc
        logger.warning(f"librosa 加载失败，将尝试 ffmpeg: {exc}")

    # 3) ffmpeg
    try:
        waveform, sr = _load_with_ffmpeg(path, target_sr)
        if ensure_mono:
            waveform = _to_mono_tensor(waveform)
        return waveform, sr
    except Exception as exc:
        logger.error(f"ffmpeg 加载也失败，放弃: {exc}")
        # 抛出首个错误以便定位原始问题
        raise RuntimeError(f"音频加载失败: {last_error}")


