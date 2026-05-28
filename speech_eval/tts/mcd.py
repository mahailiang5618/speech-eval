"""Mel-Cepstral Distortion (MCD) metric for TTS evaluation.

Inspired by ESPnet's evaluate_mcd.py, provides both a class-based interface
and a simple function interface for computing MCD between generated and
ground truth audio.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import pysptk
except ImportError:
    pysptk = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

from scipy import spatial


def _check_dependencies():
    missing = []
    if librosa is None:
        missing.append("librosa")
    if pysptk is None:
        missing.append("pysptk")
    if sf is None:
        missing.append("soundfile")
    if fastdtw is None:
        missing.append("fastdtw")
    if missing:
        raise RuntimeError(
            f"Missing dependencies for MCD: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    """Get best mcep dimension and alpha for a given sampling rate."""
    params = {
        16000: (23, 0.42),
        22050: (34, 0.45),
        24000: (34, 0.46),
        44100: (39, 0.53),
        48000: (39, 0.55),
    }
    if fs not in params:
        raise ValueError(
            f"No preset mcep params for fs={fs}. "
            f"Supported: {list(params.keys())}. "
            f"Please specify mcep_dim and mcep_alpha manually."
        )
    return params[fs]


def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim: Optional[int] = None,
    mcep_alpha: Optional[float] = None,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.

    Args:
        x: 1D waveform array (int16 or float).
        fs: Sampling rate.
        n_fft: FFT length in points.
        n_shift: Shift length in points.
        mcep_dim: Dimension of mel-cepstrum (auto-set if None).
        mcep_alpha: All-pass filter coefficient (auto-set if None).
        is_padding: Whether to pad the end of signal.

    Returns:
        Mel-cepstrum array with shape (n_frames, mcep_dim + 1).
    """
    _check_dependencies()

    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    n_frame = (len(x) - n_fft) // n_shift + 1

    win = pysptk.sptk.hamming(n_fft)

    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    mcep = [
        pysptk.mcep(
            x[n_shift * i: n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)


def compute_mcd_pair(
    gen_path: Union[str, Path],
    gt_path: Union[str, Path],
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim: Optional[int] = None,
    mcep_alpha: Optional[float] = None,
) -> float:
    """Compute MCD between a single pair of audio files.

    Args:
        gen_path: Path to generated/synthesized audio.
        gt_path: Path to ground truth audio.
        n_fft: FFT length.
        n_shift: Shift length.
        mcep_dim: Mel-cepstrum dimension (auto if None).
        mcep_alpha: All-pass coefficient (auto if None).

    Returns:
        MCD value in dB (lower is better).
    """
    _check_dependencies()

    gen_x, gen_fs = sf.read(str(gen_path), dtype="int16")
    gt_x, gt_fs = sf.read(str(gt_path), dtype="int16")

    fs = gen_fs
    if gen_fs != gt_fs:
        gt_x = librosa.resample(
            gt_x.astype(np.float64), orig_sr=gt_fs, target_sr=gen_fs
        ).astype(np.int16)

    gen_mcep = sptk_extract(
        x=gen_x, fs=fs, n_fft=n_fft, n_shift=n_shift,
        mcep_dim=mcep_dim, mcep_alpha=mcep_alpha,
    )
    gt_mcep = sptk_extract(
        x=gt_x, fs=fs, n_fft=n_fft, n_shift=n_shift,
        mcep_dim=mcep_dim, mcep_alpha=mcep_alpha,
    )

    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_mcep_dtw = gen_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]

    diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return float(mcd)


def find_audio_files(root_dir: str, extensions: List[str] = None) -> List[str]:
    """Recursively find audio files in a directory."""
    if extensions is None:
        extensions = ["*.wav", "*.flac"]
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, ext):
                files.append(os.path.join(root, filename))
    return sorted(files)


class MCD:
    """Compute Mel-Cepstral Distortion for TTS evaluation.

    MCD measures the spectral distance between generated and ground truth
    speech, commonly used to evaluate TTS/voice conversion quality.

    Args:
        n_fft: FFT length in points.
        n_shift: Shift length in points.
        mcep_dim: Mel-cepstrum dimension (auto-determined from sample rate if None).
        mcep_alpha: All-pass filter coefficient (auto-determined if None).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        n_shift: int = 256,
        mcep_dim: Optional[int] = None,
        mcep_alpha: Optional[float] = None,
    ) -> None:
        _check_dependencies()
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha

    def compute(
        self,
        gen_path: Union[str, Path],
        gt_path: Union[str, Path],
    ) -> Dict[str, float]:
        """Compute MCD for a single audio pair.

        Args:
            gen_path: Path to generated audio file.
            gt_path: Path to ground truth audio file.

        Returns:
            Dict with "mcd" value in dB.
        """
        mcd = compute_mcd_pair(
            gen_path, gt_path,
            n_fft=self.n_fft, n_shift=self.n_shift,
            mcep_dim=self.mcep_dim, mcep_alpha=self.mcep_alpha,
        )
        return {"mcd": round(mcd, 4)}

    def compute_batch(
        self,
        gen_paths: List[Union[str, Path]],
        gt_paths: List[Union[str, Path]],
    ) -> Dict[str, object]:
        """Compute MCD for multiple audio pairs.

        Args:
            gen_paths: List of generated audio file paths.
            gt_paths: List of ground truth audio file paths (aligned with gen_paths).

        Returns:
            Dict with "mean_mcd", "std_mcd", and per-file "details".
        """
        assert len(gen_paths) == len(gt_paths), (
            f"Mismatch: {len(gen_paths)} generated vs {len(gt_paths)} ground truth files"
        )

        results = {}
        for gen_path, gt_path in zip(gen_paths, gt_paths):
            basename = Path(gen_path).stem
            mcd = compute_mcd_pair(
                gen_path, gt_path,
                n_fft=self.n_fft, n_shift=self.n_shift,
                mcep_dim=self.mcep_dim, mcep_alpha=self.mcep_alpha,
            )
            results[basename] = round(mcd, 4)

        values = list(results.values())
        return {
            "mean_mcd": round(float(np.mean(values)), 4),
            "std_mcd": round(float(np.std(values)), 4),
            "num_utterances": len(values),
            "details": results,
        }

    def compute_from_dir(
        self,
        gen_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Compute MCD for all matching audio files between two directories.

        Files are matched by basename (filename without extension).

        Args:
            gen_dir: Directory of generated audio files.
            gt_dir: Directory of ground truth audio files.
            output_dir: If provided, write results to this directory.

        Returns:
            Dict with mean/std MCD and per-utterance details.
        """
        gen_files = find_audio_files(str(gen_dir))
        gt_files = find_audio_files(str(gt_dir))

        if not gen_files:
            raise FileNotFoundError(f"No audio files found in {gen_dir}")
        if not gt_files:
            raise FileNotFoundError(f"No audio files found in {gt_dir}")

        gt_map = {Path(f).stem: f for f in gt_files}

        matched_gen = []
        matched_gt = []
        for gen_path in gen_files:
            basename = Path(gen_path).stem
            if basename in gt_map:
                matched_gen.append(gen_path)
                matched_gt.append(gt_map[basename])

        if not matched_gen:
            raise ValueError(
                f"No matching files between gen_dir and gt_dir. "
                f"gen has {len(gen_files)} files, gt has {len(gt_files)} files."
            )

        logging.info(f"Found {len(matched_gen)} matched utterances")

        result = self.compute_batch(matched_gen, matched_gt)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "utt2mcd", "w") as f:
                for utt_id in sorted(result["details"].keys()):
                    f.write(f"{utt_id} {result['details'][utt_id]:.4f}\n")

            with open(output_dir / "mcd_avg_result.txt", "w") as f:
                f.write(f"#utterances: {result['num_utterances']}\n")
                f.write(f"Average: {result['mean_mcd']:.4f} +/- {result['std_mcd']:.4f}\n")

            result["output_dir"] = str(output_dir)

        return result
