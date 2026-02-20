#!/usr/bin/env python3
"""Score narration clips by audio quality and bin into tiers.

Post-processing step after isolate_narration.py. Compares energy in the
Demucs vocals vs no_vocals tracks to measure background contamination,
then adds SNR, spectral flatness, and clipping metrics.

Usage:
    python tools/score_narration.py
    python tools/score_narration.py --good-threshold 70 --reject-threshold 30
"""

import argparse
import json
import math
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio

HAL_SAMPLES = Path(__file__).resolve().parent.parent / "hal_samples"
CLIPS_DIR = HAL_SAMPLES / "narration_clips"
DEMUCS_OUTPUT = HAL_SAMPLES / "separated"
SCORED_DIR = HAL_SAMPLES / "narration_scored"

# Scoring weights
WEIGHT_VAR = 0.50
WEIGHT_SNR = 0.25
WEIGHT_SF = 0.15
WEIGHT_CLIP = 0.10

# Default tier thresholds
DEFAULT_GOOD = 65
DEFAULT_REJECT = 35

# VAD parameters
VAD_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as float32 mono numpy array. Returns (samples, sr)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def find_track(output_dir: Path, stem: str, track: str) -> Path:
    """Find a demucs output track (vocals.wav or no_vocals.wav)."""
    path = output_dir / "htdemucs" / stem / track
    if path.exists():
        return path
    candidates = list(output_dir.rglob(track))
    if candidates:
        return candidates[0]
    print(f"Error: {track} not found under {output_dir}")
    sys.exit(1)


def rms(signal: np.ndarray) -> float:
    """Root mean square of a signal."""
    return float(np.sqrt(np.mean(signal ** 2))) if len(signal) > 0 else 0.0


def sigmoid_norm(value: float, center: float, scale: float) -> float:
    """Map a value to 0-1 via sigmoid, centered at `center`."""
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def compute_spectral_flatness(signal: np.ndarray, sr: int) -> float:
    """Spectral flatness (Wiener entropy) of a signal. 0=tonal, 1=white noise."""
    if len(signal) < 256:
        return 0.0
    # Use short windows and average
    win_size = 1024
    hop = 512
    flatness_values = []
    for start in range(0, len(signal) - win_size, hop):
        frame = signal[start : start + win_size]
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = spectrum[1:]  # drop DC
        spectrum = np.maximum(spectrum, 1e-10)
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        if arith_mean > 0:
            flatness_values.append(geo_mean / arith_mean)
    return float(np.mean(flatness_values)) if flatness_values else 0.0


# ---------------------------------------------------------------------------
# Stage 1: Load & Validate
# ---------------------------------------------------------------------------

def load_inputs(clips_dir: Path, demucs_dir: Path):
    """Load manifest, vocals, and no_vocals tracks."""
    print("=" * 60)
    print("STAGE 1: LOAD & VALIDATE")
    print("=" * 60)

    manifest_path = clips_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  Error: manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    clips = manifest["clips"]
    print(f"  Manifest: {len(clips)} clips")

    # Determine demucs stem name from the manifest's source path
    source_stem = Path(manifest.get("source", "")).stem
    if not source_stem:
        source_stem = "universe_1960_narration"

    vocals_path = find_track(demucs_dir, source_stem, "vocals.wav")
    no_vocals_path = find_track(demucs_dir, source_stem, "no_vocals.wav")

    print(f"  Vocals:    {vocals_path}")
    print(f"  No-vocals: {no_vocals_path}")

    # Load full tracks (stdlib wave — avoids torchcodec dependency)
    vocals_np, vocals_sr = load_wav(vocals_path)
    no_vocals_np, nv_sr = load_wav(no_vocals_path)

    print(f"  Vocals:    {len(vocals_np) / vocals_sr:.1f}s at {vocals_sr}Hz")
    print(f"  No-vocals: {len(no_vocals_np) / nv_sr:.1f}s at {nv_sr}Hz")

    if vocals_sr != nv_sr:
        print("  Warning: sample rates differ, resampling no_vocals to match")
        no_vocals_t = torchaudio.functional.resample(
            torch.tensor(no_vocals_np).unsqueeze(0), nv_sr, vocals_sr
        )
        no_vocals_np = no_vocals_t.squeeze(0).numpy()
        nv_sr = vocals_sr

    # Load Silero VAD
    print("  Loading Silero VAD...")
    vad_model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_ts = utils[0]
    print("  Ready.\n")

    return manifest, clips, vocals_np, no_vocals_np, vocals_sr, vad_model, get_speech_ts


# ---------------------------------------------------------------------------
# Stage 2: Score Each Clip
# ---------------------------------------------------------------------------

def score_clip(
    clip: dict,
    clips_dir: Path,
    vocals_np: np.ndarray,
    no_vocals_np: np.ndarray,
    sr: int,
    vad_model,
    get_speech_ts,
) -> dict:
    """Compute 4 quality metrics for a single clip."""
    start_sample = int(clip["start"] * sr)
    end_sample = int(clip["end"] * sr)

    # Slice the demucs tracks for this clip's time range
    voc_slice = vocals_np[start_sample:end_sample]
    nov_slice = no_vocals_np[start_sample:end_sample]

    # --- Metric 1: Vocals-to-Accompaniment Ratio (VAR) ---
    voc_rms = rms(voc_slice)
    nov_rms = rms(nov_slice)
    if nov_rms > 0 and voc_rms > 0:
        var_db = 20 * math.log10(voc_rms / nov_rms)
    elif voc_rms > 0:
        var_db = 60.0  # effectively no accompaniment
    else:
        var_db = 0.0
    var_db = max(min(var_db, 60.0), -20.0)  # clamp

    # --- Load the processed clip for remaining metrics ---
    clip_path = clips_dir / clip["file"]
    clip_np, clip_sr = load_wav(clip_path)

    # --- Metric 4: Clipping fraction (compute before resampling) ---
    peak = float(np.max(np.abs(clip_np)))
    clip_fraction = float(np.mean(np.abs(clip_np) > 0.99)) if peak > 0 else 0.0

    # --- Run VAD on clip to find speech vs silence frames ---
    clip_tensor = torch.tensor(clip_np).unsqueeze(0)
    if clip_sr != VAD_SAMPLE_RATE:
        clip_16k = torchaudio.functional.resample(clip_tensor, clip_sr, VAD_SAMPLE_RATE)
    else:
        clip_16k = clip_tensor
    clip_16k_np = clip_16k.squeeze(0).numpy()

    vad_model.reset_states()
    speech_ts = get_speech_ts(
        clip_16k.squeeze(0), vad_model, sampling_rate=VAD_SAMPLE_RATE
    )

    has_silence = False
    snr_db = 0.0
    spectral_flat = 0.0

    if speech_ts:
        # Build speech and silence masks at 16kHz
        speech_mask = np.zeros(len(clip_16k_np), dtype=bool)
        for ts in speech_ts:
            speech_mask[ts["start"]:ts["end"]] = True

        speech_samples = clip_16k_np[speech_mask]
        silence_samples = clip_16k_np[~speech_mask]

        if len(silence_samples) > 160:  # at least 10ms of silence at 16kHz
            has_silence = True

            # --- Metric 2: Speech-to-Silence SNR ---
            speech_rms = rms(speech_samples)
            silence_rms = rms(silence_samples)
            if silence_rms > 0 and speech_rms > 0:
                snr_db = 20 * math.log10(speech_rms / silence_rms)
            elif speech_rms > 0:
                snr_db = 60.0
            snr_db = max(min(snr_db, 60.0), 0.0)

            # --- Metric 3: Spectral flatness of non-speech frames ---
            spectral_flat = compute_spectral_flatness(silence_samples, VAD_SAMPLE_RATE)

    # --- Composite score ---
    var_score = sigmoid_norm(var_db, 10.0, 5.0)      # center=10dB, scale=5
    clip_score = sigmoid_norm(-clip_fraction * 100, 0, 2.0)  # penalize clipping

    if has_silence:
        snr_score = sigmoid_norm(snr_db, 15.0, 5.0)
        sf_score = sigmoid_norm(-spectral_flat * 10, -2.0, 1.0)
        composite = (
            WEIGHT_VAR * var_score
            + WEIGHT_SNR * snr_score
            + WEIGHT_SF * sf_score
            + WEIGHT_CLIP * clip_score
        ) * 100
    else:
        # No silence frames — redistribute SNR + SF weight to VAR + clip
        w_var = WEIGHT_VAR + WEIGHT_SNR + WEIGHT_SF * 0.5
        w_clip = WEIGHT_CLIP + WEIGHT_SF * 0.5
        composite = (w_var * var_score + w_clip * clip_score) * 100

    composite = max(0.0, min(100.0, composite))

    return {
        "var_db": round(var_db, 1),
        "snr_db": round(snr_db, 1),
        "spectral_flatness": round(spectral_flat, 2),
        "clip_fraction": round(clip_fraction, 3),
        "composite": round(composite, 1),
    }


def score_all_clips(
    clips: list[dict],
    clips_dir: Path,
    vocals_np: np.ndarray,
    no_vocals_np: np.ndarray,
    sr: int,
    vad_model,
    get_speech_ts,
) -> list[dict]:
    """Score every clip and print per-clip results."""
    print("=" * 60)
    print("STAGE 2: SCORE EACH CLIP")
    print("=" * 60)

    for i, clip in enumerate(clips, 1):
        quality = score_clip(
            clip, clips_dir, vocals_np, no_vocals_np, sr, vad_model, get_speech_ts
        )
        clip["quality"] = quality

        transcript = clip.get("transcript", "")[:50]
        q = quality
        print(
            f"  [{i:3d}] VAR={q['var_db']:5.1f} SNR={q['snr_db']:5.1f} "
            f"SF={q['spectral_flatness']:.2f} CL={q['clip_fraction']:.3f} "
            f"score={q['composite']:5.1f}  \"{transcript}\""
        )

        if i % 25 == 0:
            print(f"  --- {i}/{len(clips)} clips scored ---")

    print(f"\n  All {len(clips)} clips scored.")
    return clips


# ---------------------------------------------------------------------------
# Stage 3: Classify into Tiers
# ---------------------------------------------------------------------------

def classify_tiers(clips: list[dict], good_thresh: int, reject_thresh: int):
    """Assign tiers and print distribution histogram."""
    print("\n" + "=" * 60)
    print("STAGE 3: CLASSIFY INTO TIERS")
    print("=" * 60)

    for clip in clips:
        score = clip["quality"]["composite"]
        if score >= good_thresh:
            clip["tier"] = "good"
        elif score >= reject_thresh:
            clip["tier"] = "marginal"
        else:
            clip["tier"] = "reject"

    # Histogram
    bins = [0] * 10  # 0-9, 10-19, ..., 90-100
    for clip in clips:
        idx = min(int(clip["quality"]["composite"] // 10), 9)
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    print("\n  Score distribution:")
    for i in range(9, -1, -1):
        lo = i * 10
        hi = lo + 9 if i < 9 else 100
        bar = "#" * int(bins[i] / max_count * 40) if max_count > 0 else ""
        label = ""
        if lo >= good_thresh:
            label = " GOOD"
        elif lo >= reject_thresh:
            label = " MARGINAL"
        else:
            label = " REJECT"
        print(f"  {lo:3d}-{hi:3d} | {bar:<40s} {bins[i]:3d}{label}")

    # Tier summary
    tier_counts = {"good": 0, "marginal": 0, "reject": 0}
    tier_durations = {"good": 0.0, "marginal": 0.0, "reject": 0.0}
    for clip in clips:
        t = clip["tier"]
        tier_counts[t] += 1
        tier_durations[t] += clip["duration"]

    print(f"\n  Tier summary:")
    for tier in ("good", "marginal", "reject"):
        tag = tier.upper()
        n = tier_counts[tier]
        dur = tier_durations[tier]
        print(f"    {tag:<10s} {n:3d} clips  ({dur:.1f}s)")

    total = sum(tier_counts.values())
    total_dur = sum(tier_durations.values())
    print(f"    {'TOTAL':<10s} {total:3d} clips  ({total_dur:.1f}s)")


# ---------------------------------------------------------------------------
# Stage 4: Write Output
# ---------------------------------------------------------------------------

def write_output(clips: list[dict], clips_dir: Path, output_dir: Path, manifest: dict):
    """Copy clips into tier subdirectories and write scored manifest."""
    print("\n" + "=" * 60)
    print("STAGE 4: WRITE OUTPUT")
    print("=" * 60)

    # Create tier directories
    for tier in ("good", "marginal", "reject"):
        (output_dir / tier).mkdir(parents=True, exist_ok=True)

    # Copy clips into tier dirs
    for clip in clips:
        src = clips_dir / clip["file"]
        dst = output_dir / clip["tier"] / clip["file"]
        shutil.copy2(src, dst)

    # Build summary
    tier_counts = {"good": 0, "marginal": 0, "reject": 0}
    tier_durations = {"good": 0.0, "marginal": 0.0, "reject": 0.0}
    for clip in clips:
        t = clip["tier"]
        tier_counts[t] += 1
        tier_durations[t] += clip["duration"]

    summary = {}
    for tier in ("good", "marginal", "reject"):
        summary[tier] = {
            "count": tier_counts[tier],
            "duration": round(tier_durations[tier], 1),
        }
    summary["total"] = {
        "count": sum(tier_counts.values()),
        "duration": round(sum(tier_durations.values()), 1),
    }

    # Write manifest
    scored_manifest = {
        "source": manifest.get("source", ""),
        "vocals": manifest.get("vocals", ""),
        "whisper_model": manifest.get("whisper_model", ""),
        "summary": summary,
        "clips": clips,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(scored_manifest, f, indent=2)

    for tier in ("good", "marginal", "reject"):
        n = tier_counts[tier]
        print(f"  {tier}/: {n} clips copied")
    print(f"  Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score narration clips by audio quality and bin into tiers"
    )
    parser.add_argument(
        "--clips-dir", type=Path, default=CLIPS_DIR,
        help=f"Input clips directory (default: {CLIPS_DIR})",
    )
    parser.add_argument(
        "--demucs-dir", type=Path, default=DEMUCS_OUTPUT,
        help=f"Demucs separated output directory (default: {DEMUCS_OUTPUT})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=SCORED_DIR,
        help=f"Output scored directory (default: {SCORED_DIR})",
    )
    parser.add_argument(
        "--good-threshold", type=int, default=DEFAULT_GOOD,
        help=f"Minimum composite score for 'good' tier (default: {DEFAULT_GOOD})",
    )
    parser.add_argument(
        "--reject-threshold", type=int, default=DEFAULT_REJECT,
        help=f"Maximum composite score for 'reject' tier (default: {DEFAULT_REJECT})",
    )
    args = parser.parse_args()

    # Stage 1: Load
    manifest, clips, vocals_np, no_vocals_np, sr, vad_model, get_speech_ts = (
        load_inputs(args.clips_dir, args.demucs_dir)
    )

    # Stage 2: Score
    clips = score_all_clips(
        clips, args.clips_dir, vocals_np, no_vocals_np, sr, vad_model, get_speech_ts
    )

    # Stage 3: Classify
    classify_tiers(clips, args.good_threshold, args.reject_threshold)

    # Stage 4: Write
    write_output(clips, args.clips_dir, args.output, manifest)

    # Done
    total_good = sum(1 for c in clips if c["tier"] == "good")
    total_dur = sum(c["duration"] for c in clips if c["tier"] == "good")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  {total_good} good clips ({total_dur:.1f}s) ready for training")
    print(f"  Output: {args.output}/")
    print(f"\nNext: listen to marginal/ clips and promote or reject by ear")


if __name__ == "__main__":
    main()
