#!/usr/bin/env python3
"""Isolate Douglas Rain's voice from the Universe 1960 narration.

4-stage pipeline:
  1. Demucs vocal isolation  — separate vocals from music/SFX/ambient
  2. Silero VAD segmentation — find speech regions in the vocals track
  3. Clip extraction + FFT denoise — extract and clean individual segments
  4. Whisper verification    — transcribe clips, drop non-speech

Usage:
    uv pip install demucs
    python tools/isolate_narration.py

    # Skip demucs on reruns (it takes a few minutes):
    python tools/isolate_narration.py --reuse-vocals
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio

HAL_SAMPLES = Path(__file__).resolve().parent.parent / "hal_samples"
NARRATION_FILE = HAL_SAMPLES / "universe_1960_narration.wav"
DEMUCS_OUTPUT = HAL_SAMPLES / "separated"
CLIPS_DIR = HAL_SAMPLES / "narration_clips"

# VAD parameters
VAD_SAMPLE_RATE = 16000
MERGE_GAP_MS = 300
MIN_SEGMENT_SECS = 0.5

# Clip extraction parameters
CLIP_PAD_SECS = 0.15
CLIP_SAMPLE_RATE = 44100
FFMPEG_DENOISE = "afftdn=nf=-20"

# Whisper verification thresholds
MAX_NO_SPEECH_PROB = 0.7
MIN_WORD_COUNT = 2


# ---------------------------------------------------------------------------
# Stage 1: Demucs vocal isolation
# ---------------------------------------------------------------------------

def run_demucs(input_path: Path, output_dir: Path):
    """Run htdemucs with two-stem vocal separation."""
    print("=" * 60)
    print("STAGE 1: DEMUCS VOCAL ISOLATION")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", str(output_dir),
        str(input_path),
    ]
    print(f"  Running: {' '.join(cmd)}")
    print("  (this takes a few minutes...)\n")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("Demucs failed.")
        sys.exit(1)

    vocals_path = find_vocals(output_dir, input_path.stem)
    print(f"\n  Vocals extracted: {vocals_path}")
    return vocals_path


def find_vocals(output_dir: Path, stem: str) -> Path:
    """Find the vocals.wav in demucs output tree."""
    # demucs outputs to <output_dir>/htdemucs/<stem>/vocals.wav
    vocals = output_dir / "htdemucs" / stem / "vocals.wav"
    if vocals.exists():
        return vocals
    # fallback: search for it
    candidates = list(output_dir.rglob("vocals.wav"))
    if candidates:
        return candidates[0]
    print(f"Error: vocals.wav not found under {output_dir}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stage 2: Silero VAD segmentation
# ---------------------------------------------------------------------------

def run_vad(vocals_path: Path) -> list[tuple[float, float]]:
    """Run Silero VAD on the vocals track and return speech segments."""
    print("\n" + "=" * 60)
    print("STAGE 2: SILERO VAD SEGMENTATION")
    print("=" * 60)

    # Load and resample to 16kHz for Silero
    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != VAD_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, VAD_SAMPLE_RATE)

    # Load Silero VAD
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]

    print(f"  Audio: {waveform.shape[1] / VAD_SAMPLE_RATE:.1f}s at {VAD_SAMPLE_RATE}Hz")

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        waveform.squeeze(0), model, sampling_rate=VAD_SAMPLE_RATE
    )

    # Convert sample indices to seconds
    raw_segments = [
        (ts["start"] / VAD_SAMPLE_RATE, ts["end"] / VAD_SAMPLE_RATE)
        for ts in speech_timestamps
    ]
    print(f"  Raw VAD segments: {len(raw_segments)}")

    # Merge segments closer than MERGE_GAP_MS
    merged = []
    for start, end in raw_segments:
        if merged and (start - merged[-1][1]) < MERGE_GAP_MS / 1000:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    print(f"  After merging (<{MERGE_GAP_MS}ms gap): {len(merged)}")

    # Filter short segments
    filtered = [(s, e) for s, e in merged if (e - s) >= MIN_SEGMENT_SECS]
    print(f"  After filtering (>={MIN_SEGMENT_SECS}s): {len(filtered)}")

    total_speech = sum(e - s for s, e in filtered)
    print(f"  Total speech: {total_speech:.1f}s")

    return filtered


# ---------------------------------------------------------------------------
# Stage 3: Clip extraction + FFT denoise
# ---------------------------------------------------------------------------

def extract_clips(
    vocals_path: Path,
    segments: list[tuple[float, float]],
    output_dir: Path,
) -> list[dict]:
    """Extract each speech segment with padding and FFT denoise."""
    print("\n" + "=" * 60)
    print("STAGE 3: CLIP EXTRACTION + FFT DENOISE")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []

    for i, (start, end) in enumerate(segments, 1):
        padded_start = max(0, start - CLIP_PAD_SECS)
        padded_end = end + CLIP_PAD_SECS
        duration = padded_end - padded_start

        clip_path = output_dir / f"narration_{i:03d}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(padded_start),
            "-i", str(vocals_path),
            "-t", str(duration),
            "-af", FFMPEG_DENOISE,
            "-ar", str(CLIP_SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [{i:3d}] FFMPEG ERROR: {result.stderr[:100]}")
            continue

        size_kb = clip_path.stat().st_size / 1024
        clips.append({
            "file": clip_path.name,
            "start": round(padded_start, 3),
            "end": round(padded_end, 3),
            "duration": round(duration, 2),
            "size_kb": round(size_kb, 1),
        })

        if i % 25 == 0 or i == len(segments):
            print(f"  Extracted {i}/{len(segments)} clips")

    total_dur = sum(c["duration"] for c in clips)
    print(f"\n  {len(clips)} clips extracted ({total_dur:.1f}s total)")
    return clips


# ---------------------------------------------------------------------------
# Stage 4: Whisper verification
# ---------------------------------------------------------------------------

def verify_clips(clips: list[dict], clips_dir: Path, whisper_model: str) -> list[dict]:
    """Run Whisper on each clip, skip non-speech, and add transcripts."""
    print("\n" + "=" * 60)
    print("STAGE 4: WHISPER VERIFICATION")
    print("=" * 60)

    import mlx_whisper

    verified = []
    skipped = 0

    for i, clip in enumerate(clips, 1):
        clip_path = str(clips_dir / clip["file"])

        result = mlx_whisper.transcribe(
            clip_path,
            path_or_hf_repo=whisper_model,
            language="en",
        )

        segments = result.get("segments", [])
        full_text = " ".join(s["text"].strip() for s in segments).strip()
        word_count = len(full_text.split()) if full_text else 0
        avg_no_speech = (
            sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            if segments else 1.0
        )

        if avg_no_speech > MAX_NO_SPEECH_PROB or word_count < MIN_WORD_COUNT:
            skipped += 1
            tag = "SKIP"
            # Remove the clip file
            Path(clip_path).unlink(missing_ok=True)
        else:
            clip["transcript"] = full_text
            clip["word_count"] = word_count
            clip["no_speech_prob"] = round(avg_no_speech, 3)
            verified.append(clip)
            tag = "OK  "

        print(
            f"  [{i:3d}] {tag} nsp={avg_no_speech:.2f} w={word_count:3d} "
            f"{clip['duration']:.1f}s  \"{full_text[:60]}\""
        )

    total_dur = sum(c["duration"] for c in verified)
    print(f"\n  Verified: {len(verified)} clips ({total_dur:.1f}s)")
    print(f"  Skipped:  {skipped} clips")

    return verified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Isolate Douglas Rain's voice from Universe 1960 narration"
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=NARRATION_FILE,
        help=f"Input narration WAV (default: {NARRATION_FILE})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=CLIPS_DIR,
        help=f"Output clips directory (default: {CLIPS_DIR})",
    )
    parser.add_argument(
        "--reuse-vocals", action="store_true",
        help="Skip demucs if vocals.wav already exists",
    )
    parser.add_argument(
        "--whisper-model", default="mlx-community/whisper-large-v3-turbo",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Stage 1: Demucs
    if args.reuse_vocals:
        vocals_path = find_vocals(DEMUCS_OUTPUT, args.input.stem)
        print("=" * 60)
        print("STAGE 1: DEMUCS (reusing existing vocals)")
        print("=" * 60)
        print(f"  {vocals_path}")
    else:
        vocals_path = run_demucs(args.input, DEMUCS_OUTPUT)

    # Stage 2: Silero VAD
    segments = run_vad(vocals_path)

    # Stage 3: Extract + denoise
    clips = extract_clips(vocals_path, segments, args.output)

    # Stage 4: Whisper verification
    verified = verify_clips(clips, args.output, args.whisper_model)

    # Renumber surviving clips sequentially
    for i, clip in enumerate(verified, 1):
        old_path = args.output / clip["file"]
        new_name = f"narration_{i:03d}.wav"
        new_path = args.output / new_name
        if old_path != new_path:
            old_path.rename(new_path)
        clip["file"] = new_name

    # Save manifest
    manifest = {
        "source": str(args.input),
        "vocals": str(vocals_path),
        "whisper_model": args.whisper_model,
        "clips": verified,
    }
    manifest_path = args.output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_dur = sum(c["duration"] for c in verified)
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  {len(verified)} clean narration clips ({total_dur:.1f}s)")
    print(f"  Output:   {args.output}/")
    print(f"  Manifest: {manifest_path}")
    print(f"\nNext: python tools/upload_voice.py")


if __name__ == "__main__":
    main()
