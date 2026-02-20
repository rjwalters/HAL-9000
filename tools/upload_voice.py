#!/usr/bin/env python3
"""Upload HAL voice samples to ElevenLabs as an Instant Voice Clone.

Sources:
  - hal_samples/trimmed/               denoised dialogue clips from 2001: A Space Odyssey
  - hal_samples/narration_scored/good/  quality-filtered narration clips from the 1960
    NFB documentary "Universe" (produced by tools/score_narration.py)

All sources are concatenated with silence gaps and split into <=10 MB chunks
before uploading.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs


HAL_SAMPLES = Path(__file__).resolve().parent.parent / "hal_samples"
TRIMMED_DIR = HAL_SAMPLES / "trimmed"
NARRATION_DIR = HAL_SAMPLES / "narration_scored" / "good"
SILENCE_GAP_SECS = 2
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB to stay under 11 MB limit
# WAV at 44100 Hz mono 16-bit ≈ 88200 bytes/sec
BYTES_PER_SEC = 88200


def concatenate_clips(wav_files, output_path):
    """Concatenate WAV files with silence gaps into a single file using ffmpeg."""
    inputs = []
    filter_parts = []
    for i, f in enumerate(wav_files):
        inputs.extend(["-i", str(f)])
        filter_parts.append(f"[{i}:a]aresample=44100[s{i}]")
        if i < len(wav_files) - 1:
            filter_parts.append(
                f"anullsrc=r=44100:cl=mono:d={SILENCE_GAP_SECS}[gap{i}]"
            )

    concat_inputs = []
    for i in range(len(wav_files)):
        concat_inputs.append(f"[s{i}]")
        if i < len(wav_files) - 1:
            concat_inputs.append(f"[gap{i}]")

    n_segments = len(wav_files) * 2 - 1
    filter_str = ";".join(filter_parts)
    filter_str += f";{''.join(concat_inputs)}concat=n={n_segments}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-map", "[out]", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
        sys.exit(1)


def build_clip_chunks(wav_files, tmpdir, prefix="clips"):
    """Split clips into chunks that each concatenate to under the size limit."""
    silence_bytes = SILENCE_GAP_SECS * BYTES_PER_SEC

    chunks = []
    current_chunk = []
    current_size = 0

    for f in wav_files:
        file_size = f.stat().st_size
        added_size = file_size + (silence_bytes if current_chunk else 0)
        if current_chunk and current_size + added_size > MAX_FILE_BYTES:
            chunks.append(current_chunk)
            current_chunk = [f]
            current_size = file_size
        else:
            current_chunk.append(f)
            current_size += added_size

    if current_chunk:
        chunks.append(current_chunk)

    output_files = []
    for i, chunk in enumerate(chunks):
        out_path = Path(tmpdir) / f"{prefix}_part{i + 1}.wav"
        concatenate_clips(chunk, out_path)
        output_files.append(out_path)

    return output_files


def main():
    load_dotenv()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not set in .env")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp()
    all_parts = []

    # --- Movie dialogue clips ---
    wav_files = sorted(TRIMMED_DIR.glob("*.wav"))
    if not wav_files:
        print(f"Warning: No .wav files found in {TRIMMED_DIR}")
    else:
        print(f"Found {len(wav_files)} movie dialogue clips")
        clip_parts = build_clip_chunks(wav_files, tmpdir, prefix="movie")
        for p in clip_parts:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")
        all_parts.extend(clip_parts)

    # --- Universe 1960 narration clips ---
    narration_wavs = sorted(NARRATION_DIR.glob("*.wav")) if NARRATION_DIR.exists() else []
    if narration_wavs:
        print(f"\nFound {len(narration_wavs)} narration clips in {NARRATION_DIR.name}/")
        narration_parts = build_clip_chunks(narration_wavs, tmpdir, prefix="narration")
        for p in narration_parts:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")
        all_parts.extend(narration_parts)
    else:
        print(f"\nWarning: No narration clips found in {NARRATION_DIR}")
        print("  Run: python tools/isolate_narration.py")

    if not all_parts:
        print("Error: No audio files to upload")
        sys.exit(1)

    total_mb = sum(p.stat().st_size for p in all_parts) / (1024 * 1024)
    print(f"\nTotal: {len(all_parts)} files, {total_mb:.1f} MB")

    client = ElevenLabs(api_key=api_key)

    print(f"\nUploading to ElevenLabs Instant Voice Clone...")
    voice = client.voices.ivc.create(
        name="HAL 9000",
        description=(
            "HAL 9000 from 2001: A Space Odyssey (Douglas Rain). "
            "Trained on movie dialogue and the 1960 Universe narration."
        ),
        files=[open(p, "rb") for p in all_parts],
    )

    print(f"\nVoice created successfully!")
    print(f"  Voice ID: {voice.voice_id}")
    print(f"\nAdd this to your .env:")
    print(f"  ELEVENLABS_VOICE_ID={voice.voice_id}")

    # Clean up temp files
    for p in all_parts:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
