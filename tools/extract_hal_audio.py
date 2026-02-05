#!/usr/bin/env python3
"""
HAL-9000 Audio Sample Extractor

Extracts HAL's dialogue from 2001: A Space Odyssey using subtitles
to identify timestamps, then uses FFmpeg to extract clean audio clips
for ElevenLabs voice cloning.

Usage:
    python extract_hal_audio.py movie.mkv subtitles.srt --output-dir ./hal_samples
"""

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""
    index: int
    start_time: str  # HH:MM:SS,mmm format
    end_time: str
    text: str
    speaker: str | None = None


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp to seconds."""
    # Format: HH:MM:SS,mmm
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts)
    if not match:
        raise ValueError(f"Invalid timestamp: {ts}")

    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(millis) / 1000
    )


def format_timestamp_ffmpeg(seconds: float) -> str:
    """Convert seconds to FFmpeg timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_srt(srt_path: str) -> list[SubtitleEntry]:
    """Parse an SRT subtitle file."""
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # Split into blocks
    blocks = re.split(r"\n\n+", content.strip())
    entries = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        # Parse timestamp line
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1]
        )
        if not time_match:
            continue

        start_time, end_time = time_match.groups()
        text = "\n".join(lines[2:])

        # Try to identify speaker
        speaker = None
        speaker_match = re.match(r"^([A-Z][A-Z\s]+):\s*(.+)$", text, re.DOTALL)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            text = speaker_match.group(2).strip()

        entries.append(SubtitleEntry(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker=speaker,
        ))

    return entries


def find_hal_entries(entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
    """Filter entries to find HAL's dialogue."""
    hal_entries = []

    # Patterns that indicate HAL is speaking
    hal_patterns = [
        r"^HAL",
        r"^H\.?A\.?L\.?",
        r"^HAL[ -]?9000",
    ]

    # Known HAL lines (for subtitles without speaker labels)
    known_hal_lines = [
        "I'm sorry, Dave",
        "I'm afraid I can't do that",
        "This mission is too important",
        "I know I've made some very poor decisions",
        "my mind is going",
        "I can feel it",
        "Good afternoon, gentlemen",
        "I am a HAL 9000 computer",
        "I became operational",
        "completely operational",
        "all my circuits are functioning perfectly",
        "Daisy",
    ]

    for entry in entries:
        is_hal = False

        # Check speaker label
        if entry.speaker:
            for pattern in hal_patterns:
                if re.match(pattern, entry.speaker, re.IGNORECASE):
                    is_hal = True
                    break

        # Check for known HAL lines
        if not is_hal:
            for line in known_hal_lines:
                if line.lower() in entry.text.lower():
                    is_hal = True
                    break

        if is_hal:
            hal_entries.append(entry)

    return hal_entries


def extract_audio_clip(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    output_path: str,
    padding: float = 0.2,
) -> bool:
    """Extract an audio clip using FFmpeg."""
    # Add padding for cleaner cuts
    start_seconds = max(0, start_seconds - padding)
    end_seconds = end_seconds + padding
    duration = end_seconds - start_seconds

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-ss", format_timestamp_ffmpeg(start_seconds),
        "-t", str(duration),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "44100",  # 44.1kHz sample rate
        "-ac", "1",  # Mono
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return False


def merge_adjacent_entries(
    entries: list[SubtitleEntry],
    max_gap: float = 2.0,
) -> list[tuple[float, float, str]]:
    """Merge adjacent subtitle entries into longer clips."""
    if not entries:
        return []

    merged = []
    current_start = parse_timestamp(entries[0].start_time)
    current_end = parse_timestamp(entries[0].end_time)
    current_text = entries[0].text

    for entry in entries[1:]:
        entry_start = parse_timestamp(entry.start_time)
        entry_end = parse_timestamp(entry.end_time)

        # If this entry is close to the previous one, merge them
        if entry_start - current_end <= max_gap:
            current_end = entry_end
            current_text += " " + entry.text
        else:
            # Save current and start new
            merged.append((current_start, current_end, current_text))
            current_start = entry_start
            current_end = entry_end
            current_text = entry.text

    # Don't forget the last one
    merged.append((current_start, current_end, current_text))

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Extract HAL-9000 audio samples from 2001: A Space Odyssey"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("subtitles", help="Path to SRT subtitle file")
    parser.add_argument(
        "--output-dir", "-o",
        default="./hal_samples",
        help="Output directory for audio clips",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum clip duration in seconds",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list HAL's lines, don't extract audio",
    )

    args = parser.parse_args()

    # Parse subtitles
    print(f"Parsing subtitles: {args.subtitles}")
    entries = parse_srt(args.subtitles)
    print(f"Found {len(entries)} subtitle entries")

    # Find HAL's lines
    hal_entries = find_hal_entries(entries)
    print(f"Found {len(hal_entries)} HAL dialogue entries")

    if not hal_entries:
        print("\nNo HAL dialogue found. The subtitle file may not have speaker labels.")
        print("You may need to manually identify HAL's lines or use a different subtitle file.")
        return

    # Merge adjacent entries
    merged = merge_adjacent_entries(hal_entries)
    print(f"Merged into {len(merged)} clips")

    # List HAL's lines
    print("\n" + "=" * 60)
    print("HAL-9000 DIALOGUE")
    print("=" * 60)

    for i, (start, end, text) in enumerate(merged, 1):
        duration = end - start
        print(f"\n[{i}] {format_timestamp_ffmpeg(start)} - {format_timestamp_ffmpeg(end)} ({duration:.1f}s)")
        print(f"    {text[:100]}{'...' if len(text) > 100 else ''}")

    if args.list_only:
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio clips
    print("\n" + "=" * 60)
    print("EXTRACTING AUDIO CLIPS")
    print("=" * 60)

    extracted = 0
    manifest = []

    for i, (start, end, text) in enumerate(merged, 1):
        duration = end - start

        # Skip clips that are too short or too long
        if duration < args.min_duration:
            print(f"[{i}] Skipping (too short: {duration:.1f}s)")
            continue
        if duration > args.max_duration:
            print(f"[{i}] Skipping (too long: {duration:.1f}s)")
            continue

        # Generate output filename
        safe_text = re.sub(r"[^\w\s-]", "", text[:30]).strip().replace(" ", "_")
        output_path = output_dir / f"hal_{i:03d}_{safe_text}.wav"

        print(f"[{i}] Extracting: {output_path.name}")

        if extract_audio_clip(args.video, start, end, str(output_path)):
            extracted += 1
            manifest.append({
                "file": output_path.name,
                "text": text,
                "duration": duration,
            })

    # Write manifest file
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# HAL-9000 Audio Samples\n")
        f.write("# For ElevenLabs Voice Cloning\n\n")
        for item in manifest:
            f.write(f"## {item['file']}\n")
            f.write(f"Duration: {item['duration']:.1f}s\n")
            f.write(f"Text: {item['text']}\n\n")

    print("\n" + "=" * 60)
    print(f"COMPLETE: Extracted {extracted} audio clips to {output_dir}")
    print(f"Manifest saved to {manifest_path}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the extracted clips for quality")
    print("2. Remove any clips with background music or noise")
    print("3. Upload clean clips to ElevenLabs for voice cloning")


if __name__ == "__main__":
    main()
