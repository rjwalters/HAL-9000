#!/usr/bin/env python3
"""
HAL-9000 Audio Sample Extractor

Two-pass extraction of HAL's dialogue from 2001: A Space Odyssey:
  1. Dialogue database (built by build_dialogue_db.py) identifies HAL lines
     by P(HAL) probability score
  2. MLX Whisper provides word-level timestamps for precise cropping

Usage:
    # From database + pre-extracted center channel WAV:
    python extract_hal_audio.py --db dialogue.db --audio center_channel.wav

    # From database + video (extracts center channel from 5.1):
    python extract_hal_audio.py --db dialogue.db --video movie.mkv

    # Preview what will be extracted:
    python extract_hal_audio.py --db dialogue.db --list-only

    # Adjust probability threshold (default 0.45):
    python extract_hal_audio.py --db dialogue.db --audio wav --threshold 0.3
"""

import argparse
import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class HalLine:
    """A HAL dialogue line from the database."""
    id: int
    start: float
    end: float
    text: str
    hal_probability: float
    speaker: str | None = None


@dataclass
class Window:
    """A rough extraction window around HAL dialogue."""
    start: float
    end: float
    lines: list[HalLine] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def fmt_ffmpeg(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_human(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

def load_hal_lines(db_path: str, min_prob: float = 0.45) -> list[HalLine]:
    """Load HAL dialogue lines from the database above a probability threshold."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, start_seconds, end_seconds, clean_text, "
        "hal_probability, speaker "
        "FROM dialogue WHERE hal_probability >= ? "
        "ORDER BY start_seconds",
        (min_prob,),
    ).fetchall()
    conn.close()

    return [
        HalLine(id=r[0], start=r[1], end=r[2], text=r[3],
                hal_probability=r[4], speaker=r[5])
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def create_windows(
    lines: list[HalLine],
    padding: float = 30.0,
    cluster_gap: float = 10.0,
) -> list[Window]:
    """Create extraction windows around clusters of HAL dialogue."""
    if not lines:
        return []

    clusters: list[list[HalLine]] = []
    current = [lines[0]]

    for line in lines[1:]:
        if line.start - current[-1].end <= cluster_gap:
            current.append(line)
        else:
            clusters.append(current)
            current = [line]
    clusters.append(current)

    windows = []
    for cluster in clusters:
        windows.append(Window(
            start=max(0, cluster[0].start - padding),
            end=cluster[-1].end + padding,
            lines=cluster,
        ))

    # Merge overlapping windows
    merged = [windows[0]]
    for w in windows[1:]:
        if w.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, w.end)
            merged[-1].lines.extend(w.lines)
        else:
            merged.append(w)

    return merged


# ---------------------------------------------------------------------------
# Audio extraction (FFmpeg)
# ---------------------------------------------------------------------------

def _ffmpeg_extract(
    source: str, start: float, duration: float, output: str,
    from_video: bool = False,
) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", fmt_ffmpeg(start),
        "-i", source,
        "-t", str(duration),
    ]
    if from_video:
        cmd += ["-map", "0:a:0", "-af", "pan=mono|c0=FC"]
    cmd += ["-acodec", "pcm_s24le", "-ar", "48000", output]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg error: {e.stderr[:200]}")
        return False


def extract_window(source: str, window: Window, output: str,
                   from_video: bool = False) -> bool:
    return _ffmpeg_extract(
        source, window.start, window.end - window.start, output, from_video
    )


def extract_clip(source: str, start: float, end: float, output: str,
                 padding: float = 0.1, from_video: bool = False) -> bool:
    s = max(0, start - padding)
    return _ffmpeg_extract(source, s, (end + padding) - s, output, from_video)


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_window(
    audio_path: str,
    model: str = "mlx-community/whisper-large-v3-turbo",
) -> list[dict]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        word_timestamps=True,
        language="en",
    )
    return result.get("segments", [])


# ---------------------------------------------------------------------------
# Matching: DB entries <-> Whisper words
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _find_best_span(
    words: list[dict], target_text: str, min_ratio: float = 0.45,
) -> tuple[int, int, float] | None:
    """Find the contiguous word span that best matches target_text.

    Uses a sliding window over word sequences, trying spans of varying
    length centered on the expected word count. Returns (start_idx,
    end_idx, score) or None if no span exceeds min_ratio.
    """
    if not words:
        return None

    target_norm = _normalize(target_text)
    target_words = target_norm.split()
    n_target = len(target_words)
    n_words = len(words)

    best = None
    best_ratio = min_ratio

    # Try spans from 60% to 150% of expected word count
    min_span = max(1, int(n_target * 0.6))
    max_span = min(n_words, int(n_target * 1.5) + 2)

    for span_len in range(min_span, max_span + 1):
        for start_idx in range(n_words - span_len + 1):
            span_text = "".join(
                w["word"] for w in words[start_idx:start_idx + span_len]
            )
            span_norm = _normalize(span_text)
            ratio = SequenceMatcher(None, target_norm, span_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = (start_idx, start_idx + span_len, ratio)

    return best


def _calibrate_offset(
    all_words: list[dict],
    hal_lines: list[HalLine],
    n_anchors: int = 3,
) -> float:
    """Estimate the time offset between DB timestamps and audio.

    Matches a few of the longest HAL lines (best anchor candidates)
    against the full word list, then returns the median offset.
    Returns 0.0 if calibration fails.
    """
    # Pick the longest lines as anchors (more words = more reliable match)
    candidates = sorted(hal_lines, key=lambda l: len(l.text), reverse=True)
    offsets = []

    for line in candidates[:n_anchors * 2]:
        span = _find_best_span(all_words, line.text, min_ratio=0.7)
        if span is None:
            continue
        si, ei, score = span
        whisper_mid = (all_words[si]["start"] + all_words[ei - 1]["end"]) / 2
        db_mid = (line.start + line.end) / 2
        offsets.append(whisper_mid - db_mid)
        if len(offsets) >= n_anchors:
            break

    if not offsets:
        return 0.0
    offsets.sort()
    return offsets[len(offsets) // 2]  # median


def match_hal_segments(
    whisper_segments: list[dict],
    hal_lines: list[HalLine],
    window_start: float,
    tolerance: float = 5.0,
) -> list[dict]:
    """Match whisper word timestamps to HAL database entries.

    Two-step approach to handle DVD SRT / audio time drift:
      1. Calibrate: match a few long anchor lines against the full
         transcript to compute the local time offset
      2. Match: search a narrow time window (offset-corrected ±tolerance)
         for each remaining line using text similarity

    Deduplicates overlapping matches.
    """
    # Flatten all words with absolute timestamps
    all_words = []
    for seg in whisper_segments:
        for word in seg.get("words", []):
            all_words.append({
                "word": word["word"],
                "start": word["start"] + window_start,
                "end": word["end"] + window_start,
            })

    if not all_words:
        return []

    # Step 1: calibrate time offset
    offset = _calibrate_offset(all_words, hal_lines)

    matched = []

    for line in hal_lines:
        # Corrected time window
        adj_start = line.start + offset - tolerance
        adj_end = line.end + offset + tolerance

        candidates = [
            (i, w) for i, w in enumerate(all_words)
            if w["start"] >= adj_start and w["end"] <= adj_end
        ]

        if not candidates:
            # Fall back to wider search
            candidates = [
                (i, w) for i, w in enumerate(all_words)
                if w["start"] >= adj_start - 15 and w["end"] <= adj_end + 15
            ]

        if not candidates:
            continue

        indices = [c[0] for c in candidates]
        words = [c[1] for c in candidates]

        span = _find_best_span(words, line.text, min_ratio=0.55)
        if span is None:
            continue

        si, ei, score = span
        span_words = words[si:ei]
        # Map back to all_words indices for dedup
        global_si = indices[si]
        global_ei = indices[si + (ei - si) - 1] + 1
        whisper_text = "".join(w["word"] for w in span_words).strip()

        matched.append({
            "db_text": line.text,
            "hal_probability": line.hal_probability,
            "whisper_text": whisper_text,
            "start": span_words[0]["start"],
            "end": span_words[-1]["end"],
            "match_score": round(score, 3),
            "word_indices": (global_si, global_ei),
            "words": span_words,
        })

    # Deduplicate: if multiple DB entries matched overlapping word spans,
    # keep the one with the highest match_score * hal_probability
    matched.sort(key=lambda m: m["start"])
    deduped = []
    for m in matched:
        si, ei = m["word_indices"]
        if deduped:
            prev_si, prev_ei = deduped[-1]["word_indices"]
            if si < prev_ei:  # overlapping
                prev_score = (deduped[-1]["match_score"] *
                              deduped[-1]["hal_probability"])
                this_score = m["match_score"] * m["hal_probability"]
                if this_score > prev_score:
                    deduped[-1] = m
                continue
        deduped.append(m)

    for m in deduped:
        del m["word_indices"]

    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract HAL-9000 audio samples from 2001: A Space Odyssey"
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to dialogue.db (built by build_dialogue_db.py)",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", "-a", help="Pre-extracted center channel WAV")
    source.add_argument("--video", "-v", help="Video file (extracts center from 5.1)")

    parser.add_argument("--output-dir", "-o", default="./hal_samples")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.45,
        help="Minimum P(HAL) to include (default: 0.45)",
    )
    parser.add_argument("--window-padding", type=float, default=30.0)
    parser.add_argument(
        "--whisper-model", default="mlx-community/whisper-large-v3-turbo",
    )
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--clip-padding", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--reuse-windows", action="store_true",
        help="Reuse existing window WAVs and transcripts (skip Whisper)",
    )

    args = parser.parse_args()

    if not args.list_only and not args.audio and not args.video:
        parser.error("--audio or --video required (unless --list-only)")

    audio_source = args.audio or args.video
    from_video = args.video is not None

    # === Pass 1: Load from database ===
    print("=" * 60)
    print("PASS 1: DIALOGUE DATABASE")
    print("=" * 60)

    hal_lines = load_hal_lines(args.db, min_prob=args.threshold)
    print(f"Loaded {len(hal_lines)} lines with P(HAL) >= {args.threshold}")

    if not hal_lines:
        print("\nNo HAL lines found above threshold.")
        return

    windows = create_windows(hal_lines, padding=args.window_padding)
    print(f"Created {len(windows)} extraction windows\n")

    total_hal_dur = sum(l.end - l.start for l in hal_lines)
    total_win_dur = sum(w.end - w.start for w in windows)
    print(f"HAL dialogue:  {total_hal_dur:.0f}s across {len(hal_lines)} lines")
    print(f"Window audio:  {total_win_dur:.0f}s to transcribe\n")

    for i, w in enumerate(windows, 1):
        dur = w.end - w.start
        hi = sum(1 for l in w.lines if l.hal_probability >= 0.7)
        med = len(w.lines) - hi
        print(f"  Window {i}: {fmt_human(w.start)} - {fmt_human(w.end)} "
              f"({dur:.0f}s, {len(w.lines)} lines, {hi} high/{med} med)")
        for line in w.lines[:5]:
            prob_bar = "██" if line.hal_probability >= 0.7 else "▓▓"
            print(f"    {prob_bar} P={line.hal_probability:.2f} "
                  f"\"{line.text[:65]}\"")
        if len(w.lines) > 5:
            print(f"    ... and {len(w.lines) - 5} more")

    if args.list_only:
        return

    # === Pass 2: Whisper transcription & fine crop ===
    print("\n" + "=" * 60)
    print("PASS 2: WHISPER TRANSCRIPTION & FINE CROP")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    windows_dir = output_dir / "windows"
    finals_dir = output_dir / "finals"
    windows_dir.mkdir(parents=True, exist_ok=True)
    finals_dir.mkdir(parents=True, exist_ok=True)

    all_clips = []

    for i, window in enumerate(windows, 1):
        dur = window.end - window.start
        print(f"\n--- Window {i}/{len(windows)} "
              f"({fmt_human(window.start)} - {fmt_human(window.end)}, "
              f"{dur:.0f}s, {len(window.lines)} HAL lines) ---")

        window_path = str(windows_dir / f"window_{i:03d}.wav")
        transcript_path = windows_dir / f"window_{i:03d}_transcript.json"

        if args.reuse_windows and transcript_path.exists():
            print("  Reusing saved transcript...")
            with open(transcript_path) as f:
                segments = json.load(f)
            word_count = sum(len(s.get("words", [])) for s in segments)
            print(f"  {len(segments)} segments, {word_count} words")
        else:
            print("  Extracting window audio...")
            if not extract_window(
                audio_source, window, window_path, from_video
            ):
                print("  FAILED - skipping")
                continue

            print("  Running Whisper...")
            segments = transcribe_window(
                window_path, model=args.whisper_model
            )
            word_count = sum(len(s.get("words", [])) for s in segments)
            print(f"  {len(segments)} segments, {word_count} words")

            with open(transcript_path, "w") as f:
                json.dump(segments, f, indent=2)

        # Match
        matched = match_hal_segments(
            segments, window.lines, window.start,
            tolerance=args.tolerance,
        )
        print(f"  Matched {len(matched)}/{len(window.lines)} HAL entries")

        # Extract clips
        for clip in matched:
            clip_dur = clip["end"] - clip["start"]
            if clip_dur < args.min_duration:
                print(f"    SKIP ({clip_dur:.2f}s): \"{clip['whisper_text'][:50]}\"")
                continue

            clip_num = len(all_clips) + 1
            safe = re.sub(r"[^\w\s-]", "", clip["whisper_text"][:30])
            safe = safe.strip().replace(" ", "_")
            clip_path = finals_dir / f"hal_{clip_num:03d}_{safe}.wav"

            prob = clip["hal_probability"]
            score = clip.get("match_score", 0)
            marker = "██" if prob >= 0.7 else "▓▓"
            print(f"    {marker} [{clip_num:03d}] {fmt_human(clip['start'])} - "
                  f"{fmt_human(clip['end'])} ({clip_dur:.1f}s P={prob:.2f} "
                  f"M={score:.2f}) \"{clip['whisper_text'][:50]}\"")

            if extract_clip(
                audio_source, clip["start"], clip["end"],
                str(clip_path), args.clip_padding, from_video,
            ):
                clip["file"] = clip_path.name
                clip["duration"] = clip_dur
                all_clips.append(clip)

    # === Manifest ===
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    manifest = {
        "source": str(audio_source),
        "whisper_model": args.whisper_model,
        "hal_threshold": args.threshold,
        "clips": [
            {
                "file": c["file"],
                "duration": round(c["duration"], 2),
                "start": round(c["start"], 3),
                "end": round(c["end"], 3),
                "hal_probability": c["hal_probability"],
                "match_score": c.get("match_score", 0),
                "db_text": c["db_text"],
                "whisper_text": c["whisper_text"],
            }
            for c in all_clips
        ],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_dur = sum(c["duration"] for c in all_clips)
    high_conf = sum(1 for c in all_clips if c["hal_probability"] >= 0.7)
    print(f"\nExtracted {len(all_clips)} clips ({total_dur:.1f}s total)")
    print(f"  High confidence (P>=0.7): {high_conf}")
    print(f"  Medium confidence:        {len(all_clips) - high_conf}")
    print(f"\nFinal clips:   {finals_dir}/")
    print(f"Window audio:  {windows_dir}/")
    print(f"Manifest:      {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Review clips in {finals_dir}/")
    print(f"  2. Check window transcripts in {windows_dir}/ for missed lines")
    print(f"  3. Remove clips with background music/noise")
    print(f"  4. Upload clean clips to ElevenLabs Voice Lab")


if __name__ == "__main__":
    main()
