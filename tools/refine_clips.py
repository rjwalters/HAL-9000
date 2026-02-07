#!/usr/bin/env python3
"""
Refine HAL clips with Whisper-in-the-loop verification.

For each clip:
  1. Whisper the wide clip (from finals/) to find HAL's words
  2. Re-crop tightly from the source audio
  3. Whisper the cropped clip to verify all words survived
  4. If truncated, extend and retry until Whisper confirms the full text

Usage:
    python refine_clips.py --manifest manifest.json --audio center_channel.wav
"""

import argparse
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _find_best_span(words, target_text, min_ratio=0.50):
    if not words:
        return None

    target_norm = _normalize(target_text)
    target_words = target_norm.split()
    n_target = len(target_words)
    n_words = len(words)

    best = None
    best_ratio = min_ratio

    min_span = max(1, int(n_target * 0.6))
    max_span = min(n_words, int(n_target * 1.5) + 2)

    for span_len in range(min_span, max_span + 1):
        for si in range(n_words - span_len + 1):
            span_text = "".join(w["word"] for w in words[si:si + span_len])
            span_norm = _normalize(span_text)
            ratio = SequenceMatcher(None, target_norm, span_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = (si, si + span_len, ratio)

    return best


def fmt_ffmpeg(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_wav(source: str, start: float, duration: float, output: str):
    cmd = [
        "ffmpeg", "-y",
        "-ss", fmt_ffmpeg(start),
        "-i", source,
        "-t", str(duration),
        "-acodec", "pcm_s24le", "-ar", "48000",
        output,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def whisper_words(path: str, model: str) -> list[dict]:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        path, path_or_hf_repo=model,
        word_timestamps=True, language="en",
    )
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(w)
    return words


def verify_text(words: list[dict], target: str, min_ratio: float = 0.70) -> float:
    """Check how well the whispered clip matches the target text."""
    if not words:
        return 0.0
    full_text = _normalize("".join(w["word"] for w in words))
    target_norm = _normalize(target)
    return SequenceMatcher(None, target_norm, full_text).ratio()


def main():
    parser = argparse.ArgumentParser(
        description="Refine HAL clips with Whisper verification loop"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio", required=True,
                        help="Source audio (center channel WAV)")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--initial-padding", type=float, default=0.3,
                        help="Initial padding around crop (default: 0.3s)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--whisper-model",
                        default="mlx-community/whisper-large-v3-turbo")
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--clip-padding", type=float, default=2.0,
                        help="Padding used in extract_hal_audio.py (default: 2.0)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    manifest_dir = Path(args.manifest).parent
    finals_dir = manifest_dir / "finals"
    output_dir = (Path(args.output_dir) if args.output_dir
                  else manifest_dir / "refined")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so --help works without mlx_whisper
    import mlx_whisper  # noqa: F401

    clips = manifest["clips"]
    refined = []
    stats = {"skipped": 0, "unchanged": 0, "ok": 0, "extended": 0}

    for i, clip in enumerate(clips, 1):
        if clip["hal_probability"] < args.min_prob:
            stats["skipped"] += 1
            continue

        clip_path = finals_dir / clip["file"]
        if not clip_path.exists():
            print(f"  [{i}] MISSING: {clip['file']}")
            continue

        # Step 1: Whisper the wide clip to find HAL's words
        words = whisper_words(str(clip_path), args.whisper_model)
        if not words:
            print(f"  [{i}] NO WORDS: {clip['file']}")
            continue

        span = _find_best_span(words, clip["db_text"])
        if span is None:
            print(f"  [{i}] NO MATCH (keeping original): "
                  f"\"{clip['db_text'][:50]}\"")
            stats["unchanged"] += 1
            subprocess.run(["cp", str(clip_path),
                           str(output_dir / clip["file"])], check=True)
            refined.append(clip)
            continue

        si, ei, score = span
        span_words = words[si:ei]
        whisper_text = "".join(w["word"] for w in span_words).strip()
        w_start = span_words[0]["start"]
        w_end = span_words[-1]["end"]

        # Absolute time in source audio
        file_start = clip["start"] - args.clip_padding
        abs_word_start = file_start + w_start
        abs_word_end = file_start + w_end

        # Step 2-4: Extract, verify, extend loop
        out_path = output_dir / clip["file"]
        pad = args.initial_padding
        attempt = 0
        final_text = ""
        verified = False

        while attempt <= args.max_retries:
            # Asymmetric padding: less at start, more at end
            crop_start = max(0, abs_word_start - pad)
            crop_end = abs_word_end + pad * 1.5  # extra at end
            duration = crop_end - crop_start

            extract_wav(args.audio, crop_start, duration, str(out_path))

            # Verify with Whisper
            verify_words = whisper_words(str(out_path), args.whisper_model)
            match_ratio = verify_text(verify_words, clip["db_text"])
            final_text = "".join(
                w["word"] for w in verify_words
            ).strip() if verify_words else ""

            if match_ratio >= 0.80:
                verified = True
                break

            attempt += 1
            pad += 0.4  # extend 400ms each retry

        duration = crop_end - crop_start
        old_dur = clip["duration"]

        if attempt > 0 and verified:
            tag = f"EXT(x{attempt})"
            stats["extended"] += 1
        elif not verified:
            tag = "UNVERIFIED"
        else:
            tag = "ok"
            stats["ok"] += 1

        print(f"  [{i:3d}] {tag:10s} M={score:.2f} V={match_ratio:.2f} "
              f"{duration:.1f}s (was {old_dur:.1f}s) "
              f"\"{final_text[:55]}\"")

        refined.append({
            **clip,
            "duration": round(duration, 2),
            "start": round(crop_start, 3),
            "end": round(crop_end, 3),
            "whisper_text": whisper_text,
            "verified_text": final_text,
            "match_score": round(score, 3),
            "verify_score": round(match_ratio, 3),
            "refined": True,
        })

    # Save manifest
    refined_manifest = {**manifest, "clips": refined, "refined": True}
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(refined_manifest, f, indent=2)

    total = sum(c["duration"] for c in refined)
    print(f"\nRefined {len(refined)} clips ({total:.0f}s total)")
    print(f"  OK first try:  {stats['ok']}")
    print(f"  Extended:      {stats['extended']}")
    print(f"  Unchanged:     {stats['unchanged']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
