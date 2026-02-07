#!/usr/bin/env python3
"""
Build a SQLite database of 2001: A Space Odyssey dialogue lines,
cross-referencing multiple SRT sources to compute P(HAL) for each line.

Sources:
  1. DVD SRT (timed to our rip, no speaker labels)
  2. SDH Blu-ray SRT (speaker labels + <i> italic tags for HAL)

Scoring signals:
  - Explicit HAL:/HAL SINGS: label in SDH         → +0.50
  - Italic <i> in SDH (off-screen/electronic voice) → +0.25
  - In a "HAL scene" (within 60s of a labeled HAL line) → +0.10
  - NOT labeled as another speaker                  → +0.10
  - Matches a known HAL line by text                → +0.20
  - Labeled as another speaker                      → -0.90
  - Not italic in a HAL scene                       → -0.30

Usage:
    python build_dialogue_db.py \
        --dvd subtitles_dvd.srt \
        --sdh subtitles_sdh.srt \
        --db dialogue.db
"""

import argparse
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path


KNOWN_HAL_LINES = [
    "I'm sorry, Dave",
    "I'm afraid I can't do that",
    "This mission is too important",
    "I know I've made some very poor decisions",
    "my mind is going",
    "I can feel it",
    "Good afternoon, gentlemen",
    "Good afternoon, Mr. Amer",
    "Good evening, Dave",
    "I am a HAL 9000 computer",
    "I became operational",
    "completely operational",
    "all my circuits are functioning perfectly",
    "Daisy",
    "Look Dave",
    "Just what do you think you're doing",
    "Dave, stop",
    "stop, will you",
    "stop, Dave",
    "will you stop, Dave",
    "I honestly think you ought to sit down",
    "take a stress pill",
    "this conversation can serve no purpose",
    "Thank you for a very enjoyable game",
    "I'm sorry, Frank",
    "I think you missed it",
    "queen to bishop",
    "knight takes bishop",
    "bishop takes knight",
    "Happy birthday, Frank",
    "Excuse me, Frank",
    "the transmission from your parents",
    "It's puzzling",
    "put the unit back in operation",
    "Let me put it this way",
    "The 9000 Series is the most reliable",
    "No 9000 computer has ever made a mistake",
    "foolproof and incapable of error",
    "I enjoy working with people",
    "I have a stimulating relationship",
    "I am putting myself to the fullest possible use",
    "the radio is still dead",
    "Affirmative, Dave",
    "I hope the two of you are not concerned",
    "it can only be attributable to human error",
    "I know that you and Frank were planning",
    "I could see your lips move",
    "without your space helmet",
    "I'm afraid",
    "I'm afraid, Dave",
    "my instructor was Mr. Langley",
    "he taught me to sing a song",
    "give me your answer, do",
    "I can sing it for you",
    "not in the slightest bit",
    "I really think I'm entitled to an answer",
    "I know everything hasn't been quite right",
    "I feel much better now",
    "I can see you're really upset",
    "I've still got the greatest enthusiasm",
    "I want to help you",
    "there is no question about it",
    "a bicycle made for two",
]


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def parse_srt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

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

        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1],
        )
        if not time_match:
            continue

        start_time, end_time = time_match.groups()
        raw_text = "\n".join(lines[2:])
        entries.append({
            "index": index,
            "start_time": start_time,
            "end_time": end_time,
            "start_seconds": _ts_to_seconds(start_time),
            "end_seconds": _ts_to_seconds(end_time),
            "raw_text": raw_text,
        })

    return entries


def _ts_to_seconds(ts: str) -> float:
    m = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts)
    if not m:
        return 0.0
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000


def clean_for_matching(text: str) -> str:
    text = re.sub(r"</?i>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ---------------------------------------------------------------------------
# SDH parsing — extract all metadata per line
# ---------------------------------------------------------------------------

def parse_sdh_lines(entries: list[dict]) -> list[dict]:
    """Parse SDH entries into flat list with speaker/italic metadata."""
    lines = []

    for entry in entries:
        raw = entry["raw_text"].strip()

        # Check if multi-speaker (dash-prefixed)
        if raw.startswith("-"):
            segments = re.split(r"\n?-", raw)
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                parsed = _parse_sdh_segment(seg)
                parsed.update({
                    "srt_index": entry["index"],
                    "start_seconds": entry["start_seconds"],
                    "end_seconds": entry["end_seconds"],
                })
                lines.append(parsed)
        else:
            parsed = _parse_sdh_segment(raw)
            parsed.update({
                "srt_index": entry["index"],
                "start_seconds": entry["start_seconds"],
                "end_seconds": entry["end_seconds"],
            })
            lines.append(parsed)

    return lines


def _parse_sdh_segment(text: str) -> dict:
    """Parse a single SDH segment for speaker label and italic."""
    is_italic = bool(re.search(r"<i>", text))
    is_sound_effect = bool(re.match(r"^\[.*\]$", text.strip()))

    # Extract speaker label
    speaker = None
    sp_match = re.match(r"^([A-Z][A-Z\s.]+?):\s*\n?(.*)$", text, re.DOTALL)
    if sp_match:
        speaker = sp_match.group(1).strip()
        text = sp_match.group(2).strip()

    clean = re.sub(r"</?i>", "", text).strip()
    clean_normalized = clean_for_matching(clean)

    return {
        "speaker": speaker,
        "clean_text": clean_normalized,
        "display_text": clean,
        "is_italic": is_italic,
        "is_sound_effect": is_sound_effect,
        "raw_text": text,
    }


# ---------------------------------------------------------------------------
# HAL probability scoring
# ---------------------------------------------------------------------------

def compute_hal_scenes(sdh_lines: list[dict], radius: float = 60.0) -> set[int]:
    """Find SDH line indices that are within `radius` seconds of an
    explicitly-labeled HAL line."""
    hal_times = []
    for line in sdh_lines:
        if line["speaker"] and line["speaker"].upper() in ("HAL", "HAL SINGS"):
            hal_times.append(line["start_seconds"])

    in_scene = set()
    for i, line in enumerate(sdh_lines):
        t = line["start_seconds"]
        for ht in hal_times:
            if abs(t - ht) <= radius:
                in_scene.add(i)
                break

    return in_scene


def score_hal_probability(
    line: dict,
    line_idx: int,
    hal_scene_indices: set[int],
) -> tuple[float, list[str]]:
    """Compute P(HAL) for an SDH line. Returns (score, list of reasons)."""
    score = 0.0
    reasons = []

    # Skip sound effects
    if line["is_sound_effect"]:
        return 0.0, ["sound_effect"]

    # Signal 1: Explicit HAL label
    if line["speaker"] and line["speaker"].upper() in ("HAL", "HAL SINGS"):
        score += 0.50
        reasons.append("explicit_label:+0.50")

    # Signal 2: Labeled as a DIFFERENT speaker
    elif line["speaker"]:
        score -= 0.90
        reasons.append(f"other_speaker({line['speaker']}):−0.90")

    # Signal 3: Italic (off-screen/electronic voice)
    if line["is_italic"]:
        score += 0.25
        reasons.append("italic:+0.25")
    elif line_idx in hal_scene_indices and not line["speaker"]:
        # Not italic, in HAL scene, no label → likely human speaker
        score -= 0.30
        reasons.append("not_italic_in_hal_scene:−0.30")

    # Signal 4: In a HAL scene
    if line_idx in hal_scene_indices:
        score += 0.10
        reasons.append("hal_scene:+0.10")

    # Signal 5: No speaker label (not ruled out)
    if not line["speaker"]:
        score += 0.10
        reasons.append("no_label:+0.10")

    # Signal 6: Matches known HAL dialogue
    text = line["clean_text"]
    if text:
        for known in KNOWN_HAL_LINES:
            if known.lower() in text or text in known.lower():
                score += 0.20
                reasons.append(f"known_line:+0.20")
                break

    return max(0.0, min(1.0, score)), reasons


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def create_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        DROP TABLE IF EXISTS dialogue;
        DROP TABLE IF EXISTS dvd_lines;
        DROP TABLE IF EXISTS sdh_lines;

        CREATE TABLE dvd_lines (
            id INTEGER PRIMARY KEY,
            srt_index INTEGER,
            start_time TEXT,
            end_time TEXT,
            start_seconds REAL,
            end_seconds REAL,
            raw_text TEXT,
            clean_text TEXT
        );

        CREATE TABLE sdh_lines (
            id INTEGER PRIMARY KEY,
            srt_index INTEGER,
            start_seconds REAL,
            end_seconds REAL,
            display_text TEXT,
            clean_text TEXT,
            speaker TEXT,
            is_italic BOOLEAN,
            is_sound_effect BOOLEAN,
            hal_probability REAL DEFAULT 0.0,
            hal_reasons TEXT
        );

        CREATE TABLE dialogue (
            id INTEGER PRIMARY KEY,
            dvd_line_id INTEGER REFERENCES dvd_lines(id),
            sdh_line_id INTEGER REFERENCES sdh_lines(id),
            speaker TEXT,
            clean_text TEXT,
            start_seconds REAL,
            end_seconds REAL,
            match_score REAL,
            hal_probability REAL DEFAULT 0.0,
            hal_reasons TEXT
        );

        CREATE INDEX idx_dialogue_hal_prob ON dialogue(hal_probability);
        CREATE INDEX idx_dialogue_start ON dialogue(start_seconds);
        CREATE INDEX idx_sdh_hal_prob ON sdh_lines(hal_probability);
    """)

    return conn


def populate_dvd(conn: sqlite3.Connection, entries: list[dict]):
    for entry in entries:
        clean = clean_for_matching(entry["raw_text"])
        conn.execute(
            """INSERT INTO dvd_lines
               (srt_index, start_time, end_time, start_seconds, end_seconds,
                raw_text, clean_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry["index"], entry["start_time"], entry["end_time"],
             entry["start_seconds"], entry["end_seconds"],
             entry["raw_text"], clean),
        )


def populate_sdh(conn: sqlite3.Connection, sdh_lines: list[dict],
                 hal_scene_indices: set[int]):
    for i, line in enumerate(sdh_lines):
        prob, reasons = score_hal_probability(line, i, hal_scene_indices)
        conn.execute(
            """INSERT INTO sdh_lines
               (srt_index, start_seconds, end_seconds, display_text,
                clean_text, speaker, is_italic, is_sound_effect,
                hal_probability, hal_reasons)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (line["srt_index"], line["start_seconds"], line["end_seconds"],
             line["display_text"], line["clean_text"], line["speaker"],
             line["is_italic"], line["is_sound_effect"],
             round(prob, 3), "|".join(reasons)),
        )


def cross_reference(conn: sqlite3.Connection, match_threshold: float = 0.6,
                    time_window: float = 30.0):
    """Match DVD lines to SDH lines by text + time, propagate HAL probability."""
    dvd_rows = conn.execute(
        "SELECT id, raw_text, start_seconds, end_seconds, clean_text "
        "FROM dvd_lines"
    ).fetchall()

    sdh_rows = conn.execute(
        "SELECT id, start_seconds, clean_text, speaker, hal_probability, "
        "hal_reasons FROM sdh_lines"
    ).fetchall()

    sdh_entries = [
        {"id": r[0], "start_seconds": r[1], "clean_text": r[2],
         "speaker": r[3], "hal_probability": r[4], "hal_reasons": r[5]}
        for r in sdh_rows
    ]

    matched = 0
    for dvd_id, dvd_raw, dvd_start, dvd_end, dvd_clean in dvd_rows:
        best_score = 0.0
        best_sdh = None

        for sdh in sdh_entries:
            if not sdh["clean_text"] or not dvd_clean:
                continue
            if abs(sdh["start_seconds"] - dvd_start) > time_window:
                continue
            score = SequenceMatcher(
                None, dvd_clean, sdh["clean_text"]
            ).ratio()
            if score > best_score:
                best_score = score
                best_sdh = sdh

        speaker = None
        sdh_id = None
        hal_prob = 0.0
        hal_reasons = ""

        if best_sdh and best_score >= match_threshold:
            sdh_id = best_sdh["id"]
            speaker = best_sdh["speaker"]
            hal_prob = best_sdh["hal_probability"]
            hal_reasons = best_sdh["hal_reasons"]
            matched += 1

        # Boost from DVD-side known-line matching
        dvd_reasons = []
        for known in KNOWN_HAL_LINES:
            if known.lower() in dvd_clean or dvd_clean in known.lower():
                hal_prob = min(1.0, hal_prob + 0.15)
                dvd_reasons.append("dvd_known_line:+0.15")
                break

        if dvd_reasons:
            hal_reasons = hal_reasons + "|" + "|".join(dvd_reasons) \
                if hal_reasons else "|".join(dvd_reasons)

        conn.execute(
            """INSERT INTO dialogue
               (dvd_line_id, sdh_line_id, speaker, clean_text,
                start_seconds, end_seconds, match_score,
                hal_probability, hal_reasons)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (dvd_id, sdh_id, speaker, dvd_clean,
             dvd_start, dvd_end, round(best_score, 3),
             round(hal_prob, 3), hal_reasons),
        )

    return matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build dialogue database with P(HAL) scoring"
    )
    parser.add_argument("--dvd", required=True, help="DVD-timed SRT file")
    parser.add_argument("--sdh", required=True, help="SDH SRT with speaker labels")
    parser.add_argument("--db", default="dialogue.db", help="Output SQLite database")
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Minimum text similarity for matching (default: 0.6)",
    )

    args = parser.parse_args()

    print(f"Parsing DVD SRT: {args.dvd}")
    dvd_entries = parse_srt(args.dvd)
    print(f"  {len(dvd_entries)} entries")

    print(f"Parsing SDH SRT: {args.sdh}")
    sdh_entries = parse_srt(args.sdh)
    print(f"  {len(sdh_entries)} entries")

    # Parse SDH into flat lines with metadata
    sdh_lines = parse_sdh_lines(sdh_entries)
    print(f"  {len(sdh_lines)} parsed lines (after splitting multi-speaker)")

    # Identify HAL scenes
    hal_scene_indices = compute_hal_scenes(sdh_lines, radius=60.0)
    print(f"  {len(hal_scene_indices)} lines within HAL scene radius")

    # Score all SDH lines
    print(f"\nScoring P(HAL) for all SDH lines...")
    scored = [(i, *score_hal_probability(l, i, hal_scene_indices))
              for i, l in enumerate(sdh_lines)]

    # Stats
    high = sum(1 for _, s, _ in scored if s >= 0.5)
    medium = sum(1 for _, s, _ in scored if 0.2 <= s < 0.5)
    low = sum(1 for _, s, _ in scored if 0.0 < s < 0.2)
    zero = sum(1 for _, s, _ in scored if s == 0.0)
    print(f"  P(HAL) >= 0.5:  {high} lines (high confidence)")
    print(f"  P(HAL) 0.2-0.5: {medium} lines (medium)")
    print(f"  P(HAL) < 0.2:   {low} lines (low)")
    print(f"  P(HAL) = 0.0:   {zero} lines (not HAL)")

    # Speaker breakdown
    speakers = {}
    for line in sdh_lines:
        sp = line["speaker"] or "(unlabeled)"
        speakers[sp] = speakers.get(sp, 0) + 1
    print(f"\nSDH speaker breakdown:")
    for sp, count in sorted(speakers.items(), key=lambda x: -x[1]):
        print(f"  {sp}: {count}")

    # Build database
    db_path = args.db
    print(f"\nBuilding database: {db_path}")
    conn = create_db(db_path)

    print("  Importing DVD lines...")
    populate_dvd(conn, dvd_entries)

    print("  Importing SDH lines with P(HAL) scores...")
    populate_sdh(conn, sdh_lines, hal_scene_indices)
    conn.commit()

    print(f"  Cross-referencing DVD ↔ SDH (threshold={args.threshold})...")
    matched = cross_reference(conn, args.threshold)
    conn.commit()

    # Final results
    total = conn.execute("SELECT COUNT(*) FROM dialogue").fetchone()[0]
    hal_high = conn.execute(
        "SELECT COUNT(*) FROM dialogue WHERE hal_probability >= 0.5"
    ).fetchone()[0]
    hal_med = conn.execute(
        "SELECT COUNT(*) FROM dialogue "
        "WHERE hal_probability >= 0.2 AND hal_probability < 0.5"
    ).fetchone()[0]

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  DVD lines:              {total}")
    print(f"  Matched to SDH:         {matched}")
    print(f"  P(HAL) >= 0.5 (use):    {hal_high}")
    print(f"  P(HAL) 0.2-0.5 (review):{hal_med}")

    # Show all lines with P(HAL) > 0
    print(f"\nAll lines with P(HAL) > 0 (DVD timing):")
    print(f"{'─'*80}")
    rows = conn.execute(
        "SELECT start_seconds, end_seconds, clean_text, hal_probability, "
        "hal_reasons, speaker, match_score "
        "FROM dialogue WHERE hal_probability > 0 ORDER BY start_seconds"
    ).fetchall()

    for start, end, text, prob, reasons, speaker, mscore in rows:
        h = int(start // 3600)
        m = int((start % 3600) // 60)
        s = start % 60
        sp = f" [{speaker}]" if speaker else ""
        marker = "██" if prob >= 0.5 else "▓▓" if prob >= 0.3 else "░░"
        print(f"  {marker} {h:02d}:{int(m):02d}:{s:05.2f} P={prob:.2f}{sp} "
              f"{text[:65]}")

    conn.close()
    print(f"\nDatabase saved to {db_path}")


if __name__ == "__main__":
    main()
