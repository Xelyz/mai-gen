"""
Maimai chart tokenizer.

Converts a parsed maimai chart (from simai_interpret.py) into a flat
sequence of discrete tokens ordered by time.

Token patterns (in emission order):
  TAP:   <tap> <position {1-8}> <time {t}>  [<break>] [<slide_head>]
  HOLD:  <hold> <position {1-8}> <time {t}> <end_at> <time {t_end}>
  SLIDE: <slide> <time {t}> <head {position}> <shape> <tail {position}>

A slide note is decomposed into:
  1. A tap with <slide_head> at the note's timestamp
  2. A <slide> token at slideStartTime

Time values are whole numbers in units of 10 ms (rounded).
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
#  Import the simai parser
# ---------------------------------------------------------------------------
import maidata2objects as m2o

# ---------------------------------------------------------------------------
#  Slide shape extraction (adapted from convertor.py parse_slide_info)
# ---------------------------------------------------------------------------
# Canonical shape vocabulary – keep in sync with mai/data/utils.py
SHAPE_VOCAB = {
    '-', 'v', 's', 'z', 'V1', 'V2', 'w',
    '<', '>',
    'p', 'q', 'pp', 'qq',
}


def _parse_slide_info(note_content: str, start_position: int) -> Tuple[str, int]:
    """
    Extract (shape_str, target_position) from a Simai noteContent string.

    Returns ('?', 0) when the content cannot be parsed.
    """
    if not note_content or len(note_content) < 1:
        return '?', 0

    # Strip duration brackets
    content = re.sub(r'\[.*?\]', '', note_content)

    # --- V-shape: format "startV extraDigit targetDigit" (e.g. "1V35") ---
    match_v = re.match(r'^(\d)V(\d)(\d)$', content)
    if match_v:
        try:
            start = int(match_v.group(1))
            extra = int(match_v.group(2))
            target = int(match_v.group(3))
            if not (1 <= start <= 8 and 1 <= extra <= 8 and 1 <= target <= 8):
                return '?', 0
            diff = (extra - start + 8) % 8
            shape = 'V1' if diff == 2 else 'V2'
            return shape, target
        except ValueError:
            return '?', 0

    # --- Circle shapes: "start [<>^] target" ---
    match_circle = re.match(r'^(\d)([<>^])(\d)$', content)
    if match_circle:
        try:
            start = int(match_circle.group(1))
            shape = match_circle.group(2)
            target = int(match_circle.group(3))

            # Mirror < / > for positions 3-6
            if shape in ['>', '<'] and start in [3, 4, 5, 6]:
                shape = '<' if shape == '>' else '>'

            if shape == '^':
                # Determine clockwise vs counter-clockwise
                def _is_cw(p1: int, p2: int) -> bool:
                    if p1 == p2:
                        return False
                    diff = ((p2 - 1) * 45 - (p1 - 1) * 45 + 180) % 360 - 180
                    return diff > 0

                shape = '>' if _is_cw(start, target) else '<'

            return shape, target
        except ValueError:
            return '?', 0

    # --- Other shapes: "start [shape 1-2 chars] target" ---
    match_other = re.match(r'^(\d)([pqwszv-]{1,2})(\d)$', content)
    if match_other:
        try:
            shape = match_other.group(2) or '-'
            target = int(match_other.group(3))
            if not (1 <= target <= 8):
                target = 0
            if shape not in SHAPE_VOCAB:
                shape = '?'
            if shape in ['V1', 'V2']:
                shape = '?'
            return shape, target
        except ValueError:
            return '?', 0

    return '?', 0


# ---------------------------------------------------------------------------
#  Token types
# ---------------------------------------------------------------------------
def format_tap(pos, is_break=False, is_slide_head=False):
    prefix = 'slide_head' if is_slide_head else 'tap'
    suffix = '_break' if is_break else ''
    return f'<{prefix}{suffix}_p{pos}>'

def format_hold(pos):
    return f'<hold_p{pos}>'

def format_slide(head, tail, shape):
    return f'<slide_h{head}_t{tail}_{shape}>'

TOKEN_TIME = '<time {}>'

def _time_to_token_val(time_seconds: float) -> int:
    """Convert a time in seconds to a whole-number 10 ms unit (rounded)."""
    return round(time_seconds * 100)  # seconds -> 10ms units


# ---------------------------------------------------------------------------
#  Core tokenizer
# ---------------------------------------------------------------------------
@dataclass
class _TokenEvent:
    """Intermediate representation: a token sequence anchored to a time."""
    time_val: int        # in 10ms units, used for sorting only
    sort_priority: int   # tiebreaker: tap(0) before slide(1)
    tokens: List[str]


def tokenize_chart(maidata_path: str, difficulty) -> Optional[List[str]]:
    """
    Parse a maidata.txt file for the given difficulty and return a flat
    list of tokens in time order.

    Args:
        maidata_path: path to maidata.txt
        difficulty: int (0-6) or str ('EASY', ..., 'Re_MASTER', 'ORIGINAL')

    Returns:
        List of token strings, or None on failure.
    """
    # --- resolve difficulty index ---
    diff_map = {
        'EASY': 0, 'BASIC': 1, 'ADVANCED': 2, 'EXPERT': 3,
        'MASTER': 4, 'Re_MASTER': 5, 'ORIGINAL': 6,
    }
    if isinstance(difficulty, str):
        diff_idx = diff_map.get(difficulty)
        if diff_idx is None:
            print(f"Error: invalid difficulty name: {difficulty}", file=sys.stderr)
            return None
    else:
        diff_idx = int(difficulty)
        if not (0 <= diff_idx <= 6):
            print(f"Error: difficulty index out of range (0-6): {diff_idx}", file=sys.stderr)
            return None

    # --- read & serialize ---
    if not os.path.exists(maidata_path):
        print(f"Error: file not found: {maidata_path}", file=sys.stderr)
        return None

    if not m2o.read_data(maidata_path):
        print(f"Error: failed to read maidata: {maidata_path}", file=sys.stderr)
        return None

    if diff_idx >= len(m2o.fumens) or not m2o.fumens[diff_idx]:
        m2o.clear_data()
        print(f"Error: difficulty {difficulty} not found in {maidata_path}", file=sys.stderr)
        return None

    fumen = m2o.fumens[diff_idx]
    try:
        m2o.serialize(fumen)
    except Exception as e:
        m2o.clear_data()
        print(f"Error: serialization failed: {e}", file=sys.stderr)
        return None

    if not m2o.notelist:
        m2o.clear_data()
        print(f"Error: no notes found for difficulty {difficulty}", file=sys.stderr)
        return None

    # --- collect events ---
    events: List[_TokenEvent] = []

    for tp in m2o.notelist:
        notes = tp.getNotes()
        if not notes:
            continue

        timestamp_s = tp.time  # seconds (float)

        for note in notes:
            nt = note.noteType  # SimaiNoteType enum
            pos = note.startPosition  # 1-8

            if nt == m2o.SimaiNoteType.Tap:
                t_val = _time_to_token_val(timestamp_s)
                toks = [
                    format_tap(pos, is_break=note.isBreak, is_slide_head=False),
                    TOKEN_TIME.format(t_val)
                ]
                events.append(_TokenEvent(time_val=t_val, sort_priority=0, tokens=toks))

            elif nt == m2o.SimaiNoteType.Hold:
                t_start = _time_to_token_val(timestamp_s)
                t_end = _time_to_token_val(timestamp_s + note.holdTime)
                toks = [
                    format_hold(pos),
                    TOKEN_TIME.format(t_start),
                    TOKEN_TIME.format(t_end)
                ]
                events.append(_TokenEvent(time_val=t_start, sort_priority=0, tokens=toks))

            elif nt == m2o.SimaiNoteType.Slide:
                if not note.isSlideNoHead:
                    t_tap = _time_to_token_val(timestamp_s)
                    tap_toks = [
                        format_tap(pos, is_break=note.isBreak, is_slide_head=True),
                        TOKEN_TIME.format(t_tap)
                    ]
                    events.append(_TokenEvent(time_val=t_tap, sort_priority=0, tokens=tap_toks))

                shape_str, target_pos = _parse_slide_info(note.noteContent, pos)
                slide_start_s = note.slideStartTime
                t_slide = _time_to_token_val(slide_start_s)
                t_slide_end = _time_to_token_val(slide_start_s + note.slideTime)

                slide_toks = [
                    format_slide(pos, target_pos, shape_str),
                    TOKEN_TIME.format(t_slide),
                    TOKEN_TIME.format(t_slide_end)
                ]
                events.append(_TokenEvent(time_val=t_slide, sort_priority=1, tokens=slide_toks))

            # Touch / TouchHold are currently not tokenized (not in spec).
            # They can be added later if needed.

    m2o.clear_data()

    # --- sort by time, tiebreak tap before slide ---
    events.sort(key=lambda e: (e.time_val, e.sort_priority))

    # --- flatten ---
    tokens: List[str] = []
    for ev in events:
        tokens.extend(ev.tokens)

    return tokens


# ---------------------------------------------------------------------------
#  Vocabulary helpers
# ---------------------------------------------------------------------------
def build_vocab() -> Dict[str, int]:
    """
    Build a deterministic token-to-id vocabulary covering all possible tokens.
    Includes time tokens up to <time 18000> (180 seconds / 3 minutes).
    Returns a dict mapping token string -> integer id.
    """
    vocab: List[str] = [
        '<pad>', '<bos>', '<eos>', '<unk>'
    ]
    # Taps
    for p in range(1, 9):
        vocab.append(f'<tap_p{p}>')
        vocab.append(f'<tap_break_p{p}>')
        vocab.append(f'<slide_head_p{p}>')
        vocab.append(f'<slide_head_break_p{p}>')
        
    # Holds
    for p in range(1, 9):
        vocab.append(f'<hold_p{p}>')
        
    # Slides
    shapes = sorted(list(SHAPE_VOCAB)) + ['?']
    for h in range(1, 9):
        for t in range(1, 9):
            for s in shapes:
                vocab.append(f'<slide_h{h}_t{t}_{s}>')
                
    # Times
    for t in range(18001):
        vocab.append(TOKEN_TIME.format(t))

    return {tok: i for i, tok in enumerate(vocab)}


# ---------------------------------------------------------------------------
#  Detokenizer: tokens -> chart
# ---------------------------------------------------------------------------
_RE_TIME = re.compile(r'^<time (-?\d+)>$')
_RE_TAP = re.compile(r'^<(tap|slide_head)(_break)?_p(\d+)>$')
_RE_HOLD = re.compile(r'^<hold_p(\d+)>$')
_RE_SLIDE = re.compile(r'^<slide_h(\d+)_t(\d+)_(.+)>$')

def _match_int(token: str, pattern) -> Optional[int]:
    m = pattern.match(token)
    return int(m.group(1)) if m else None

def detokenize(tokens: List[str]) -> List[dict]:
    notes: List[dict] = []
    i = 0
    skipped = 0
    while i < len(tokens):
        tok = tokens[i]
        
        m_tap = _RE_TAP.match(tok)
        if m_tap:
            if i + 1 < len(tokens):
                t_tok = tokens[i+1]
                t_val = _match_int(t_tok, _RE_TIME)
                if t_val is not None:
                    notes.append({
                        'type': 'tap',
                        'position': int(m_tap.group(3)),
                        'time_10ms': t_val,
                        'is_break': bool(m_tap.group(2)),
                        'is_slide_head': (m_tap.group(1) == 'slide_head')
                    })
                    i += 2
                    continue
                    
        m_hold = _RE_HOLD.match(tok)
        if m_hold:
            if i + 2 < len(tokens):
                t1 = _match_int(tokens[i+1], _RE_TIME)
                t2 = _match_int(tokens[i+2], _RE_TIME)
                if t1 is not None and t2 is not None:
                    notes.append({
                        'type': 'hold',
                        'position': int(m_hold.group(1)),
                        'time_10ms': t1,
                        'end_time_10ms': t2
                    })
                    i += 3
                    continue
                    
        m_slide = _RE_SLIDE.match(tok)
        if m_slide:
            if i + 2 < len(tokens):
                t1 = _match_int(tokens[i+1], _RE_TIME)
                t2 = _match_int(tokens[i+2], _RE_TIME)
                if t1 is not None and t2 is not None:
                    notes.append({
                        'type': 'slide',
                        'head_position': int(m_slide.group(1)),
                        'tail_position': int(m_slide.group(2)),
                        'shape': m_slide.group(3),
                        'time_10ms': t1,
                        'end_time_10ms': t2
                    })
                    i += 3
                    continue
                    
        # If we reach here, it's malformed or skip
        i += 1
        skipped += 1

    if skipped:
        print(f"Warning: skipped {skipped} unexpected/malformed tokens", file=sys.stderr)
    notes.sort(key=lambda n: n['time_10ms'])
    return notes


def notes_to_hit_objects(notes: List[dict]):
    """
    Convert detokenized note dicts to HitObject instances
    (from mai.data.utils), suitable for gridify + save_maimai_file.
    """
    from chart_utils import HitObject

    hit_objects: List[HitObject] = []
    for n in notes:
        t_ms = n['time_10ms'] * 10.0

        if n['type'] == 'tap':
            obj = HitObject(
                timeStamp=t_ms,
                noteType=0,
                startPosition=n['position'],
                isBreak=n.get('is_break', False),
                isSlideStart=n.get('is_slide_head', False),
            )
            hit_objects.append(obj)

        elif n['type'] == 'hold':
            end_ms = n['end_time_10ms'] * 10.0
            obj = HitObject(
                timeStamp=t_ms,
                noteType=2,
                startPosition=n['position'],
                holdTime=end_ms - t_ms,
            )
            hit_objects.append(obj)

        elif n['type'] == 'slide':
            end_ms = n['end_time_10ms'] * 10.0
            obj = HitObject(
                timeStamp=t_ms,
                noteType=1,
                startPosition=n['head_position'],
                isSlideNoHead=True,
                slideStartTime=t_ms,
                slideTime=end_ms - t_ms,
                slideShape=n['shape'],
                slideTargetID=n['tail_position'],
            )
            hit_objects.append(obj)

    hit_objects.sort(key=lambda x: (x.timeStamp, x.startPosition or 0))
    return hit_objects


def save_chart(tokens: List[str], output_path: str):
    """
    Detokenize and write a maidata chart file.

    Uses gridify from mai.data.utils to quantize timing to beats,
    then writes standard maidata format.
    """
    from chart_utils import HitObject, Beats, gridify

    notes = detokenize(tokens)
    if not notes:
        print(f"Error: no valid notes to save", file=sys.stderr)
        return

    hit_objects = notes_to_hit_objects(notes)
    if not hit_objects:
        print(f"Error: no hit objects produced", file=sys.stderr)
        return

    print(f"Detokenized {len(hit_objects)} hit objects")

    try:
        bpm, offset, hit_objects = gridify(hit_objects)
    except Exception:
        import traceback
        traceback.print_exc()
        bpm = 120
        offset = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"&first={offset / 1000}\n")
        f.write(f"&inote_5=({bpm:.3f})\n")

        prev = Beats()
        prev_content = ''
        curr_divide = None
        first = True
        for obj in hit_objects:
            interval = obj.timeStampInBeats - prev
            if interval.count > 0:
                if interval.divide != curr_divide:
                    f.write(f'{{{interval.divide}}}')
                    curr_divide = interval.divide
                f.write(f'{prev_content}')
                f.write(',' * interval.count)
                prev = obj.timeStampInBeats
                prev_content = obj.get_note_content()
            else:
                if first:
                    prev = obj.timeStampInBeats
                    prev_content = obj.get_note_content()
                else:
                    prev_content += f'/{obj.get_note_content()}'
            first = False

        f.write(prev_content)
        f.write(',E')

    print(f"Saved chart to {output_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a maimai chart from maidata.txt into a flat token sequence."
    )
    parser.add_argument("input", help="Path to maidata.txt")
    parser.add_argument(
        "--difficulty", "-d", default="MASTER",
        help="Difficulty index (0-6) or name (EASY/BASIC/ADVANCED/EXPERT/MASTER/Re_MASTER/ORIGINAL). Default: MASTER"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path. If not specified, prints to stdout."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON array instead of one-token-per-line."
    )
    args = parser.parse_args()

    # try to parse difficulty as int
    diff = args.difficulty
    try:
        diff = int(diff)
    except ValueError:
        pass

    tokens = tokenize_chart(args.input, diff)
    if tokens is None:
        sys.exit(1)

    if args.json:
        output_str = json.dumps(tokens, ensure_ascii=False, indent=2)
    else:
        output_str = '\n'.join(tokens)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
            f.write('\n')
        print(f"Wrote {len(tokens)} tokens to {args.output}")
    else:
        print(output_str)
        print(f"\n--- Total: {len(tokens)} tokens ---", file=sys.stderr)


if __name__ == "__main__":
    main()
