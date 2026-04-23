"""
Self-contained chart utilities for maimai tokenizer/detokenizer.
"""

import time as _time
import re
import math
import functools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from sklearn.linear_model import LinearRegression


# ============================================================================
#  Slide shape mappings
# ============================================================================
SLIDE_SHAPE_TO_ID = {
    '-': 1, 'v': 2, 's': 3, 'z': 4, 'V1': 5, 'V2': 6, 'w': 7,
    '<': 8, '>': 9,
    'p': 10, 'q': 11,
    'pp': 12, 'qq': 13,
    '?': 0  # unknown / default
}
ID_TO_SLIDE_SHAPE = {v: k for k, v in SLIDE_SHAPE_TO_ID.items()}


# ============================================================================
#  Beats (from mai/data/utils.py)
# ============================================================================
@functools.total_ordering
class Beats:
    def __init__(self, divide=4, count=0):
        if divide == 0 and count != 0:
            raise ValueError("Beats denominator cannot be zero unless count is also zero.")
        self.divide = divide
        self.count = count
        self.reduce()

    def value(self):
        if self.divide == 0:
            return 0.0
        return self.count / self.divide

    def reduce(self):
        if self.count == 0:
            self.divide = 1
            return
        if self.divide == 0:
            raise ValueError("Cannot reduce Beats with zero denominator and non-zero numerator.")
        common = math.gcd(abs(self.count), abs(self.divide))
        if common != 0:
            self.count //= common
            self.divide //= common
        if self.divide < 0:
            self.divide = -self.divide
            self.count = -self.count

    def __repr__(self):
        return f"{self.count}/{self.divide}"

    def __add__(self, other):
        if isinstance(other, Beats):
            d1, d2 = self.divide, other.divide
            new_divide = math.lcm(d1, d2)
            if new_divide == 0:
                new_divide = max(d1, d2, 1)
            new_count = (new_divide // d1 * self.count) + (new_divide // d2 * other.count)
            return Beats(new_divide, new_count)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Beats):
            d1, d2 = self.divide, other.divide
            new_divide = math.lcm(d1, d2)
            if new_divide == 0:
                new_divide = max(d1, d2, 1)
            new_count = (new_divide // d1 * self.count) - (new_divide // d2 * other.count)
            return Beats(new_divide, new_count)
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Beats):
            return NotImplemented
        return self.count * other.divide == other.count * self.divide

    def __lt__(self, other):
        if not isinstance(other, Beats):
            return NotImplemented
        return self.count * other.divide < other.count * self.divide


# ============================================================================
#  HitObject (from mai/data/utils.py)
# ============================================================================
@dataclass
class HitObject:
    timeStamp: float = 0.0
    timeStampInBeats: Beats = field(default_factory=lambda: Beats(4, 0))
    holdTime: float = 0.0
    holdTimeInBeats: Beats = field(default_factory=lambda: Beats(4, 0))
    isBreak: bool = False
    isEx: bool = False
    isHanabi: bool = False
    isSlideNoHead: bool = False
    isSlideStart: bool = False
    noteType: int = 0
    slideStartTime: float = 0.0
    slideStartTimeInBeats: Beats = field(default_factory=lambda: Beats(4, 0))
    slideTime: float = 0.0
    slideTimeInBeats: Beats = field(default_factory=lambda: Beats(4, 0))
    slideShape: str = " "
    startPosition: int = 0
    slideTargetID: int = 0

    def __str__(self):
        return str({
            "timeStamp": self.timeStamp,
            "timeStampInBeats": self.timeStampInBeats,
            "holdTime": self.holdTime,
            "holdTimeInBeats": self.holdTimeInBeats,
            "isBreak": self.isBreak,
            "isEx": self.isEx,
            "isHanabi": self.isHanabi,
            "isSlideNoHead": self.isSlideNoHead,
            "noteType": self.noteType,
            "slideStartTime": self.slideStartTime,
            "slideTime": self.slideTime,
            "slideShape": self.slideShape,
            "startPosition": self.startPosition,
        })

    def get_note_content(self):
        if self.noteType in [0, 1, 2]:
            content = str(self.startPosition)
            if self.isBreak:
                content += "b"
            if self.isEx:
                content += "x"
            if self.isSlideStart and self.noteType == 0:
                content += "$$"

        if self.noteType in [2, 4]:
            content += "h"
            if self.holdTimeInBeats.count != 0:
                content += f"[{self.holdTimeInBeats.divide}:{self.holdTimeInBeats.count}]"

        if self.noteType == 1:
            if self.isSlideNoHead:
                content += "?"
            if "V" in self.slideShape:
                content += "V"
                content += str((self.startPosition + (2 if self.slideShape[1] == '1' else -2)) % 8)
            elif self.slideShape in ['>', '<']:
                if self.startPosition in [3, 4, 5, 6]:
                    content += '<' if self.slideShape == '>' else '>'
                else:
                    content += self.slideShape
            else:
                content += self.slideShape

            content += str(self.slideTargetID)
            content += f"[{self.slideTimeInBeats.divide}:{self.slideTimeInBeats.count}]"

        return content


# ============================================================================
#  Timing detection (from mug/data/utils.py)
# ============================================================================
def test_timing(time_list, test_bpm, test_offset, div, refine):
    cur_offset = test_offset
    cur_bpm = test_bpm

    epsilon = 10
    gap = 60 * 1000 / (test_bpm * div)
    delta_time_list = time_list - test_offset
    meter_list = delta_time_list / gap
    meter_list_round = np.round(meter_list)
    timing_error = np.abs(meter_list - meter_list_round)
    valid = (timing_error < epsilon / gap).astype(np.int32)
    valid_count = np.sum(valid)

    if valid_count >= 2 and refine:
        rgs = LinearRegression(fit_intercept=True)
        rgs.fit(meter_list_round.reshape((-1, 1)), time_list, sample_weight=valid)
        if not np.isinf(rgs.coef_) and not np.isnan(rgs.coef_) and rgs.coef_[0] != 0:
            cur_offset = rgs.intercept_
            cur_bpm = 60000 / rgs.coef_[0] / 4

            while cur_bpm < 150:
                cur_bpm = cur_bpm * 2
            while cur_bpm >= 300:
                cur_bpm = cur_bpm / 2

    valid_ratio = valid_count / test_bpm
    return valid_ratio, valid, cur_bpm, cur_offset


def timing(time_list, verbose=True):
    offset = time_list[0]

    best_bpm = None
    best_offset = None
    best_valid_ratio = -1

    st = _time.time()
    for test_bpm in np.arange(150, 300, 0.1):
        valid_ratio, valid, cur_bpm, cur_offset = test_timing(
            time_list, test_bpm, offset, div=1, refine=False)

        if valid_ratio > best_valid_ratio:
            valid_ratio, valid, cur_bpm, cur_offset = test_timing(
                time_list, test_bpm, offset, div=1, refine=True)
            best_valid_ratio = valid_ratio
            best_bpm = cur_bpm
            best_offset = cur_offset
            if verbose:
                print(f"[valid: {valid_ratio} / {len(valid)}] bpm {test_bpm} -> {cur_bpm}, "
                      f"offset {offset} -> {cur_offset}")

        gap = 60000 / cur_bpm
        for test_offset in np.arange(best_offset, best_offset - gap, -gap / 4):
            valid_ratio, valid, cur_bpm, cur_offset = test_timing(
                time_list, cur_bpm, test_offset, div=1, refine=False)
            if valid_ratio > best_valid_ratio:
                valid_ratio, valid, cur_bpm, cur_offset = test_timing(
                    time_list, cur_bpm, test_offset, div=1, refine=True)
                best_valid_ratio = valid_ratio
                best_bpm = cur_bpm
                best_offset = cur_offset
                if verbose:
                    print(f"[valid: {valid_ratio} / {len(valid)}] bpm {best_bpm} -> {cur_bpm}, "
                          f"offset {offset} -> {cur_offset}")

    _, valid_8, best_bpm, best_offset = test_timing(
        time_list, best_bpm, best_offset, div=16, refine=False)
    _, valid_6, best_bpm, best_offset = test_timing(
        time_list, best_bpm, best_offset, div=6, refine=False)
    valid = np.clip(valid_6 + valid_8, 0, 1)

    if verbose:
        print("Test time:", _time.time() - st)
        print(f"Final bpm: {best_bpm}, offset: {best_offset}")
        print(f"Final valid: {np.sum(valid)} / {len(valid)}")
        print(f"Invalid: {time_list[valid == 0]}")

    return best_bpm, best_offset


# ============================================================================
#  Gridify (from mai/data/utils.py)
# ============================================================================
epsilon = 10


def parse_hit_objects(obj: HitObject):
    if obj is None:
        return None, None, None
    return obj.timeStamp, obj.startPosition, obj.holdTime


def gridify(hit_objects: list, verbose=True):
    if not hit_objects:
        raise ValueError("gridify received an empty hit_objects list, nothing to reconstruct.")

    times = []
    for obj in hit_objects:
        st, _, _ = parse_hit_objects(obj)
        times.append(st)
    times = np.asarray(times, dtype=np.float32)
    bpm, offset = timing(times, verbose)

    def format_time(t, _offset):
        for div in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
            gap = 60 * 1000 / (bpm * div / 4)
            meter = (t - _offset) / gap
            meter_round = round(meter)
            timing_error = abs(meter - meter_round)
            if timing_error < epsilon / gap:
                return str(int(meter_round * gap + _offset)), Beats(div, meter_round)
        div = 256
        gap = 60 * 1000 / (bpm * div / 4)
        meter = (t - _offset) / gap
        meter_round = round(meter)
        return int(t), Beats(div, meter_round)

    new_hit_objects: list = []
    for obj in hit_objects:
        obj.timeStamp, obj.timeStampInBeats = format_time(obj.timeStamp, offset)
        if obj.noteType == 1:
            obj.slideStartTime, obj.slideStartTimeInBeats = format_time(obj.slideStartTime, offset)
            obj.slideTime, obj.slideTimeInBeats = format_time(obj.slideTime, 0)
            obj.timeStampInBeats = obj.slideStartTimeInBeats - Beats(4, 1)
        if obj.noteType in [2, 4]:
            obj.holdTime, obj.holdTimeInBeats = format_time(obj.holdTime, 0)
        new_hit_objects.append(obj)

    new_hit_objects.sort(key=lambda x: (x.timeStampInBeats, x.startPosition if x.startPosition else 0))

    return bpm, offset, new_hit_objects
