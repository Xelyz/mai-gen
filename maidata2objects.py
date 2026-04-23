import io
import os
import sys
from enum import Enum
from typing import List, Optional, Tuple

# translated from C# (majdataEdit)

# --- Enums ---
class SimaiNoteType(Enum):
    Tap = 0
    Slide = 1
    Hold = 2
    Touch = 3
    TouchHold = 4

# --- Data Classes ---
class SimaiNote:
    def __init__(self):
        self.holdTime: float = 0.0
        self.isBreak: bool = False
        self.isEx: bool = False
        self.isFakeRotate: bool = False
        self.isForceStar: bool = False
        self.isHanabi: bool = False
        self.isSlideBreak: bool = False
        self.isSlideNoHead: bool = False
        self.noteContent: Optional[str] = None  # used for star explain
        self.noteType: SimaiNoteType = SimaiNoteType.Tap # Default
        self.slideStartTime: float = 0.0
        self.slideTime: float = 0.0
        self.startPosition: int = 1  # Key position (1-8)
        self.touchArea: str = ' '

class SimaiTimingPoint:
    def __init__(self, _time: float, textposX: int = 0, textposY: int = 0,
                 _content: str = "", bpm: float = 0.0, _hspeed: float = 1.0):
        self.time: float = _time
        self.rawTextPositionX: int = textposX
        self.rawTextPositionY: int = textposY
        self.notesContent: str = _content.replace("\n", "").replace(" ", "")
        self.currentBpm: float = bpm if bpm > 0 else -1.0 # Ensure valid BPM or -1
        self.HSpeed: float = _hspeed
        self.havePlayed: bool = False
        self.noteList: List[SimaiNote] = [] # Cache for getNotes()

    def _is_slide_note(self, noteText: str) -> bool:
        SLIDE_MARKS = "-^v<>Vpqszw"
        return any(mark in noteText for mark in SLIDE_MARKS)

    def _is_touch_note(self, noteText: str) -> bool:
        TOUCH_MARKS = "ABCDE"
        return any(noteText.startswith(mark) for mark in TOUCH_MARKS)

    def _get_time_from_beats(self, noteText: str) -> float:
        """Calculates duration based on beat notation [:] or [#]"""
        total_duration = 0.0
        current_index = 0

        while True:
            start_index = noteText.find('[', current_index)
            if start_index == -1:
                break
            over_index = noteText.find(']', start_index)
            if over_index == -1:
                break # Malformed

            inner_string = noteText[start_index + 1 : over_index]
            current_index = over_index + 1 # Move past this bracket pair

            time_one_beat = 1.0 / (self.currentBpm / 60.0) if self.currentBpm > 0 else 0

            hash_count = inner_string.count('#')

            try:
                if hash_count == 1:
                    parts = inner_string.split('#')
                    # Format: [bpm#duration] or [bpm#beat:count]
                    if ':' in parts[1]:
                        # [bpm#beat:count]
                        inner_string = parts[1] # Process beat:count below
                        time_one_beat = 1.0 / (float(parts[0]) / 60.0) if float(parts[0]) > 0 else 0
                    else:
                        # [bpm#absolute_time] - This is the duration directly
                        total_duration += float(parts[1])
                        continue # Move to next bracket pair if any

                elif hash_count == 2:
                    # Format: [absolute_start#bpm#absolute_duration]
                    parts = inner_string.split('#')
                    total_duration += float(parts[2]) # Duration is the last part
                    continue # Move to next bracket pair if any

                # Format: [beat:count] (or after parsing [bpm#beat:count])
                if ':' in inner_string:
                    numbers = inner_string.split(':')
                    divide = int(numbers[0])
                    count = int(numbers[1])
                    if divide > 0:
                         total_duration += time_one_beat * 4.0 / divide * count
                # Note: Original code didn't explicitly handle cases without ':',
                # assuming malformed or handled by other logic. Added check.

            except (ValueError, IndexError, ZeroDivisionError) as e:
                print(f"Warning: Could not parse beat time '{inner_string}': {e}", file=sys.stderr)
                # Continue calculation if possible, otherwise duration might be inaccurate

        # If no brackets were found, check legacy format (hold without brackets)
        # The C# code implies holdTime=0 if 'h' is last and no brackets
        # This is handled within getSingleNote now.
        # If brackets *were* found, return the calculated total_duration
        if noteText.count('[') > 0:
             return total_duration

        # Fallback if no brackets - should be handled by caller logic (e.g. hold 'h')
        # but return 0 as a safety default if called unexpectedly.
        return 0.0


    def _get_star_wait_time(self, noteText: str) -> float:
        """Calculates slide start delay based on [#] notation"""
        start_index = noteText.find('[')
        over_index = noteText.find(']')
        if start_index == -1 or over_index == -1 or start_index >= over_index:
            # Default wait time: 1 beat if no valid bracket notation
             return 1.0 / (self.currentBpm / 60.0) if self.currentBpm > 0 else 0.0

        inner_string = noteText[start_index + 1 : over_index]
        bpm = self.currentBpm
        wait_time = 0.0
        parsed = False

        try:
            hash_count = inner_string.count('#')
            if hash_count == 1:
                # Format [bpm#...] - use specified bpm for default 1 beat wait
                parts = inner_string.split('#')
                bpm = float(parts[0])
                parsed = True
            elif hash_count == 2:
                # Format [absolute_start#bpm#absolute_duration]
                parts = inner_string.split('#')
                wait_time = float(parts[0]) # Absolute start time is the wait
                return wait_time # Return absolute time directly
            # else: format [beat:count] - use currentBpm for default 1 beat wait

        except (ValueError, IndexError) as e:
             print(f"Warning: Could not parse star wait BPM/time '{inner_string}': {e}", file=sys.stderr)
             bpm = self.currentBpm # Revert to current BPM on error

        # Default wait time is 1 beat (using potentially overridden bpm)
        if bpm > 0:
            return 1.0 / (bpm / 60.0)
        else:
            return 0.0 # Cannot calculate wait time without BPM


    def _get_single_note(self, noteText: str) -> SimaiNote:
        simaiNote = SimaiNote()
        original_note_text = noteText # Keep for modifications

        # Determine Base Type and Position
        if self._is_touch_note(original_note_text):
            simaiNote.touchArea = original_note_text[0]
            try:
                # Position is 1-8 for A,B,D,E; C maps to position 8 internally
                if simaiNote.touchArea != 'C':
                    simaiNote.startPosition = int(original_note_text[1])
                else:
                    simaiNote.startPosition = 8 # Special case for Center
                simaiNote.noteType = SimaiNoteType.Touch
            except (IndexError, ValueError):
                 print(f"Warning: Could not parse touch note position: {original_note_text}", file=sys.stderr)
                 simaiNote.startPosition = 1 # Default fallback
                 simaiNote.noteType = SimaiNoteType.Touch # Assume touch despite error
        else:
            try:
                simaiNote.startPosition = int(original_note_text[0])
                simaiNote.noteType = SimaiNoteType.Tap # Default, might change
            except (IndexError, ValueError):
                 print(f"Warning: Could not parse tap note position: {original_note_text}", file=sys.stderr)
                 simaiNote.startPosition = 1 # Default fallback

        # Modifiers
        if 'f' in original_note_text:
            simaiNote.isHanabi = True

        # Hold
        if 'h' in original_note_text:
            if simaiNote.noteType == SimaiNoteType.Touch:
                simaiNote.noteType = SimaiNoteType.TouchHold
                simaiNote.holdTime = self._get_time_from_beats(original_note_text)
            else:
                simaiNote.noteType = SimaiNoteType.Hold
                # Hold duration requires brackets `[...]` or defaults to 0 if 'h' is last
                if original_note_text.endswith('h') and '[' not in original_note_text:
                     simaiNote.holdTime = 0.0
                else:
                    simaiNote.holdTime = self._get_time_from_beats(original_note_text)

        # Slide (must check *after* hold, as hold takes precedence if 'h' exists)
        # Check if it *wasn't* already determined to be a Hold/TouchHold
        is_basic_hold = simaiNote.noteType in (SimaiNoteType.Hold, SimaiNoteType.TouchHold)
        if not is_basic_hold and self._is_slide_note(original_note_text):
            simaiNote.noteType = SimaiNoteType.Slide
            simaiNote.slideTime = self._get_time_from_beats(original_note_text)
            timeStarWait = self._get_star_wait_time(original_note_text)
            simaiNote.slideStartTime = self.time + timeStarWait # Add wait time offset

            # Handle slide head visibility modifiers ('!' or '?')
            # Need to remove them *after* parsing time, but before storing noteContent
            if '!' in original_note_text:
                simaiNote.isSlideNoHead = True
                noteText = noteText.replace('!', '', 1) # Modify the copy for noteContent
            elif '?' in original_note_text:
                simaiNote.isSlideNoHead = True
                noteText = noteText.replace('?', '', 1) # Modify the copy for noteContent

        # Break ('b') - Complex logic depending on context
        if 'b' in original_note_text:
            is_break_set = False
            is_slide_break_set = False
            if simaiNote.noteType == SimaiNoteType.Slide:
                 # Check context for 'b': break note head vs break slide body
                 b_indices = [i for i, char in enumerate(original_note_text) if char == 'b']
                 for b_index in b_indices:
                    is_potentially_slide_break = False
                    if b_index < len(original_note_text) - 1:
                        # If 'b' is not the last char, check if next char is '['
                        if original_note_text[b_index + 1] == '[':
                           is_potentially_slide_break = True
                    else:
                        # If 'b' is the last char, it's a slide break according to Simai syntax
                        is_potentially_slide_break = True

                    if is_potentially_slide_break:
                         is_slide_break_set = True
                    else:
                         # If not followed by '[' or end of string, assume it's for the head note
                         is_break_set = True
            else:
                # If not a slide, 'b' always means a break note
                is_break_set = True

            simaiNote.isBreak = is_break_set
            simaiNote.isSlideBreak = is_slide_break_set
            noteText = noteText.replace('b', '') # Remove all 'b's for final content

        # EX Note ('x')
        if 'x' in original_note_text:
            simaiNote.isEx = True
            noteText = noteText.replace('x', '') # Remove 'x' for final content

        # Star Head ('$') / Fake Rotate ('$$')
        if '$' in original_note_text:
            simaiNote.isForceStar = True
            if original_note_text.count('$') >= 2:
                simaiNote.isFakeRotate = True
            noteText = noteText.replace('$', '') # Remove '$' for final content

        simaiNote.noteContent = noteText # Store potentially modified text
        return simaiNote

    def _get_same_head_slide(self, content: str) -> List[SimaiNote]:
        """Parses slides sharing the same start note (e.g., 1*v[4:1]*<[4:1])"""
        simaiNotes: List[SimaiNote] = []
        noteContents = content.split('*')
        if not noteContents:
            return simaiNotes

        # First part is the head note
        note1 = self._get_single_note(noteContents[0])
        simaiNotes.append(note1)

        # Subsequent parts are slide segments starting from the same position
        for item in noteContents[1:]:
            if not item: continue # Skip empty parts resulting from consecutive '*'
            # Construct the text for the segment note using the head's start position
            # Need to handle touch notes correctly (e.g., A1*v -> use A1)
            start_prefix = ""
            if self._is_touch_note(noteContents[0]):
                 # Need the area + position (e.g., "A1") if not 'C'
                 if note1.touchArea != 'C':
                     start_prefix = f"{note1.touchArea}{note1.startPosition}"
                 else:
                     start_prefix = note1.touchArea # Just 'C'
            else:
                start_prefix = str(note1.startPosition) # Just the position number

            note2text = start_prefix + item
            note2 = self._get_single_note(note2text)
            note2.isSlideNoHead = True # These segments don't show a head note
            simaiNotes.append(note2)

        return simaiNotes

    def getNotes(self) -> List[SimaiNote]:
        """Parses the notesContent string into a list of SimaiNote objects."""
        if self.noteList: # Return cached list if already parsed
            return self.noteList

        simaiNotes: List[SimaiNote] = []
        if not self.notesContent:
            return simaiNotes

        try:
            # Handle two-digit taps (e.g., "15")
            if len(self.notesContent) == 2 and self.notesContent.isdigit():
                simaiNotes.append(self._get_single_note(self.notesContent[0]))
                simaiNotes.append(self._get_single_note(self.notesContent[1]))
            # Handle simultaneous notes separated by '/'
            elif '/' in self.notesContent:
                notes = self.notesContent.split('/')
                for note in notes:
                    if not note: continue
                    if '*' in note: # Multi-segment slide within simultaneous group
                        simaiNotes.extend(self._get_same_head_slide(note))
                    else:
                        simaiNotes.append(self._get_single_note(note))
            # Handle same-head slides '*'
            elif '*' in self.notesContent:
                simaiNotes.extend(self._get_same_head_slide(self.notesContent))
            # Handle single note
            else:
                simaiNotes.append(self._get_single_note(self.notesContent))

            self.noteList = simaiNotes # Cache the result
            return simaiNotes
        except Exception as e:
            print(f"Error parsing notes content '{self.notesContent}' at time {self.time}: {e}", file=sys.stderr)
            self.noteList = [] # Cache empty list on error
            return self.noteList

# --- Module-level state (equivalent to static class members) ---
title: Optional[str] = ""
artist: Optional[str] = ""
designer: Optional[str] = ""
other_commands: Optional[str] = ""
first: float = 0.0
# Initialize lists with None or empty strings for the fixed size
fumens: List[Optional[str]] = [None] * 7
levels: List[Optional[str]] = [None] * 7
notelist: List[SimaiTimingPoint] = []
timinglist: List[SimaiTimingPoint] = []

# --- Module-level functions (equivalent to static class methods) ---

def clear_data():
    """Resets all the module-level data."""
    global title, artist, designer, other_commands, first, fumens, levels, notelist, timinglist
    title = ""
    artist = ""
    designer = ""
    other_commands = ""
    first = 0.0
    fumens = [None] * 7
    levels = [None] * 7
    notelist = []
    timinglist = []

def _get_value(varline: str) -> str:
    """Helper to extract value after '=' sign."""
    try:
        return varline.split("=", 1)[1]
    except IndexError:
        return "" # Return empty if '=' is not found

def read_data(filename: str) -> bool:
    """
    Reads maidata.txt into module variables.
    Prints error messages instead of showing a MessageBox.
    Returns True on success, False on error.
    """
    global title, artist, designer, other_commands, first, fumens, levels
    line_num = 0
    other_cmds_list = [] # Collect other commands line by line

    clear_data() # Start fresh

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            maidata_txt = f.readlines()

        i = 0
        while i < len(maidata_txt):
            line = maidata_txt[i].strip() # Remove leading/trailing whitespace
            line_num = i + 1

            if not line: # Skip empty lines
                i += 1
                continue

            if line.startswith("&title="):
                title = _get_value(line)
            elif line.startswith("&artist="):
                artist = _get_value(line)
            elif line.startswith("&des="):
                designer = _get_value(line)
            elif line.startswith("&first="):
                try:
                    first = float(_get_value(line))
                except ValueError:
                    print(f"Warning: Invalid float value for '&first=' at line {line_num}: {line}", file=sys.stderr)
                    first = 0.0 # Default value
            elif line.startswith("&lv_") or line.startswith("&inote_"):
                # This section needs careful index handling
                # Find which level/note this line corresponds to
                processed_line = False
                for j in range(1, 8): # Check levels 1 through 7
                    level_prefix = f"&lv_{j}="
                    inote_prefix = f"&inote_{j}="

                    if line.startswith(level_prefix):
                        if j-1 < len(levels):
                            levels[j-1] = _get_value(line)
                        processed_line = True
                        break # Found the level, move to next line

                    elif line.startswith(inote_prefix):
                        if j-1 < len(fumens):
                            the_note_lines = [_get_value(line)] # Start with the value on the current line
                            # Read subsequent lines until another '&' command or EOF
                            i += 1 # Move to the next line
                            while i < len(maidata_txt):
                                next_line = maidata_txt[i] # Don't strip yet, keep original newlines
                                if next_line.strip().startswith("&"):
                                     i -= 1 # Backtrack: this line belongs to the next command
                                     break
                                the_note_lines.append(next_line.rstrip('\r\n')) # Add line, remove trailing newline chars
                                i += 1
                            fumens[j-1] = "\n".join(the_note_lines) # Join with newlines preserved
                        processed_line = True
                        break # Found the inote block, break inner loop (outer loop continues from updated i)
                # If the line started with &lv_ or &inote_ but didn't match 1-7
                if not processed_line:
                     other_cmds_list.append(line) # Treat as other command
            else:
                other_cmds_list.append(line)

            i += 1 # Move to the next line in the file

        other_commands = "\n".join(other_cmds_list).strip()
        return True

    except FileNotFoundError:
        print(f"Error: File not found: {filename}", file=sys.stderr)
        # Mimic C# message box content
        print(f"读取谱面时出现错误: 在maidata.txt第{line_num}行:\n找不到文件 '{filename}'", file=sys.stderr)
        return False
    except Exception as e:
        # Mimic C# message box content
        print(f"读取谱面时出现错误: 在maidata.txt第{line_num}行:\n{e}", file=sys.stderr)
        return False

def save_data(filename: str):
    """Saves the current module data back to maidata.txt."""
    global title, artist, designer, other_commands, first, fumens, levels

    maidata: List[str] = []

    # Ensure required fields have defaults if None
    _title = title if title is not None else ""
    _artist = artist if artist is not None else ""
    _designer = designer if designer is not None else ""
    _other_commands = other_commands if other_commands is not None else ""

    maidata.append(f"&title={_title}")
    maidata.append(f"&artist={_artist}")
    maidata.append(f"&first={first}") # float should format ok
    maidata.append(f"&des={_designer}")

    # Add other commands if they exist
    if _other_commands:
        maidata.append(_other_commands) # Assumes other_commands includes necessary newlines

    # Add levels
    for i, level in enumerate(levels):
        if level is not None and level.strip() != "":
            maidata.append(f"&lv_{i+1}={level.strip()}")

    # Add fumens (inote)
    for i, fumen in enumerate(fumens):
        if fumen is not None and fumen.strip() != "":
            # Ensure the first line starts correctly after '='
            lines = fumen.split('\n', 1)
            first_line = lines[0].strip()
            rest_of_fumen = lines[1] if len(lines) > 1 else ""
            maidata.append(f"&inote_{i+1}={first_line}")
            if rest_of_fumen:
                 maidata.append(rest_of_fumen.strip()) # Add the rest, stripped

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(maidata)) # Write lines separated by newline
            f.write("\n") # Add trailing newline for consistency
    except Exception as e:
        print(f"Error saving data to {filename}: {e}", file=sys.stderr)


def _is_note_char(note_char: str) -> bool:
    """Checks if a character represents a note start (Tap, Touch, Hold, Slide head)."""
    # Includes 0 which might be valid in some simai contexts? Kept from original.
    NOTE_CHARS = "1234567890ABCDE"
    return note_char in NOTE_CHARS


def serialize(text: str, position: int = 0) -> float:
    """
    Parses the fumen (chart) text and populates notelist and timinglist.
    Returns the song time corresponding to the given character position.
    """
    global notelist, timinglist, first # Need to read 'first' offset

    _notelist: List[SimaiTimingPoint] = []
    _timinglist: List[SimaiTimingPoint] = []
    requested_time: float = 0.0 # Time at the requested cursor position

    try:
        bpm: float = 0.0 # Must be initialized before first note
        cur_h_speed: float = 1.0
        time: float = float(first) # Start time from metadata (in seconds)
        beats: int = 4 # Default beats per measure
        have_note: bool = False
        note_temp: str = ""
        y_count: int = 0 # Line number (approx)
        x_count: int = 0 # Character position on line (approx)

        # --- First Pass: Find initial BPM ---
        # Simai requires BPM to be set before the first note time calculation
        initial_bpm_found = False
        temp_i = 0
        while temp_i < len(text):
             if text[temp_i] == '(':
                 bpm_s = ""
                 temp_i += 1
                 while temp_i < len(text) and text[temp_i] != ')':
                     bpm_s += text[temp_i]
                     temp_i += 1
                 try:
                     bpm = float(bpm_s)
                     if bpm <= 0:
                          raise ValueError("BPM must be positive")
                     initial_bpm_found = True
                     break # Found the first BPM
                 except ValueError as e:
                      print(f"Warning: Invalid initial BPM '{bpm_s}': {e}. Chart processing may fail.", file=sys.stderr)
                      # Continue searching in case there's another BPM marker later
                      # If no valid BPM is found before the first note, time calculation will be wrong.
             temp_i +=1

        if not initial_bpm_found:
             print("Warning: No initial BPM '(value)' found before first note/comma. Timing will be incorrect.", file=sys.stderr)
             bpm = 60.0 # Arbitrary fallback BPM to avoid division by zero, but timing *will* be wrong

        # --- Second Pass: Process notes and timing ---
        i = 0
        while i < len(text):
            char = text[i]

            # --- Update cursor position and requested time ---
            # Calculate time *before* processing the character at 'i'
            # requestedTime should reflect the time *up to* the character before 'position'
            if i < position:
                requested_time = time

            # --- Handle Comments ---
            if char == '|' and i + 1 < len(text) and text[i+1] == '|':
                x_count += 1 # Count first '|'
                i += 1 # Move to second '|'
                x_count += 1 # Count second '|'
                i += 1 # Move past second '|'
                # Skip until newline
                while i < len(text) and text[i] != '\n':
                    i += 1
                    x_count += 1
                # If we stopped at \n, the loop increments 'i' later
                # If we stopped at EOF, the loop terminates
                if i < len(text) and text[i] == '\n':
                     y_count += 1
                     x_count = 0
                     i += 1 # Consume the newline
                     continue # Go to start of loop for the next character
                else:
                     break # End of file after comment

            # --- Handle Line Breaks and Position Tracking ---
            if char == '\n':
                y_count += 1
                x_count = 0
            else:
                x_count += 1

            # --- Parse Commands ---
            if char == '(': # BPM Change
                have_note = False
                note_temp = ""
                bpm_s = ""
                i += 1 # Move past '('
                x_count +=1
                start_bracket = i
                while i < len(text) and text[i] != ')':
                    bpm_s += text[i]
                    i += 1
                    x_count += 1
                if i < len(text) and text[i] == ')': # Check if ')' was found
                    try:
                        new_bpm = float(bpm_s)
                        if new_bpm <= 0 : raise ValueError("BPM must be positive")
                        bpm = new_bpm
                    except ValueError:
                         print(f"Warning: Invalid BPM value '({bpm_s})' at line ~{y_count+1}. Using previous BPM {bpm}.", file=sys.stderr)
                    # i is already at ')' here
                else:
                     print(f"Warning: Unmatched '(' for BPM at line ~{y_count+1}. BPM not changed.", file=sys.stderr)
                     i = start_bracket -1 # Backtrack to re-process content if no ')' found

            elif char == '{': # Beat Signature Change
                have_note = False
                note_temp = ""
                beats_s = ""
                i += 1 # Move past '{'
                x_count += 1
                start_bracket = i
                while i < len(text) and text[i] != '}':
                    beats_s += text[i]
                    i += 1
                    x_count += 1
                if i < len(text) and text[i] == '}':
                     try:
                        new_beats = int(beats_s)
                        if new_beats <= 0 : raise ValueError("Beats must be positive")
                        beats = new_beats
                     except ValueError:
                        print(f"Warning: Invalid beats value '{{{beats_s}}}' at line ~{y_count+1}. Using previous beats {beats}.", file=sys.stderr)
                else:
                    print(f"Warning: Unmatched '{{' for beats at line ~{y_count+1}. Beats not changed.", file=sys.stderr)
                    i = start_bracket -1 # Backtrack

            elif char == 'H': # Hi-Speed Change (HS*)
                 # Check for HS* syntax: H followed by S then *
                 if i + 2 < len(text) and text[i+1] == 'S' and text[i+2] == '*':
                    have_note = False
                    note_temp = ""
                    hs_s = ""
                    i += 3 # Move past 'HS*'
                    x_count += 3
                    start_marker = i
                    while i < len(text) and text[i] != '>':
                         hs_s += text[i]
                         i += 1
                         x_count += 1
                    if i < len(text) and text[i] == '>':
                         try:
                              cur_h_speed = float(hs_s)
                         except ValueError:
                              print(f"Warning: Invalid HS value 'HS*{hs_s}>' at line ~{y_count+1}. Using previous HS {cur_h_speed}.", file=sys.stderr)
                    else:
                         print(f"Warning: Unmatched 'HS*...>' for Hi-Speed at line ~{y_count+1}. HS not changed.", file=sys.stderr)
                         i = start_marker -1 # Backtrack
                 # If not HS*, treat 'H' potentially as part of a note later

            # --- Accumulate Note Data ---
            # Check if it's a note character *after* checking for commands
            elif _is_note_char(char):
                have_note = True
                note_temp += char
            # Accumulate other note parts if we've started a note
            elif have_note and char != ',':
                note_temp += char

            # --- Process Timing Point (Comma) ---
            if char == ',':
                if have_note:
                    note_temp = note_temp.strip() # Clean up whitespace
                    if '`' in note_temp: # Handle fake doubles (伪双)
                        fake_each_list = note_temp.split('`')
                        fake_time = time
                        # 128th note interval: (60/bpm) * (4/128) = (60/bpm) / 32 = 1.875 / bpm
                        time_interval = 1.875 / bpm if bpm > 0 else 0
                        for fake_each_group in fake_each_list:
                            if not fake_each_group: continue # Skip empty strings from split
                            # print(f"Fake Note: {fake_each_group} at time {fake_time:.3f}") # Debugging
                            _notelist.append(SimaiTimingPoint(fake_time, x_count, y_count, fake_each_group, bpm, cur_h_speed))
                            fake_time += time_interval
                    else:
                        # print(f"Note: {note_temp} at time {time:.3f}") # Debugging
                        _notelist.append(SimaiTimingPoint(time, x_count, y_count, note_temp, bpm, cur_h_speed))

                    note_temp = "" # Reset for next note segment
                    have_note = False

                # Add timing point regardless of whether a note existed
                _timinglist.append(SimaiTimingPoint(time, x_count, y_count, "", bpm))

                # Advance time for the next segment
                if bpm > 0 and beats > 0:
                    time += (60.0 / bpm) * (4.0 / beats)
                else:
                    # Cannot advance time without valid BPM/beats, print warning once?
                    if bpm <=0 : print(f"Warning: Cannot advance time at line ~{y_count+1}, BPM is {bpm}", file=sys.stderr)
                    if beats <=0 : print(f"Warning: Cannot advance time at line ~{y_count+1}, Beats is {beats}", file=sys.stderr)
                    # Time will remain stuck here

            # Move to the next character
            i += 1

        # --- Final Update ---
        # After loop, if i is exactly position, update requested_time one last time
        if i == position:
             requested_time = time

        # Assign the processed lists to the module variables
        notelist = _notelist
        timinglist = _timinglist

        return requested_time

    except Exception as e:
        print(f"Error during serialization at char index ~{i}, line ~{y_count+1}: {e}", file=sys.stderr)
        # Clear lists on error to indicate failure
        notelist = []
        timinglist = []
        return 0.0 # Return 0 time on error

def clear_note_list_played_state():
    """Resets the 'havePlayed' flag for all notes in the notelist."""
    global notelist
    # Sort by time first to ensure chronological order if needed elsewhere
    notelist.sort(key=lambda p: p.time)
    for point in notelist:
        point.havePlayed = False

def get_difficulty_text(index: int) -> str:
    """Returns the string name for a difficulty index (0-6)."""
    difficulty_map = {
        0: "EASY",
        1: "BASIC",
        2: "ADVANCED",
        3: "EXPERT",
        4: "MASTER",
        5: "Re_MASTER",
        6: "ORIGINAL" # Or maybe UTAGE? Check MaiMai terminology
    }
    return difficulty_map.get(index, "DEFAULT") # Return "DEFAULT" if index is out of range

# Example Usage (can be removed or placed under if __name__ == "__main__":)
if __name__ == "__main__":
    # Create a dummy maidata.txt for testing
    dummy_maidata_content = """&title=Test Song
&artist=Tester
&des=Chart Designer
&first=1.5
&lv_1=3
&inote_1=(120)E,1,2,3,4,
{8}5,6,7,8,
(180)1/5,2/6,3/7,4/8,
&lv_4=11
&inote_4=(180)1h[4:1],5h[4:1],
1-v[4:1],5-v[4:1],
1b,5x,
C1,
E4,
A1-v[180#4:1],
B8-v[2:1.5],
D3<>pq[1:1][1:1],
1*>[4:1]*<[4:1],
8`4,
HS*2.0>E,
|| This is a comment
1,
"""
    test_filename = "maidata_test.txt"
    with open(test_filename, "w", encoding="utf-8") as f:
        f.write(dummy_maidata_content)

    print(f"--- Testing ReadData ---")
    if read_data(test_filename):
        print(f"Title: {title}")
        print(f"Artist: {artist}")
        print(f"Designer: {designer}")
        print(f"First Offset: {first}")
        print(f"Levels: {levels}")
        # print(f"Fumens: {fumens}") # Can be very long
        print(f"Other Commands:\n{other_commands}")
        print("-" * 10)

        # Test Serialize on the first fumen
        if fumens[0]: # Basic chart
             print(f"--- Testing Serialize (Basic) ---")
             chart_text_basic = fumens[0]
             cursor_pos = 25 # Somewhere in the middle
             time_at_cursor = serialize(chart_text_basic, cursor_pos)
             print(f"Chart Text (Basic):\n{chart_text_basic}")
             print(f"Time at cursor position {cursor_pos}: {time_at_cursor:.3f}s")
             print(f"Total Note Timing Points: {len(notelist)}")
             print(f"Total Timing Points (,): {len(timinglist)}")

             # Test getNotes for the first few points
             for idx, tp in enumerate(notelist[:5]):
                  notes_in_tp = tp.getNotes()
                  print(f" Notes at time {tp.time:.3f} ({tp.notesContent}): {len(notes_in_tp)}")
                  for note in notes_in_tp:
                       print(f"  - Type: {note.noteType.name}, Pos: {note.touchArea}{note.startPosition}, Content: {note.noteContent}")
             print("-" * 10)


        if fumens[3]: # Expert chart
             print(f"--- Testing Serialize (Expert) ---")
             chart_text_expert = fumens[3]
             time_at_cursor = serialize(chart_text_expert) # Test without cursor pos
             print(f"Chart Text (Expert):\n{chart_text_expert}")
             print(f"Time at end: {time_at_cursor:.3f}s") # Should be time of last char processed
             print(f"Total Note Timing Points: {len(notelist)}")
             print(f"Total Timing Points (,): {len(timinglist)}")

             # Test getNotes
             for idx, tp in enumerate(notelist):
                  notes_in_tp = tp.getNotes()
                  print(f" Notes at time {tp.time:.3f} ({tp.notesContent}): {len(notes_in_tp)}")
                  for note in notes_in_tp:
                       print(f"  - Type: {note.noteType.name}, Pos: {note.touchArea}{note.startPosition}, Brk: {note.isBreak}, SB: {note.isSlideBreak}, Hold: {note.holdTime:.3f}, Slide: {note.slideTime:.3f}, Content: {note.noteContent}")

             # Test GetDifficultyText
             print(f"--- Testing GetDifficultyText ---")
             print(f"Index 0: {get_difficulty_text(0)}")
             print(f"Index 3: {get_difficulty_text(3)}")
             print(f"Index 5: {get_difficulty_text(5)}")
             print(f"Index 99: {get_difficulty_text(99)}")

             # Test SaveData
             # print(f"--- Testing SaveData ---")
             # save_data("maidata_saved.txt")
             # print("Data saved to maidata_saved.txt")

    # Clean up the dummy file
    try:
        os.remove(test_filename)
        # os.remove("maidata_saved.txt") # Uncomment if testing save
    except OSError:
        pass