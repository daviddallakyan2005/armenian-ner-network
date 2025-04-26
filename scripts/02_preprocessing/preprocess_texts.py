import os
import re
import json
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Calculate Project Root dynamically ---
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume the project root is two levels up from the script's directory (scripts/02_preprocessing -> project root)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# --- Define paths relative to Project Root ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Regex patterns
# Chapter/Section markers
# - Lines like "ԳԼՈՒԽ ԱՌԱՋԻՆ", "ԳԼՈՒԽ 1", etc. (Allow mixed case)
CHAPTER_MARKER_1 = re.compile(r"^\s*(?:ԳԼՈՒԽ|Գլուխ)\s+[Ա-Ֆա-ֆ\dIVXLCDM]+\s*$", re.IGNORECASE) # Added |Գլուխ and ա-ֆ, keep IGNORECASE for Roman
# - Lines like "ՄԱՍՆ ԱՌԱՋԻՆ" (Allow mixed case)
CHAPTER_MARKER_2 = re.compile(r"^\s*(?:ՄԱՍՆ|Մասն)\s+[Ա-Ֆա-ֆ]+\s*$") # Added |Մասն and ա-ֆ
# - Single Armenian letter on its own line (potential section marker, allow lower)
CHAPTER_MARKER_3 = re.compile(r"^\s*[Ա-Ֆա-ֆ]\s*$") # Added ա-ֆ
# - All-caps titles (potentially indicating sections/poems) - Check if preceded/followed by blank lines
#   This one is broad, apply carefully. We check context later.
CHAPTER_MARKER_4 = re.compile(r"^\s*[Ա-Ֆ]{2,}[Ա-Ֆ\s]+\s*$") # At least two letters, all caps Armenian + space - *KEEP UPPERCASE ONLY*

ALL_CHAPTER_MARKERS = [
    CHAPTER_MARKER_1,
    CHAPTER_MARKER_2,
    CHAPTER_MARKER_3,
    CHAPTER_MARKER_4,
]

# Lines to remove
DASH_LINE = re.compile(r"^\s*-{3,}\s*$") # Line with 3 or more dashes
YEAR_LINE = re.compile(r"^\s*\d{4}\s*$") # Line with only a 4-digit number

# Regex to identify base name and number suffix (e.g., _1, _2)
FILENAME_PATTERN = re.compile(r"^(.*?)(?:_(\d+))?\.txt$")

def group_and_read_files(directory):
    """Groups files by work, reads, sorts numbered parts, and concatenates."""
    files_by_work = defaultdict(list)
    all_files = [f for f in os.listdir(directory) if f.endswith('.txt') and not f.startswith('.')]

    # Group files
    for filename in all_files:
        match = FILENAME_PATTERN.match(filename)
        if match:
            base_name = match.group(1)
            number_str = match.group(2)
            number = int(number_str) if number_str else -1 # -1 for non-numbered files
            files_by_work[base_name].append((number, filename))
        else:
            logging.warning(f"Could not parse filename: {filename}")

    # Read and concatenate
    concatenated_texts = {}
    for base_name, file_list in files_by_work.items():
        # Check if numbered parts exist
        numbered_files = sorted([(num, name) for num, name in file_list if num != -1])
        non_numbered_file = next((name for num, name in file_list if num == -1), None)

        text_content = ""
        files_to_read = []

        if numbered_files:
            # Prioritize numbered files if they exist
            files_to_read = [name for num, name in numbered_files]
            if non_numbered_file:
                 logging.info(f"Work '{base_name}': Found numbered parts, ignoring '{non_numbered_file}'.")
        elif non_numbered_file:
            # Use the non-numbered file if no numbered parts exist
            files_to_read = [non_numbered_file]
        else:
            # Should not happen if parsing worked, but handle just in case
             logging.warning(f"Work '{base_name}': No usable files found.")
             continue

        logging.info(f"Work '{base_name}': Reading and concatenating {files_to_read}")
        for filename in files_to_read:
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Add newline separator between concatenated files
                    if text_content:
                        text_content += "\n\n"
                    text_content += f.read()
            except Exception as e:
                logging.error(f"Error reading file {filepath}: {e}")
                # Decide if we should skip this file or the whole work
                continue # Skip this file

        if text_content:
            concatenated_texts[base_name] = text_content
        else:
            logging.warning(f"Work '{base_name}': No content read.")


    return concatenated_texts

def clean_text(text):
    """Removes irrelevant lines from the text."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if DASH_LINE.match(line) or YEAR_LINE.match(line):
            continue # Skip these lines
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def segment_text_option_b(text):
    """Segments text into chapters/sections if markers found, else paragraphs."""
    lines = text.split('\n')
    units = []
    current_unit_lines = []
    markers_found = False
    potential_marker_indices = []

    # First pass: check if any chapter markers exist and their indices
    for i, line in enumerate(lines):
        is_potential_marker = False
        for marker_re in ALL_CHAPTER_MARKERS:
            if marker_re.match(line):
                 # Specific check for potentially broad ALL_CAPS marker
                 if marker_re == CHAPTER_MARKER_4:
                     # Check context: require blank lines before and after (or start/end of doc)
                     prev_line_blank = (i == 0) or (lines[i-1].strip() == "")
                     next_line_blank = (i == len(lines) - 1) or (lines[i+1].strip() == "")
                     if prev_line_blank and next_line_blank:
                         is_potential_marker = True
                 else:
                     is_potential_marker = True
                 break # Found a marker for this line
        if is_potential_marker:
            markers_found = True
            potential_marker_indices.append(i)

    if markers_found:
        logging.info("Chapter/Section markers found. Segmenting by markers.")
        start_index = 0
        for marker_index in potential_marker_indices:
             # Add the segment before the marker
             segment = '\n'.join(lines[start_index:marker_index]).strip()
             if segment: # Avoid adding empty segments
                 units.append(segment)
             # The marker line itself is usually not part of the content
             start_index = marker_index + 1

        # Add the last segment after the final marker
        last_segment = '\n'.join(lines[start_index:]).strip()
        if last_segment:
            units.append(last_segment)

        # Further refine units - remove any potentially captured marker lines if needed
        # (The current logic excludes the marker line by using marker_index+1, which is likely correct)

    else:
        logging.info("No chapter/section markers found. Segmenting by paragraphs.")
        # Split by one or more blank lines
        paragraphs = re.split(r'\n\s*\n+', text)
        units = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs

    return units

def segment_text_option_a(text):
    """Segments text strictly by paragraphs based on blank lines."""
    logging.info("Segmenting strictly by paragraphs (Option A).")
    # Split by one or more blank lines
    paragraphs = re.split(r'\n\s*\n+', text)
    units = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs
    return units

def segment_text_option_c(text):
    """Segments text by major markers (ՄԱՍՆ..., ALL CAPS title) then paragraphs within."""
    major_markers = [CHAPTER_MARKER_2, CHAPTER_MARKER_4] # Define which are 'major'
    lines = text.split('\n')
    units = []
    major_markers_found = False
    potential_major_marker_indices = []

    # Find indices of major markers
    for i, line in enumerate(lines):
        is_potential_major_marker = False
        for marker_re in major_markers:
            if marker_re.match(line):
                 if marker_re == CHAPTER_MARKER_4:
                     # Context check for ALL CAPS
                     prev_line_blank = (i == 0) or (lines[i-1].strip() == "")
                     next_line_blank = (i == len(lines) - 1) or (lines[i+1].strip() == "")
                     if prev_line_blank and next_line_blank:
                         is_potential_major_marker = True
                 else:
                     is_potential_major_marker = True
                 break
        if is_potential_major_marker:
            major_markers_found = True
            potential_major_marker_indices.append(i)

    if major_markers_found:
        logging.info("Major Section markers found. Segmenting by major markers, then paragraphs (Option C).")
        major_segments_text = []
        start_index = 0
        for marker_index in potential_major_marker_indices:
             segment_text_before_marker = '\n'.join(lines[start_index:marker_index]).strip()
             if segment_text_before_marker:
                 major_segments_text.append(segment_text_before_marker)
             # Exclude marker line itself
             start_index = marker_index + 1

        # Add the last major segment
        last_major_segment_text = '\n'.join(lines[start_index:]).strip()
        if last_major_segment_text:
            major_segments_text.append(last_major_segment_text)

        # Now, segment each major segment by paragraphs
        final_units = []
        for major_segment in major_segments_text:
            paragraphs = re.split(r'\n\s*\n+', major_segment)
            final_units.extend([p.strip() for p in paragraphs if p.strip()])
        units = final_units

    else:
        logging.info("No major markers found for Option C. Falling back to paragraph segmentation.")
        # Fallback to paragraph segmentation (same as Option A)
        paragraphs = re.split(r'\n\s*\n+', text)
        units = [p.strip() for p in paragraphs if p.strip()]

    return units

def main():
    logging.info(f"Starting preprocessing from {RAW_DATA_DIR}")
    texts_by_work = group_and_read_files(RAW_DATA_DIR)

    for work_name, full_text in texts_by_work.items():
        logging.info(f"Processing work: {work_name}")
        # 1. Clean text
        cleaned_text = clean_text(full_text)
        if not cleaned_text.strip():
            logging.warning(f"Work '{work_name}' is empty after cleaning.")
            continue

        # 2. Segment text
        segmented_units = segment_text_option_b(cleaned_text)
        if not segmented_units:
             logging.warning(f"Work '{work_name}': No segments found after processing.")
             continue

        # Save the segments for this specific work to its own file
        output_path = os.path.join(PROCESSED_DATA_DIR, f"{work_name}_segmented.json")
        logging.info(f"Work '{work_name}': Found {len(segmented_units)} segments/paragraphs. Saving to {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Save only the list of segments for this work
                json.dump(segmented_units, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.error(f"Error writing processed data for {work_name} to {output_path}: {e}")

if __name__ == "__main__":
    main() 