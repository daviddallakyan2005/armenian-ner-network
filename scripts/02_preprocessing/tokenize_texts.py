import os
import sys
import json
from tqdm import tqdm

# Adjust the path to include the ArmTokenizer directory
# This assumes the script is run from the root of the Armenian_ML project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_repos/ArmTokenizer')))

try:
    from tokenizer import Tokenizer
except ImportError as e:
    print("Error importing Tokenizer. Make sure ArmTokenizer is in the expected path.")
    print(f"Current sys.path: {sys.path}")
    print(e)
    sys.exit(1)

def tokenize_armenian_text(input_dir, output_dir):
    """
    Tokenizes segmented Armenian text files using ArmTokenizer.

    Args:
        input_dir (str): Directory containing segmented JSON files.
        output_dir (str): Directory where tokenized JSON files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    tokenizer_instance = Tokenizer() # Instantiate the tokenizer

    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('_segmented.json')]
    print(f"Found {len(files_to_process)} segmented files in {input_dir}")

    for filename in tqdm(files_to_process, desc="Tokenizing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('_segmented.json', '_tokenized.json'))

        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)

            tokenized_data = []
            # Assuming the input JSON is a list of strings (segments)
            # Adjust if the structure is different (e.g., list of dicts with a 'text' key)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                 # Simple list of strings format
                for segment_text in tqdm(data, desc=f"Tokenizing {filename}", leave=False):
                    if not segment_text or not isinstance(segment_text, str):
                        print(f"Skipping empty or invalid segment in {filename}")
                        tokenized_data.append({'original_segment': segment_text, 'tokens': []})
                        continue

                    # The tokenizer internally segments and then tokenizes.
                    # We pass the whole segment text to the tokenize method.
                    tokenizer_instance.tokenize(segment_text)
                    # Extract tokens for this specific segment
                    # The tokenizer instance stores results for the last call.
                    # We need the tokens only for the current segment.
                    segment_tokens = tokenizer_instance.tokens() # Gets all tokens from the last .tokenize() call

                    tokenized_data.append({
                        'original_segment': segment_text,
                        'tokens': segment_tokens
                    })

            # --- Add handling for other potential JSON structures if needed ---
            # Example: if data is a list of dictionaries [{'id': 1, 'text': '...'}, ...]
            # elif isinstance(data, list) and all(isinstance(item, dict) and 'text' in item for item in data):
            #     for segment_obj in tqdm(data, desc=f"Tokenizing {filename}", leave=False):
            #         segment_text = segment_obj.get('text', '')
            #         if not segment_text:
            #             print(f"Skipping empty segment in {filename} (id: {segment_obj.get('id', 'N/A')})")
            #             segment_obj['tokens'] = []
            #             tokenized_data.append(segment_obj)
            #             continue
            #         tokenizer_instance.tokenize(segment_text)
            #         segment_obj['tokens'] = tokenizer_instance.tokens()
            #         tokenized_data.append(segment_obj)
            # ------------------------------------------------------------------

            else:
                 print(f"Warning: Unexpected JSON structure in {filename}. Expected a list of strings.")
                 # Attempt basic tokenization if it's just a single string
                 if isinstance(data, str):
                     tokenizer_instance.tokenize(data)
                     tokenized_data = [{'original_text': data, 'tokens': tokenizer_instance.tokens()}]
                 else: # Skip file if structure is unknown
                    print(f"Skipping file {filename} due to unknown structure.")
                    continue


            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(tokenized_data, f_out, ensure_ascii=False, indent=4)

        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {input_path}")
        except Exception as e:
            print(f"An error occurred processing file {input_path}: {e}")

    print(f"Tokenization complete. Output saved to {output_dir}")

if __name__ == "__main__":
    # Define relative paths based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..')) # Navigate up two levels

    input_directory = os.path.join(project_root, 'data', 'processed')
    output_directory = os.path.join(project_root, 'data', 'tokenized')

    print(f"Project Root: {project_root}")
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")

    # Check if input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found at {input_directory}")
        print("Please ensure the segmented data exists in 'data/processed'.")
        sys.exit(1)

    tokenize_armenian_text(input_directory, output_directory) 