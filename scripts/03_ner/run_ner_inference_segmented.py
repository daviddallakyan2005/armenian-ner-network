import os
import json
import sys
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tokenizer import Tokenizer as ArmTokenizer
import logging


SCRIPT_DIR_INFERENCE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_INFERENCE = os.path.dirname(os.path.dirname(SCRIPT_DIR_INFERENCE))
ARMTOKENIZER_PATH = os.path.join(PROJECT_ROOT_INFERENCE, "external_repos", "ArmTokenizer")
sys.path.append(ARMTOKENIZER_PATH)

# --- Diagnostics ---
print(f"--- Running: {__file__} ---")
print(f"Python Executable: {sys.executable}")
print(f"CUDA Available Check: {torch.cuda.is_available()}")
print(f"PyTorch CUDA Built Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes script is in <root>/scripts/03_ner
    return os.path.dirname(os.path.dirname(script_dir))


PROJECT_ROOT = get_project_root()
SEGMENTED_TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "extracted_entities")
os.makedirs(RESULTS_DIR, exist_ok=True)


MODEL_CHECKPOINT = "run_16-lr_2e-05-acc_1-wd_0.01-bs_8-ep_7"
HARDCODED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", MODEL_CHECKPOINT)

def aggregate_predictions(predictions, word_ids, id_to_label):
    word_predictions = {}
    current_word_id = None
    current_entity_parts = []

    for pred, word_id in zip(predictions, word_ids):
        if word_id is None: # Special token
            continue

        label = id_to_label[pred]

        if word_id != current_word_id: # New word starts
            # Finalize previous word if any
            if current_word_id is not None and current_entity_parts:
                 # Simple aggregation: Take the first label part (ignoring B-/I- for now)
                 # More sophisticated logic needed for B-I-O tagging
                 first_label = current_entity_parts[0].split('-')[-1] # Get 'PER', 'LOC' etc.
                 if first_label != 'O': # Only store if not Outside entity
                     word_predictions[current_word_id] = {'label': first_label} # Store word_id and label

            current_word_id = word_id
            current_entity_parts = []
            if label != 'O':
                 current_entity_parts.append(label)
        else: # Continuation of the current word
            if label != 'O':
                current_entity_parts.append(label)

    # Handle the last word
    if current_word_id is not None and current_entity_parts:
        first_label = current_entity_parts[0].split('-')[-1]
        if first_label != 'O':
            word_predictions[current_word_id] = {'label': first_label}

    return word_predictions # Returns dict {word_id: {'label': 'PER'}}

def run_segmented_inference(model_dir):
    """
    Loads a fine-tuned NER model and runs inference on segmented text files,
    using ArmTokenizer for pre-tokenization.
    """
    logging.info(f"Loading resources from {model_dir}")
    if not os.path.isdir(model_dir):
        logging.error(f"Model directory not found: {model_dir}")
        return

    # Tokenization Check: Keep this block for verification if needed
    try:
        print(f"--- Loading tokenizer for check from: {model_dir} ---")
        inference_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        sample_text = "Որպես սկիզբԵրկերիս ժողովածուն — Արփիկին"
        print(f"--- Tokenizing sample in run_ner_inference_segmented.py ({model_dir}) ---")
        tokens_inference = inference_tokenizer.tokenize(sample_text)
        print(tokens_inference)
        print("----------------------------------------------------------------------")
    except Exception as e:
        print(f"Error loading tokenizer for check: {e}")

    try:
        logging.info("Loading ArmTokenizer...")
        arm_tokenizer = ArmTokenizer()
        logging.info("ArmTokenizer loaded.")

        logging.info("Loading fine-tuned XLM-R model and tokenizer...")
        xlm_roberta_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        logging.info("Fine-tuned model and tokenizer loaded.")
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)
        model.to(device) # Move model to appropriate device
        logging.info(f"Using device: {device_type.upper()}")

        id_to_label = model.config.id2label

    except Exception as e:
        logging.error(f"Failed to load model or tokenizers: {e}", exc_info=True)
        return

    processed_files_count = 0
    for file_name in os.listdir(SEGMENTED_TEXT_DIR):
        if not file_name.endswith("_segmented.json"):
            continue

        segmented_file_path = os.path.join(SEGMENTED_TEXT_DIR, file_name)
        output_file_name = file_name.replace("_segmented.json", "_ner_segmented.json")
        output_file_path = os.path.join(RESULTS_DIR, output_file_name)

        logging.info(f"Processing {file_name}...")

        try:
            # Load the segmented file (list of strings)
            with open(segmented_file_path, 'r', encoding='utf-8') as file:
                segments = json.load(file)

            if not isinstance(segments, list):
                logging.warning(f"Skipping {file_name}: Expected a list of segments, found {type(segments)}.")
                continue

            if not segments:
                logging.warning(f"Skipping {file_name}: Contains no segments.")
                continue

            segmented_results = {}
            total_segments = len(segments)
            logging.info(f"Found {total_segments} segments in {file_name}.")

            for i, segment_text in enumerate(segments):
                segment_id = f"segment_{i}"
                if not isinstance(segment_text, str) or not segment_text.strip():
                    logging.debug(f"Skipping empty or non-string segment {i} in {file_name}.")
                    segmented_results[segment_id] = []
                    continue

                try:
                    tokenizer_obj = arm_tokenizer.tokenize(segment_text)
                    raw_words = tokenizer_obj.tokens()
                    logging.debug(f"Segment {i}: ArmTokenizer output type: {type(raw_words)}, content preview: {raw_words[:10] if isinstance(raw_words, list) else raw_words}")

                    if not isinstance(raw_words, list):
                        logging.error(f"Segment {i}: ArmTokenizer did not return a list. Got type {type(raw_words)}. Content: {raw_words}")
                        segmented_results[segment_id] = [{"errpr": f"ArmTokenizer returned unexpected type: {type(raw_words)}"}]
                        continue
                
                    words = [word for word in raw_words if isinstance(word, str) and word.strip()]

                    if not words:
                         logging.debug(f"Segmnet {i}: No valid words found after filtering ArmTokenizer output.")
                         segmented_results[segment_id] = []
                         continue

                    tokenized_inputs = xlm_roberta_tokenizer(
                        words,
                        is_split_into_words=True,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length", # Or True, depending on preference
                        max_length=512 # Use model's max length or a reasonable default
                    ).to(device) # Move inputs to the same device as model

                    # Store word_ids before model inference, as inputs might be altered
                    word_ids = tokenized_inputs.word_ids()

                    with torch.no_grad():
                        outputs = model(**tokenized_inputs)
                        logits = outputs.logits

                    predictions = torch.argmax(logits, dim=2)
                    predictions_cpu = predictions.squeeze().tolist()

                    cleaned_predictions = []
                    cleaned_word_ids = []
                    for pred, word_id, input_id in zip(predictions_cpu, word_ids, tokenized_inputs['input_ids'].squeeze().tolist()):
                        # Ignore predictions for special tokens (like PAD, CLS, SEP) which often have word_id=None
                        # Also ignore explicit padding tokens if padding='max_length' was used
                        if word_id is not None and input_id != xlm_roberta_tokenizer.pad_token_id:
                             cleaned_predictions.append(pred)
                             cleaned_word_ids.append(word_id)

                    word_labels = ['O'] * len(words)
                    word_scores = [0.0] * len(words)
                    logits_cpu = logits.cpu().numpy()

                    current_word_idx = -1
                    for i, (pred, word_id, input_id) in enumerate(zip(predictions_cpu, word_ids, tokenized_inputs['input_ids'].squeeze().tolist())):
                        if word_id is not None and input_id != xlm_roberta_tokenizer.pad_token_id:
                            if word_id != current_word_idx: # Start of a new word
                                current_word_idx = word_id
                                # Assign the label of the first subtoken to the word
                                word_labels[current_word_idx] = id_to_label.get(pred, 'O')
                                try:
                                    probabilities = torch.softmax(torch.tensor(logits_cpu[0, i]), dim=-1)
                                    word_scores[current_word_idx] = probabilities[pred].item()
                                except IndexError:
                                    word_scores[current_word_idx] = 0.0 # Handle potential index errors
                                    logging.warning(f"IndexError accessing logits at index {i} for word_id {current_word_idx}")

                    final_entities = []
                    current_entity_words = []
                    current_entity_scores = []

                    for idx, word in enumerate(words):
                        label = word_labels[idx]
                        score = word_scores[idx]

                        if label.startswith('B-PER'):
                            if current_entity_words:
                                entity_name = " ".join(current_entity_words)
                                avg_score = np.mean(current_entity_scores) if current_entity_scores else 0.0
                                final_entities.append({"word": entity_name, "entity_group": "PER", "score": float(avg_score)})
                            # Start new entity
                            current_entity_words = [word]
                            current_entity_scores = [score]
                        elif label.startswith('I-PER'):
                            if current_entity_words:
                                current_entity_words.append(word)
                                current_entity_scores.append(score)
                        else: # label is 'O' or another type (e.g., B-LOC)
                            # If currently accumulating a PER entity, it ends here
                            if current_entity_words:
                                entity_name = " ".join(current_entity_words)
                                avg_score = np.mean(current_entity_scores) if current_entity_scores else 0.0
                                final_entities.append({"word": entity_name, "entity_group": "PER", "score": float(avg_score)})
                            # Reset for next potential entity
                            current_entity_words = []
                            current_entity_scores = []

                    # Check if the last segment ended with an entity
                    if current_entity_words:
                         entity_name = " ".join(current_entity_words)
                         avg_score = np.mean(current_entity_scores) if current_entity_scores else 0.0
                         final_entities.append({"word": entity_name, "entity_group": "PER", "score": float(avg_score)})

                    segmented_results[segment_id] = final_entities

                    if i % 100 == 0 or i == total_segments - 1:
                        logging.info(f"  Processed segment {i+1}/{total_segments}")

                except Exception as e:
                    logging.error(f"Error processing segment {i} in {file_name}: {e}", exc_info=True)
                    segmented_results[segment_id] = [{"error": str(e)}]

            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                json.dump(segmented_results, out_file, ensure_ascii=False, indent=2)

            processed_files_count += 1
            logging.info(f"Saved segmented NER results to {output_file_path}")

        except json.JSONDecodeError:
            logging.error(f"Error: Could not parse {file_name} as JSON. Skipping.")
        except Exception as e:
            logging.error(f"General error processing {file_name}: {e}")

    logging.info(f"--- Finished processing all files ---")
    logging.info(f"Processed {processed_files_count} segmented files.")
    logging.info(f"Segmented NER results saved in: {RESULTS_DIR}")


print(f"Starting segmented NER inference using model from: {HARDCODED_MODEL_DIR}")
run_segmented_inference(HARDCODED_MODEL_DIR)
print("--- Segmented NER Inference Script Finished ---") 