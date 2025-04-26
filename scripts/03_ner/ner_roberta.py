import os
import json
import torch
import numpy as np
from datasets import load_dataset, ClassLabel, Sequence, Features
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from itertools import product
from seqeval.metrics import precision_score, recall_score, f1_score

API_KEY = "salad_cloud_user_w5S7t3k35PYpKn4XRn4uUwcphg7bNCrXGUxaK3IEBSY02PMmS"
API_URL = "https://api.salad.com/api/public/organizations/armenian-ml/inference-endpoints/transcription-lite/jobs"

# --- Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Calculate the project root directory (assuming script is in <root>/scripts/03_ner)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_CHECKPOINT = "xlm-roberta-base"
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
gradient_accumulation_steps = 4
MODEL_NAME = f"{MODEL_CHECKPOINT}-finetuned-pioner-ner"

# Define paths relative to the project root
MODELS_BASE_DIR = os.path.join(PROJECT_ROOT, "models") # Base directory for all models
PIONER_SILVER_DIR = os.path.join(PROJECT_ROOT, "external_repos", "pioner", "pioner-silver")
PIONER_GOLD_DIR = os.path.join(PROJECT_ROOT, "external_repos", "pioner", "pioner-gold")
TOKENIZED_TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "tokenized")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_BASE_DIR, exist_ok=True)

# Hyperparameters for tuning
learning_rates = [1e-5, 2e-5, 3e-5]
grad_accum_steps = [1, 2, 4]
weight_decays = [0.0, 0.01]
batch_sizes = [8]
epochs_list = [3, 7]

# Generate all combinations
hyperparam_combinations = list(product(learning_rates, grad_accum_steps, weight_decays, batch_sizes, epochs_list))

# --- 1. Load pioNER Dataset ---
print("Loading pioNER dataset...")

data_files = {
    "train": os.path.join(PIONER_SILVER_DIR, "train.conll03"),
    "validation": os.path.join(PIONER_SILVER_DIR, "dev.conll03"),
    "test": os.path.join(PIONER_GOLD_DIR, "test.conll03"),
}

# Define features using the Features object
# features = Features({
#     'tokens': Sequence(feature='string'),
#     'ner_tags': Sequence(feature=ner_tags)
# })

# Load using the 'conll2003' script
raw_datasets = load_dataset(
    'conll2003',
    data_files=data_files,
    # features=features, # Removed features argument
    trust_remote_code=True # Added to allow execution of dataset script code
)

# --- Get labels after loading ---
label_list = raw_datasets["train"].features["ner_tags"].feature.names
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)

print(f"Dataset loaded. Example train instance: {raw_datasets['train'][0]['tokens']}")
print(f"Labels: {label_list}")

# --- 2. Preprocess Data ---
print("Preprocessing dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Special token
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # Use the label of the first token of a word
            else:
                label_ids.append(-100) # Only label the first token of a multi-token word
                # Alternative: label_ids.append(label[word_idx]) if using B-, I- tagging fully
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

# --- 3. Fine-tune the Model with Hyperparameter Tuning ---
print("Setting up model and trainer...")
data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
                        for prediction, label in zip(predictions, labels)]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

best_f1 = 0
best_output_dir = None

for i, (lr, accum, wd, bs, epochs) in enumerate(hyperparam_combinations):
    print(f"Training model {i+1}/{len(hyperparam_combinations)} with lr={lr}, accum={accum}, wd={wd}, bs={bs}, epochs={epochs}")
    # Define a unique name for this run based on hyperparameters
    model_run_name = f"run_{i+1}-lr_{lr}-acc_{accum}-wd_{wd}-bs_{bs}-ep_{epochs}"
    # Set the output directory for this specific run within the main models directory
    run_dir = os.path.join(MODELS_BASE_DIR, model_run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    args = TrainingArguments(
        run_dir,  # Updated to use run_dir instead of OUTPUT_DIR
        eval_strategy="epoch",
        learning_rate=lr,  # Use current learning rate
        per_device_train_batch_size=bs,  # Use current batch size
        per_device_eval_batch_size=bs,  # Use current batch size
        num_train_epochs=epochs,  # Use current epochs
        weight_decay=wd,  # Use current weight decay
        logging_dir=f'{run_dir}/logs',  # Updated logging dir
        logging_steps=50,
        # early_stopping_patience=3,  # Wait 3 epochs
        # early_stopping_threshold=0.0,  # Any improvement counts
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Or use eval_lostt, precision, recall after implementing compute_metrics
        # --- Modifications for GPU Training ---
        # no_cuda=True,  # Disable CUDA/MPS, forcing CPU # Commented out to enable GPU
        # use_cpu=True,  # Force CPU usage             # Commented out to enable GPU
        fp16=True, # Enable mixed-precision training (requires compatible GPU and CUDA setup)
        # --- End Modifications ---
        gradient_accumulation_steps=accum,  # Use current gradient accumulation steps
        # push_to_hub=False, # Set to True if you want to upload to Hugging Face Hub
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=num_labels, id2label=id_to_label, label2id=label_to_id
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics # Add this if implemented
    )

    print(f"Starting training for run {i+1}...")
    trainer.train()

    print(f"Training finished for run {i+1}. Saving model...")
    trainer.save_model(run_dir)  # Save to run-specific directory
    tokenizer.save_pretrained(run_dir)
    print(f"Model saved to {run_dir}")

    # Evaluate on validation set to determine best model
    metrics = trainer.evaluate()
    log_message = f"Run {i+1} Hyperparameters:\n" \
                  f"  Learning Rate: {lr}\n" \
                  f"  Gradient Accumulation Steps: {accum}\n" \
                  f"  Weight Decay: {wd}\n" \
                  f"  Batch Size: {bs}\n" \
                  f"  Epochs: {epochs}\n\n" \
                  f"Evaluation Metrics:\n" \
                  f"  Eval Loss: {metrics['eval_loss']}\n" \
                  f"  Eval Precision: {metrics['eval_precision']}\n" \
                  f"  Eval Recall: {metrics['eval_recall']}\n" \
                  f"  Eval F1: {metrics['eval_f1']}\n"
    print(log_message)
    # Save log to a file specific to this run within its directory
    log_file_path = os.path.join(run_dir, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(log_message)
    
    # --- Evaluate the current model on the test set ---
    print(f"Evaluating run {i+1} model on test set...")
    # Load the model just saved for this run
    current_model = AutoModelForTokenClassification.from_pretrained(run_dir)
    current_tokenizer = AutoTokenizer.from_pretrained(run_dir)

    # Define test arguments - could potentially reuse some args, but defining separately for clarity
    test_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "test_eval"), # Dedicated output dir for test eval within the run dir
        per_device_eval_batch_size=bs, # Use the same batch size as training/validation for consistency
        # no_cuda=True, # Ensure consistency with training setup (comment/uncomment as needed)
        # use_cpu=True,
        fp16=True, # Ensure consistency with training setup
        report_to="none" # Avoid logging test eval to external platforms like wandb by default
    )

    # Create a new Trainer instance for test evaluation
    test_trainer = Trainer(
        model=current_model,
        args=test_args,
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=current_tokenizer,
        compute_metrics=compute_metrics
    )

    # Run evaluation on the test set
    test_metrics = test_trainer.evaluate(metric_key_prefix="test") # Use prefix to distinguish test metrics
    test_log_message = f"\nTest Set Evaluation Results for Run {i+1}:\n" \
                       f"  Test Loss: {test_metrics.get('test_loss', 'N/A')}\n" \
                       f"  Test Precision: {test_metrics.get('test_precision', 'N/A')}\n" \
                       f"  Test Recall: {test_metrics.get('test_recall', 'N/A')}\n" \
                       f"  Test F1: {test_metrics.get('test_f1', 'N/A')}\n"
    print(test_log_message)

    # Append test results to the run-specific log file
    with open(log_file_path, "a") as log_file:
        log_file.write(test_log_message)
    # --- End Test Set Evaluation ---

    if metrics["eval_f1"] > best_f1:
        best_f1 = metrics["eval_f1"]
        best_output_dir = run_dir

print(f"Best model based on validation F1 found at {best_output_dir} with F1 score {best_f1}")



# --- 5. Load Fine-tuned Model and Apply NER ---
print(f"Loading best model (based on validation F1) from {best_output_dir} for inference...")
# Ensure the model is loaded from the correct path where it was saved
ner_pipeline = pipeline(
    "ner",
    model=best_output_dir,  # Updated to use best_output_dir
    tokenizer=best_output_dir,  # Updated to use best_output_dir
    aggregation_strategy="simple", # Groups B-TAG and I-TAG entities
    device=0 if torch.cuda.is_available() else -1 # Use GPU (device 0) if available, otherwise use CPU (-1)
)

print("Processing tokenized text files...")
all_extracted_entities = {}

# Use the absolute path for listing files
for filename in os.listdir(TOKENIZED_TEXT_DIR):
    if filename.endswith("_tokenized.json"):
        # Use the absolute path for opening files
        filepath = os.path.join(TOKENIZED_TEXT_DIR, filename)
        print(f"Processing {filename}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Assuming JSON contains a list of strings (sentences or paragraphs)
                # Modify this part based on the actual structure of your JSON files
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    text_units = data
                elif isinstance(data, dict) and 'text' in data and isinstance(data['text'], list):
                     text_units = data['text'] # Adapt if structure is different
                else:
                    print(f"Warning: Unexpected JSON structure in {filename}. Skipping.")
                    continue

            extracted_persons = []
            # Process text units in batches for efficiency if needed
            results = ner_pipeline(text_units)

            # Ensure results is a list of lists
            if not isinstance(results, list) or not all(isinstance(sublist, list) for sublist in results):
                 print(f"Warning: Unexpected NER pipeline output format for {filename}. Output: {results}. Skipping.")
                 continue

            for unit_result in results:
                 for entity in unit_result:
                    # entity format example: {'entity_group': 'PER', 'score': 0.99, 'word': 'Name', 'start': 10, 'end': 14}
                    if entity['entity_group'] == 'PER':
                        extracted_persons.append({
                            'name': entity['word'],
                            'confidence': float(entity['score']) # Ensure score is JSON serializable
                            # Add start/end character indices if needed: 'start': entity['start'], 'end': entity['end']
                        })

            all_extracted_entities[filename.replace('_tokenized.json', '')] = extracted_persons
            print(f"Finished processing {filename}. Found {len(extracted_persons)} PER entities.")

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}. Skipping.")
        except Exception as e:
            print(f"An error occurred processing {filename}: {e}")

# --- 6. Save Results ---
# Use the absolute path for the output file
output_json_path = os.path.join(RESULTS_DIR, "extracted_characters.json")
print(f"Saving all extracted PER entities to {output_json_path}...")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_extracted_entities, f, ensure_ascii=False, indent=4)

print("NER process completed.")