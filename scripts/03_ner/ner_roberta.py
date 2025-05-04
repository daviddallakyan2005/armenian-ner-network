import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch
from itertools import product
from seqeval.metrics import precision_score, recall_score, f1_score # type: ignore

# --- Diagnostics ---
print(f"--- Running: {__file__} ---")
print(f"Python Executable: {sys.executable}")
print(f"CUDA Available Check: {torch.cuda.is_available()}") # Check before transformers import
print(f"PyTorch CUDA Built Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# These are the parameters used for the best performing model, but the core logic is based on the list of hyperparameters below
MODEL_CHECKPOINT = "xlm-roberta-base"
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 7
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
MODEL_NAME = f"{MODEL_CHECKPOINT}-finetuned-pioner-ner"

# Define paths relative to the project root
MODELS_BASE_DIR = os.path.join(PROJECT_ROOT, "models")
PIONER_SILVER_DIR = os.path.join(PROJECT_ROOT, "external_repos", "pioner", "pioner-silver")
PIONER_GOLD_DIR = os.path.join(PROJECT_ROOT, "external_repos", "pioner", "pioner-gold")
TOKENIZED_TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "tokenized")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_BASE_DIR, exist_ok=True)

# Hyperparameters for tuning
learning_rates = [1e-5, 2e-5, 3e-5]
grad_accum_steps = [1, 2, 4]
weight_decays = [0, 0.01]
batch_sizes = [8]
epochs_list = [3, 7]

hyperparam_combinations = list(product(learning_rates, grad_accum_steps, weight_decays, batch_sizes, epochs_list))

# --- 1. Load pioNER Dataset ---
print("Loading pioNER dataset...")

data_files = {
    "train": os.path.join(PIONER_SILVER_DIR, "train.conll03"),
    "validation": os.path.join(PIONER_SILVER_DIR, "dev.conll03"),
    "test": os.path.join(PIONER_GOLD_DIR, "test.conll03"),
}


raw_datasets = load_dataset(
    'conll2003',
    data_files=data_files,
    trust_remote_code=True # Added to allow execution of dataset script code
)


label_list = raw_datasets["train"].features["ner_tags"].feature.names
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)

print(f"Dataset loaded. Example train instance: {raw_datasets['train'][0]['tokens']}")
print(f"Labels: {label_list}")


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
    if (lr == 1e-5 and accum == 1 and wd == 0.0 and (epochs == 3 or epochs == 7)):
        continue
    print(f"Training model {i+1}/{len(hyperparam_combinations)} with lr={lr}, accum={accum}, wd={wd}, bs={bs}, epochs={epochs}")
    
    model_run_name = f"run_{i+1}-lr_{lr}-acc_{accum}-wd_{wd}-bs_{bs}-ep_{epochs}"
    run_dir = os.path.join(MODELS_BASE_DIR, model_run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    args = TrainingArguments(
        run_dir,
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=wd,
        logging_dir=f'{run_dir}/logs',
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(torch.cuda.is_available()),
        gradient_accumulation_steps=accum,
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
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"Model saved to {run_dir}")


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
    
    log_file_path = os.path.join(run_dir, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(log_message)
    

    print(f"Evaluating run {i+1} model on test set...")
    current_model = AutoModelForTokenClassification.from_pretrained(run_dir)
    current_tokenizer = AutoTokenizer.from_pretrained(run_dir)

    test_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "test_eval"),
        per_device_eval_batch_size=bs,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )


    test_trainer = Trainer(
        model=current_model,
        args=test_args,
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=current_tokenizer,
        compute_metrics=compute_metrics
    )

    test_metrics = test_trainer.evaluate(metric_key_prefix="test")
    test_log_message = f"\nTest Set Evaluation Results for Run {i+1}:\n" \
                       f"  Test Loss: {test_metrics.get('test_loss', 'N/A')}\n" \
                       f"  Test Precision: {test_metrics.get('test_precision', 'N/A')}\n" \
                       f"  Test Recall: {test_metrics.get('test_recall', 'N/A')}\n" \
                       f"  Test F1: {test_metrics.get('test_f1', 'N/A')}\n"
    print(test_log_message)

    with open(log_file_path, "a") as log_file:
        log_file.write(test_log_message)

    if metrics["eval_f1"] > best_f1:
        best_f1 = metrics["eval_f1"]
        best_output_dir = run_dir

print(f"Best model based on validation F1 found at {best_output_dir} with F1 score {best_f1}")