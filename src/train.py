import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "data/train.csv"
DEV_FILE = "data/dev.csv"

OUTPUT_DIR = "models/distilbert_depression"


# -----------------------------
# Load dataset
# -----------------------------
train_df = pd.read_csv(TRAIN_FILE)
dev_df = pd.read_csv(DEV_FILE)

# Keep only the columns we need
train_df = train_df[["text", "labels"]]
dev_df = dev_df[["text", "labels"]]

# Make sure labels are integers
train_df["labels"] = train_df["labels"].astype(int)
dev_df["labels"] = dev_df["labels"].astype(int)

print("Training data:", train_df.shape)
print("Validation data:", dev_df.shape)

print("\nLabel distribution:")
print(train_df["labels"].value_counts().sort_index())


# -----------------------------
# Convert to Hugging Face Dataset
# -----------------------------
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
dev_dataset = Dataset.from_pandas(dev_df, preserve_index=False)


# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

dev_dataset = dev_dataset.map(
    tokenize_function,
    batched=True
)


# -----------------------------
# Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)


# -----------------------------
# Evaluation metrics
# -----------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


# -----------------------------
# Training configuration
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_steps=50,

    report_to="none",

    use_cpu=True,
)


# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


# -----------------------------
# Start training
# -----------------------------
print("\nStarting training...")
print("This will run on your CPU.")

trainer.train()


# -----------------------------
# Save final model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining completed!")
print("Model saved to:", OUTPUT_DIR)