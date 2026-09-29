import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


MODEL_DIR = "models/distilbert_depression"
TEST_FILE = "data/test.csv"


# Load test data
test_df = pd.read_csv(TEST_FILE)

test_df = test_df[["text", "labels"]]
test_df["labels"] = test_df["labels"].astype(int)

print("Test data:", test_df.shape)


# Convert to Hugging Face Dataset
test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# Evaluation configuration
args = TrainingArguments(
    output_dir="reports/evaluation",
    per_device_eval_batch_size=2,
    use_cpu=True,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    processing_class=tokenizer,
)


# Predict on test dataset
print("\nRunning test evaluation...")

results = trainer.predict(test_dataset)

predictions = np.argmax(
    results.predictions,
    axis=1
)

true_labels = results.label_ids


# Metrics
accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n========== TEST RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        true_labels,
        predictions,
        zero_division=0
    )
)


print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        true_labels,
        predictions
    )
)

print("\nEvaluation completed successfully!")