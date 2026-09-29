import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "models/distilbert_depression"


print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

print("Model loaded successfully!")
print("Enter a text to analyze.")
print("Type 'exit' to stop.\n")


while True:

    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    if not text.strip():
        print("Please enter some text.\n")
        continue

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence = probabilities[0][prediction].item() * 100

    print("\nPrediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 40)