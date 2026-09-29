# Project Report: Depression Detection from Text Using DistilBERT

## Abstract
This project demonstrates an NLP-based text classification pipeline inspired by the BERT-XDD research methodology. Because the target laptop uses an Intel i5-1235U processor and 16 GB RAM without a dedicated training GPU, the implementation replaces the original BERTweet-large backbone with DistilBERT. The system performs text preprocessing, tokenization, transformer-based classification, evaluation, and single-text inference.

## Problem Statement
Social-media text can contain linguistic signals associated with different mental-health-related categories. The objective of this educational project is to investigate whether a transformer-based NLP model can classify text into predefined dataset labels.

## Objectives
1. Prepare and clean text data.
2. Tokenize text with a pretrained transformer tokenizer.
3. Fine-tune a lightweight transformer model.
4. Measure accuracy and macro-F1.
5. Produce a confusion matrix and classification report.
6. Demonstrate inference through a command-line tool and optional web interface.

## Methodology
The pipeline is:

Text → Cleaning → DistilBERT Tokenization → Transformer Encoder → Classification Head → Predicted Class

The model uses `distilbert-base-uncased`. A maximum sequence length of 128 and batch size of 2 are used to reduce memory pressure.

## Hardware Constraint
The project is configured for an Intel Core i5-1235U and 16 GB RAM. CPU training is substantially slower than GPU training, so the default configuration uses one epoch and a small batch size. Exact reproduction of the original BERT-XDD BERTweet-large experiment requires a more capable GPU environment.

## Evaluation
Use the project's test set and run:

`python src\\evaluate.py`

Record the generated accuracy, precision, recall, macro-F1 and confusion matrix in the final college report. Do not invent results.

## Limitations
- Text classification does not establish a medical diagnosis.
- Dataset labels may contain annotation noise and demographic/domain bias.
- The smaller DistilBERT model is not equivalent to BERTweet-large.
- CPU training is slow.
- Results depend on the dataset and train/validation/test split.

## Future Work
- Compare DistilBERT with TF-IDF + Logistic Regression.
- Test a Twitter/social-media-specific lightweight model.
- Add explainability such as token-level importance.
- Run a controlled GPU experiment for comparison.
- Perform bias and error analysis.

## Reference
Belcastro, L., Cantini, R., Marozzo, F., Talia, D., & Trunfio, P. (2024). *Detecting mental disorder on social media: a ChatGPT-augmented explainable approach*. BERT-XDD.
