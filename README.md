# Depression Detection from Text — CPU-Friendly College Project

An educational NLP project inspired by the BERT-XDD research repository. The original BERT-XDD uses BERTweet and a GPU-oriented training setup; this project uses **DistilBERT** so it is substantially lighter for a laptop with an Intel i5-1235U and 16 GB RAM.

> **Important:** This is a college/research demonstration, NOT a medical diagnostic system. A text classifier cannot diagnose depression or determine whether someone has a mental-health condition.

## Features
- Text cleaning and dataset validation
- DistilBERT text classification
- CPU-friendly training defaults
- Accuracy, macro-F1, precision and recall
- Confusion matrix
- Single-text prediction
- Optional Streamlit demo UI

## Expected dataset
The training scripts expect CSV files:

```text
data/train.csv
data/dev.csv
data/test.csv
```

with columns:

```text
text,labels
```

where `labels` are integer class IDs (normally 0, 1, 2 for the BERT-XDD-style three-class setup).

You can place the BERT-XDD dataset in `data/` after obtaining it from the original project. Do not claim results from the included demo data as research results.

## Windows installation

Open Command Prompt in this folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

```bat
python src\train.py
```

The defaults are deliberately conservative:

- model: `distilbert-base-uncased`
- max length: 128
- batch size: 2
- epochs: 1
- CPU unless CUDA is detected

The first run downloads the pretrained model from Hugging Face, so internet access is required.

## Evaluate

```bat
python src\evaluate.py
```

## Predict one text

```bat
python src\predict.py "I have been feeling very low lately"
```

The prediction is a model classification only and must not be interpreted as a diagnosis.

## Optional web demo

```bat
streamlit run app.py
```

## Safety / resource notes

Do not run the original BERT-XDD notebook as Administrator. This project does not require administrator privileges.

If RAM becomes heavily saturated during training, stop the run and reduce the dataset size or sequence length. CPU training can be slow.

## Academic honesty

This project is an implementation/adaptation for educational use. Cite the original BERT-XDD paper/repository when discussing the inspiration. Do not present generated/demo results as results from the original paper.
