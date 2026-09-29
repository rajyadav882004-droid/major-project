import re
import pandas as pd

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[["text", "labels"]].dropna()
    df["text"] = df["text"].map(clean_text)
    df["labels"] = pd.to_numeric(df["labels"], errors="raise").astype(int)
    return df
