# prepare_pairs.py
import os
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

OUT_DIR = "prepared_data"
TRAIN_OUT = os.path.join(OUT_DIR, "pairs.tsv")
VALID_OUT = os.path.join(OUT_DIR, "valid_pairs.tsv")

# ---------- Primary plan: community bundle (needs trust_remote_code) ----------
NLI_SOURCE = "shibing624/nli_zh"
PRIMARY_SUBSETS = ["LCQMC", "BQ", "ATEC", "PAWSX", "STS-B"]
STS_POS_THRESHOLD = 3.8  # keep strong positives

# ---------- Fallback plan: built-in loaders (no trust_remote_code needed) ----------
FALLBACKS = [
    ("clue", "afqmc", "binary"),           # sentence1, sentence2, label in {0,1}
    ("paws-x", "zh", "binary"),            # sentence1, sentence2, label in {0,1}
    ("stsb_multi_mt", "zh", "stsb"),       # sentence1, sentence2, score in [0,5]
]

MIN_LEN = 2
MAX_LEN = 128
VALID_RATIO = 0.02  # 2%

def norm(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().replace("\u3000", " ")
    return t

def ok(t: str) -> bool:
    L = len(t)
    return (L >= MIN_LEN) and (L <= MAX_LEN)

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["query"] != df["positive"]].copy()
    df["k1"] = df["query"] + "||" + df["positive"]
    df["k2"] = df["positive"] + "||" + df["query"]
    df["canon"] = df[["k1", "k2"]].min(axis=1)
    df = df.drop_duplicates(subset=["canon"]).drop(columns=["k1", "k2", "canon"])
    return df

def collect_primary() -> pd.DataFrame:
    rows = []
    for name in PRIMARY_SUBSETS:
        print(f"[INFO] Loading subset: {name}")
        ds = load_dataset(NLI_SOURCE, name, trust_remote_code=True)
        if name == "STS-B":
            for split in ds.keys():
                for ex in tqdm(ds[split], desc=f"  parsing {name}/{split}"):
                    s1 = norm(ex.get("sentence1", ""))
                    s2 = norm(ex.get("sentence2", ""))
                    score = ex.get("score", None)
                    if score is None:
                        continue
                    if ok(s1) and ok(s2) and float(score) >= STS_POS_THRESHOLD:
                        rows.append((s1, s2))
        else:
            for split in ds.keys():
                for ex in tqdm(ds[split], desc=f"  parsing {name}/{split}"):
                    s1 = norm(ex.get("sentence1", ""))
                    s2 = norm(ex.get("sentence2", ""))
                    label = ex.get("label", None)
                    if label is None:
                        continue
                    if int(label) == 1 and ok(s1) and ok(s2):
                        rows.append((s1, s2))
    df = pd.DataFrame(rows, columns=["query", "positive"])
    print(f"[INFO] Primary collected: {len(df)}")
    return df

def collect_fallbacks() -> pd.DataFrame:
    rows = []
    for name, config, kind in FALLBACKS:
        print(f"[INFO] Fallback loading: {name}/{config} ({kind})")
        ds = load_dataset(name, config)
        for split in ds.keys():
            for ex in tqdm(ds[split], desc=f"  parsing {name}/{config}/{split}"):
                s1 = norm(ex.get("sentence1", ""))
                s2 = norm(ex.get("sentence2", ""))
                if not (ok(s1) and ok(s2)):
                    continue
                if kind == "binary":
                    label = ex.get("label", None)
                    if label is None:
                        continue
                    if int(label) == 1:
                        rows.append((s1, s2))
                elif kind == "stsb":
                    score = ex.get("label", ex.get("score", None))
                    if score is None:
                        continue
                    if float(score) >= STS_POS_THRESHOLD:
                        rows.append((s1, s2))
    df = pd.DataFrame(rows, columns=["query", "positive"])
    print(f"[INFO] Fallback collected: {len(df)}")
    return df

def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", header=False, index=False, encoding="utf-8")
    print(f"[OK] Saved: {path} ({len(df)} lines)")

def main():
    all_df = pd.DataFrame(columns=["query", "positive"])
    # Try primary (needs trust_remote_code). If it fails, use fallbacks.
    try:
        primary = collect_primary()
        all_df = pd.concat([all_df, primary], ignore_index=True)
    except Exception as e:
        print(f"[WARN] Primary datasets failed: {e}")
        print("[INFO] Switching to built-in fallbacks...")
        fb = collect_fallbacks()
        all_df = pd.concat([all_df, fb], ignore_index=True)

    if all_df.empty:
        raise SystemExit("[ERROR] No data collected. Check network or dataset availability.")

    all_df = dedup(all_df)
    all_df = all_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_valid = max(1000, int(len(all_df) * VALID_RATIO))
    valid_df = all_df.iloc[:n_valid].copy()
    train_df = all_df.iloc[n_valid:].copy()

    save_tsv(train_df, TRAIN_OUT)
    save_tsv(valid_df, VALID_OUT)

    print("\n[STATS]")
    print(f"  train pairs: {len(train_df)}")
    print(f"  valid pairs: {len(valid_df)}")
    print("  sample pairs:")
    print(train_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
