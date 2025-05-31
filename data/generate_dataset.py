#!/usr/bin/env python3
"""
Generate train.jsonl (3 000 rows) and val.jsonl (300 rows) inside ./data

Each line is a JSON object:
{
  "prompt": "Count ...  Answer: (",
  "label": "k)"
}
where k is the correct integer 0-12.
"""

import json, random, pathlib, sys
from typing import Dict, List

# Vocabulary

TYPES: Dict[str, List[str]] = {
    "fruit":      ["apple", "banana", "cherry", "grape", "peach", "plum"],
    "animal":     ["dog", "cat", "horse", "cow", "sheep", "lion"],
    "vehicle":    ["car", "bus", "truck", "train", "plane", "boat"],
    "color":      ["red", "blue", "green", "yellow", "orange", "purple"],
    "country":    ["france", "china", "brazil", "india", "egypt", "spain"],
    "instrument": ["piano", "guitar", "violin", "flute", "drum", "trumpet"],
}
DISTRACT = [
    "chair", "cloud", "paper", "table", "bowl", "stone", "lamp", "river",
    "house", "sky", "mountain", "street", "window", "book", "phone", "map",
    "door", "shirt", "sock", "tree", "flower", "cup", "plate"
]

VOCAB = sorted(set(sum(TYPES.values(), []) + DISTRACT))

# Tokenizer safety check

def tokenizer_sanity(model_name="mistralai/Mistral-7B-v0.1") -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed; skipping tokenizer check")
        return
    tok = AutoTokenizer.from_pretrained(model_name)
    bad = [w for w in VOCAB if len(tok.encode(w, add_special_tokens=False)) > 1]
    if bad:
        raise ValueError(f"These words split into sub-tokens for {model_name}: {bad}")

# Generator example

PROMPT_TEMPLATE = (
    "Count the number of words in the following list that match the given "
    "type, and put the numerical answer in parentheses.\n"
    "Type: {typ}\n"
    "List: [{lst}]\n"
    "Answer: ("
)

TARGET_COUNTS = list(range(0, 11))

def make_example():
    typ   = random.choice(list(TYPES))
    k     = random.choice(TARGET_COUNTS)
    n_min = max(k, 3)
    n     = random.randint(n_min, 10)
    positives  = random.choices(TYPES[typ], k=k)
    negatives_pool = [w for w in VOCAB if w not in TYPES[typ]]
    negatives  = random.choices(negatives_pool, k=n-k)
    words      = positives + negatives
    random.shuffle(words)
    prompt = PROMPT_TEMPLATE.format(typ=typ, lst=" ".join(words))
    return {"prompt": prompt, "label": f"{k})"}

# File writer

def write_jsonl(path: pathlib.Path, n_rows: int) -> None:
    with path.open("w") as f:
        for _ in range(n_rows):
            json.dump(make_example(), f)
            f.write("\n")

# CLI entry point

def main() -> None:
    random.seed(42)
    tokenizer_sanity()
    pathlib.Path("data").mkdir(exist_ok=True)
    write_jsonl(pathlib.Path("data/train.jsonl"), 3000)
    write_jsonl(pathlib.Path("data/val.jsonl"),   300)
    print("✓ Dataset written to data/")

if __name__ == "__main__":
    main()
