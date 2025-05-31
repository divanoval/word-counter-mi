#!/usr/bin/env python3
"""
Example uses:
python eval/zeroshot_eval.py --model mistralai/Mistral-7B-v0.1
python eval/zeroshot_eval.py --model meta-llama/Meta-Llama-3-8B --split data/train.jsonl
"""
import argparse, json, re, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# settings
MAX_NEW_TOKENS   = 4         # no reasoning
TEMPERATURE      = 0.0       # greedy
NUMERIC_PATTERN  = re.compile(r"\((\d+)")   # (number

# utilities
def extract_number(text: str):
    """Return the first number inside parentheses, or None."""
    m = NUMERIC_PATTERN.search(text)
    return m.group(1) if m else None

@torch.inference_mode()
def evaluate(model_name: str, split_path: str) -> float:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    correct = total = 0
    for line in tqdm(Path(split_path).open()):
        ex   = json.loads(line)
        ids  = tok(ex["prompt"], return_tensors="pt").to(mdl.device)
        out  = mdl.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
        )
        gen_text = tok.decode(out[0][ids["input_ids"].shape[1]:])
        pred     = extract_number(gen_text)
        gold     = ex["label"][:-1]      # strip trailing )
        correct += (pred == gold)
        total   += 1
    return correct / total

# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model repo or path")
    p.add_argument("--split", default="data/val.jsonl")
    args = p.parse_args()

    acc = evaluate(args.model, args.split)
    print(f"{args.model}: {acc:.2%} accuracy on {args.split}")

if __name__ == "__main__":
    main()
