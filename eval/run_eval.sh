#!/usr/bin/env bash
SPLIT="data/val.jsonl"
MODELS=(
  mistralai/Mistral-7B-v0.1
  meta-llama/Meta-Llama-3-8B
  Qwen/Qwen1.5-4B
)
for m in "${MODELS[@]}"; do
  echo "----- $m -----"
  python eval/zeroshot_eval.py --model "$m" --split "$SPLIT"
done | tee eval/results.log
