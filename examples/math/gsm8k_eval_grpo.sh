#!/bin/bash
# Evaluate GRPO models on GSM8K with strict hash format

set -e

# Find the latest checkpoint for 1.7B
CKPT_1_7B=$(ls -d /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo/strict-hash/default/epoch* 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_1_7B" ]; then
  echo "Error: No checkpoint found for 1.7B GRPO"
  exit 1
fi

# Find the latest checkpoint for 8B
CKPT_8B=$(ls -d /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo-8b/strict-hash/default/epoch* 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_8B" ]; then
  echo "Error: No checkpoint found for 8B GRPO"
  exit 1
fi

echo "=========================================="
echo "Evaluating Qwen3-1.7B GRPO (strict hash)"
echo "Checkpoint: $CKPT_1_7B"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_eval.py \
  --config examples/math/gsm8k_eval_config.yaml \
  actor.path=$CKPT_1_7B \
  gconfig.temperature=0.0 \
  gconfig.n_samples=1 \
  valid_dataset.batch_size=32 \
  cluster.n_gpus_per_node=8 \
  allocation_mode=sglang.d8p1t1+d8p1t1 \
  experiment_name=eval-qwen3-1.7b-grpo-strict \
  trial_name=eval

echo ""
echo "=========================================="
echo "Evaluating Qwen3-8B GRPO (strict hash)"
echo "Checkpoint: $CKPT_8B"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_eval.py \
  --config examples/math/gsm8k_eval_config.yaml \
  actor.path=$CKPT_8B \
  gconfig.temperature=0.0 \
  gconfig.n_samples=1 \
  valid_dataset.batch_size=32 \
  cluster.n_gpus_per_node=8 \
  allocation_mode=sglang.d8p1t1+d8p1t1 \
  experiment_name=eval-qwen3-8b-grpo-strict \
  trial_name=eval

echo ""
echo "=========================================="
echo "Results:"
echo "  1.7B: /tmp/areal/experiments/logs/ubuntu/eval-qwen3-1.7b-grpo-strict/eval/trainer.log"
echo "  8B:   /tmp/areal/experiments/logs/ubuntu/eval-qwen3-8b-grpo-strict/eval/trainer.log"
echo "=========================================="
