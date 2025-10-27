#!/bin/bash
# Evaluate GRPO models on GSM8K with strict hash format

# Use epoch 3 checkpoint for both models
CKPT_1_7B="/tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo/strict-hash/default/epoch2epochstep116globalstep350"
CKPT_8B="/tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo-8b/strict-hash/default/epoch2epochstep116globalstep350"

# Check if 1.7B checkpoint exists
if [ ! -d "$CKPT_1_7B" ]; then
  echo "Error: Epoch 3 checkpoint not found for 1.7B GRPO at $CKPT_1_7B"
  exit 1
fi

# Check if 8B checkpoint exists
if [ ! -d "$CKPT_8B" ]; then
  echo "Warning: Epoch 3 checkpoint not found for 8B GRPO at $CKPT_8B"
  echo "Skipping 8B evaluation"
  SKIP_8B=1
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

if [ -z "$SKIP_8B" ]; then
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
fi

echo ""
echo "=========================================="
echo "Results:"
echo "  1.7B: /tmp/areal/experiments/logs/ubuntu/eval-qwen3-1.7b-grpo-strict/eval/trainer.log"
echo "  8B:   /tmp/areal/experiments/logs/ubuntu/eval-qwen3-8b-grpo-strict/eval/trainer.log"
echo "=========================================="
