#!/bin/bash
# Train GRPO models on GSM8K with strict hash format

set -e

echo "=========================================="
echo "Training Qwen3-1.7B GRPO"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_grpo_megatron.py \
  --config examples/math/gsm8k_grpo_megatron.yaml \
  actor.path=Qwen/Qwen3-1.7B \
  async_training=true \
  rollout.max_head_offpolicyness=4 \
  train_dataset.batch_size=64 \
  gconfig.n_samples=16 \
  experiment_name=gsm8k-grpo

echo ""
echo "=========================================="
echo "Training Qwen3-8B GRPO"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_grpo_megatron.py \
  --config examples/math/gsm8k_grpo_megatron.yaml \
  actor.path=Qwen/Qwen3-8B \
  async_training=true \
  rollout.max_head_offpolicyness=4 \
  train_dataset.batch_size=64 \
  gconfig.n_samples=16 \
  experiment_name=gsm8k-grpo-8b

echo ""
echo "=========================================="
echo "GRPO Training Complete!"
echo "=========================================="
echo "Checkpoints:"
echo "  1.7B: /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo/trial0/default/"
echo "  8B:   /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo-8b/trial0/default/"
echo "Logs:"
echo "  1.7B: /tmp/areal/experiments/logs/ubuntu/gsm8k-grpo/trial0/"
echo "  8B:   /tmp/areal/experiments/logs/ubuntu/gsm8k-grpo-8b/trial0/"
