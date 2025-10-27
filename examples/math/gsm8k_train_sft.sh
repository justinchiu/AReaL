#!/bin/bash
# Train SFT models on GSM8K with strict hash format

echo "=========================================="
echo "Training Qwen3-1.7B SFT"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_sft_megatron.py \
  --config examples/math/gsm8k_sft_megatron.yaml \
  model.path=Qwen/Qwen3-1.7B \
  experiment_name=gsm8k-sft-megatron \
  trial_name=trial0

echo ""
echo "=========================================="
echo "Training Qwen3-8B SFT"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_sft_megatron.py \
  --config examples/math/gsm8k_sft_megatron.yaml \
  model.path=Qwen/Qwen3-8B \
  experiment_name=gsm8k-sft-megatron-8b \
  trial_name=trial0

echo ""
echo "=========================================="
echo "Training Qwen3-8B SFT"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_sft_megatron.py \
  --config examples/math/gsm8k_sft_megatron.yaml \
  model.path=Qwen/Qwen3-14B \
  experiment_name=gsm8k-sft-megatron-14b \
  trial_name=trial0

echo ""
echo "=========================================="
echo "SFT Training Complete!"
echo "=========================================="
echo "Checkpoints:"
echo "  1.7B: /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-sft-megatron/trial0/default/"
echo "  8B:   /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-sft-megatron-8b/trial0/default/"
echo "Logs:"
echo "  1.7B: /tmp/areal/experiments/logs/ubuntu/gsm8k-sft-megatron/trial0/"
echo "  8B:   /tmp/areal/experiments/logs/ubuntu/gsm8k-sft-megatron-8b/trial0/"
