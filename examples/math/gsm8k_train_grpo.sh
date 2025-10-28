#!/bin/bash
# Train GRPO models on GSM8K with strict hash format

# echo "=========================================="
# echo "Training Qwen3-1.7B GRPO (strict hash)"
# echo "=========================================="
# python3 -m areal.launcher.local \
#   examples/math/gsm8k_grpo_megatron.py \
#   --config examples/math/gsm8k_grpo_megatron.yaml \
#   actor.path=Qwen/Qwen3-1.7B \
#   async_training=true \
#   rollout.max_head_offpolicyness=4 \
#   train_dataset.batch_size=64 \
#   gconfig.n_samples=16 \
#   total_train_epochs=3 \
#   experiment_name=gsm8k-grpo \
#   trial_name=strict-hash

# echo ""
# echo "=========================================="
# echo "Training Qwen3-8B GRPO (strict hash)"
# echo "=========================================="
# python3 -m areal.launcher.local \
#   examples/math/gsm8k_grpo_megatron.py \
#   --config examples/math/gsm8k_grpo_megatron.yaml \
#   actor.path=Qwen/Qwen3-8B \
#   async_training=true \
#   rollout.max_head_offpolicyness=4 \
#   train_dataset.batch_size=64 \
#   gconfig.n_samples=16 \
#   total_train_epochs=3 \
#   experiment_name=gsm8k-grpo-8b \
#   trial_name=strict-hash

echo ""
echo "=========================================="
echo "Training Qwen3-14B GRPO (strict hash)"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_grpo_megatron.py \
  --config examples/math/gsm8k_grpo_megatron.yaml \
  actor.path=Qwen/Qwen3-14B \
  async_training=true \
  rollout.max_head_offpolicyness=4 \
  train_dataset.batch_size=32 \
  gconfig.n_samples=32 \
  total_train_epochs=3 \
  allocation_mode=sglang.d4p1t1+d1p2t2 \
  experiment_name=gsm8k-grpo-14b \
  +actor.use_lora=true \
  +actor.lora_rank=32 \
  +actor.lora_alpha=16 \
  +actor.peft_type=lora \
  +actor.target_modules=[all] \
  +sglang.enable_lora=true \
  +sglang.max_lora_rank=32 \
  +sglang.lora_target_modules=[all] \
  trial_name=strict-hash

echo ""
echo "=========================================="
echo "GRPO Training Complete!"
echo "=========================================="
echo "Checkpoints:"
echo "  14B: /tmp/areal/experiments/checkpoints/ubuntu/gsm8k-grpo-14b/strict-hash/default/"
echo "Logs:"
echo "  14B: /tmp/areal/experiments/logs/ubuntu/gsm8k-grpo-14b/strict-hash/"
