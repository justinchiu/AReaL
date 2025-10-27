#!/bin/bash
# Evaluate base models on GSM8K with strict hash format

echo "=========================================="
echo "Evaluating Qwen3-1.7B Base"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_eval.py \
  --config examples/math/gsm8k_eval_config.yaml \
  actor.path=Qwen/Qwen3-1.7B \
  gconfig.temperature=0.0 \
  gconfig.n_samples=1 \
  valid_dataset.batch_size=32 \
  cluster.n_gpus_per_node=8 \
  allocation_mode=sglang.d8p1t1+d8p1t1 \
  experiment_name=eval-qwen3-1.7b-base \
  trial_name=eval

echo ""
echo "=========================================="
echo "Evaluating Qwen3-8B Base"
echo "=========================================="
python3 -m areal.launcher.local \
  examples/math/gsm8k_eval.py \
  --config examples/math/gsm8k_eval_config.yaml \
  actor.path=Qwen/Qwen3-8B \
  gconfig.temperature=0.0 \
  gconfig.n_samples=1 \
  valid_dataset.batch_size=32 \
  cluster.n_gpus_per_node=8 \
  allocation_mode=sglang.d8p1t1+d8p1t1 \
  experiment_name=eval-qwen3-8b-base \
  trial_name=eval

echo ""
echo "=========================================="
echo "Results:"
echo "  1.7B: /tmp/areal/experiments/logs/ubuntu/eval-qwen3-1.7b-base/eval/trainer.log"
echo "  8B:   /tmp/areal/experiments/logs/ubuntu/eval-qwen3-8b-base/eval/trainer.log"
echo "=========================================="
