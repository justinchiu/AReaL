# GSM8K Examples

This directory contains scripts for training and evaluating models on the GSM8K math reasoning dataset.

## Supervised Fine-Tuning (SFT) with Megatron

### Single Model Training

Train a single model using the Megatron backend:

```bash
python3 -m areal.launcher.local examples/math/gsm8k_sft_megatron.py \
    --config examples/math/gsm8k_sft_megatron.yaml \
    actor.path=Qwen/Qwen3-8B \
    trial_name=qwen3-8b
```

### Batch Training Multiple Model Sizes

Train all Qwen3 model sizes sequentially:

```bash
python3 examples/math/run_all_qwen3_sizes.py
```

This will train: Qwen3-1.7B, Qwen3-8B, Qwen3-14B, Qwen3-30B-A3B, Qwen3-32B

**Key configurations:**
- Backend: Megatron with FSDP
- Allocation: `d2p2t2` (data=2, pipeline=2, tensor=2)
- Batch size: 128
- Learning rate: 2e-5
- Training epochs: 1
- Dataset format: Chat template with `#### {answer}` format

### Evaluation

#### Single Model Evaluation

```bash
python3 -m areal.launcher.local examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_eval_config.yaml \
    actor.path=/path/to/checkpoint \
    trial_name=eval-run
```

#### Batch Evaluation All Models

Evaluate both base and fine-tuned models for all sizes:

```bash
python3 examples/math/eval_all_qwen3_sizes.py
```

This compares base model performance vs fine-tuned checkpoints and calculates accuracy deltas.

**Evaluation configuration:**
- Config: `gsm8k_eval_config.yaml` (optimized for 8 GPUs)
- Allocation: `sglang.d8p1t1+d8p1t1` (8 inference servers)
- Batch size: 16
- Max new tokens: 512

**Output locations:**
- Base model generations: `~/areal_experiments/logs/ubuntu/gsm8k-base-eval/{model}-base/generated/0/*.txt`
- Fine-tuned generations: `~/areal_experiments/logs/ubuntu/gsm8k-sft-megatron_eval/{model}/generated/0/*.txt`

## Hyper-parameters for GSM8K GRPO on Qwen2.5-1.5b-Instruct

The hyperparameters given in gsm8k_grpo.yaml is the set that we found to achieve the
highest max `grpo-eval/task_reward/avg` during training for `Qwen2.5-1.5b-Instruct`. You
are free to try out more of the hyperparameters listed below!

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

### Other Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)
