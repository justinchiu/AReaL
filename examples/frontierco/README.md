# FrontierCo Code Generation Examples

This directory contains training and evaluation scripts for code generation tasks using the AReaL framework with Megatron backend.

## Quick Start

### 1. Supervised Fine-Tuning (SFT)

Train a single model on code generation:

```bash
python3 -m areal.launcher.local examples/frontierco/frontierco_sft_megatron.py \
    --config examples/frontierco/frontierco_sft_megatron.yaml \
    model.path=Qwen/Qwen3-8B \
    trial_name=qwen3-8b
```

**Key SFT Configuration**:
- Backend: Megatron with model/tensor/pipeline parallelism
- Allocation: `d2p2t2` (data=2, pipeline=2, tensor=2)
- Batch size: 128
- Learning rate: 2e-5
- Max sequence length: 8192 tokens
- Training epochs: 1

### 2. Reinforcement Learning (GRPO)

Train with RL after SFT:

```bash
python3 -m areal.launcher.local examples/frontierco/frontierco_grpo_megatron.py \
    --config examples/frontierco/frontierco_grpo_megatron.yaml \
    actor.path=/path/to/sft/checkpoint \
    trial_name=qwen3-8b-rl
```

**Key GRPO Configuration**:
- Allocation: `sglang.d4p1t1+d1p2t2` (4 SGLang servers + Megatron training)
- n_samples: 8 (for pass@8 evaluation)
- Temperature: 1.0 (for diversity)
- Max new tokens: 2048
- Reward scaling: 5.0
- Training epochs: 10

### 3. Evaluation

#### Internal Evaluation (On Training Data)

Evaluate a trained model on OR-Instruct-Data-3K:

```bash
python3 -m areal.launcher.local examples/frontierco/frontierco_eval.py \
    --config examples/frontierco/frontierco_eval_config.yaml \
    actor.path=/path/to/checkpoint \
    trial_name=eval-run
```

**Evaluation Configuration**:
- Allocation: `sglang.d8p1t1+d8p1t1` (uses all 8 GPUs for faster evaluation)
- Batch size: 16
- Max new tokens: 2048

#### CoBench/FrontierCO Benchmark Evaluation

For evaluation on the **CoBench/FrontierCO benchmark** (real-world combinatorial optimization problems):

See **[COBENCH_INTEGRATION.md](COBENCH_INTEGRATION.md)** for complete integration guide.

**Quick Setup**:
```bash
# 1. Download CoBench dataset (637MB)
huggingface-cli download CO-Bench/CO-Bench --repo-type dataset --local-dir data/comb_opt

# 2. Copy AReaL agent to your CoBench repo
cp examples/frontierco/areal_cobench_agent.py /path/to/cobench/evals/agents/

# 3. Run evaluation
cd /path/to/cobench
python -m evals.tasks.cobench \
    agent=evals.agents.areal_cobench_agent.AReaLAgentConfig \
    agent.model_path=/path/to/areal/checkpoint \
    task_names='["Job shop scheduling", "Aircraft landing"]' \
    output_path=results.jsonl

# 4. Visualize results
streamlit run cobench_visualizer.py
```

**What you get**:
- ✅ Ready-to-use agent (`areal_cobench_agent.py`) that wraps AReaL models
- ✅ Automatic prompt formatting matching OR-Instruct training
- ✅ Support for 30+ optimization problems (TSP, knapsack, scheduling, etc.)
- ✅ Solution quality metrics and success rates
- ✅ Visual results dashboard with syntax-highlighted code

See [COBENCH_INTEGRATION.md](COBENCH_INTEGRATION.md) for detailed instructions.

## Dataset Setup

✅ **Currently configured**: **CardinalOperations/OR-Instruct-Data-3K**

This dataset contains 3,000 operations research problems with mathematical models and Python code using `coptpy`.

**Dataset Details**:
- **Size**: 3,000 examples
- **Format**: `prompt` (OR question + instruction) → `completion` (full solution with model + code)
- **Good for**: SFT (Supervised Fine-Tuning)
- **Not ideal for**: RL training (no test cases for code execution rewards)

**Alternative Datasets**:
- **For code completion with RL**: Use `openai/humaneval` (change `path: frontierco` to `path: openai/humaneval` in configs)
- **For custom data**: See [DATASET.md](DATASET.md) for instructions

**Dataset Format**:
```json
{
  "prompt": "Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy`...",
  "completion": "## Mathematical Model:\nTo solve...[full solution with code]"
}
```

## Code Execution & Rewards

The reward function in `examples/frontierco/reward.py` evaluates generated code by executing it against test cases.

**Execution Methods**:
1. **Fallback (default)**: Local Python `exec()` without sandboxing
2. **Swerex (recommended)**: Containerized execution with proper isolation

**Using Swerex**:
To enable swerex-based code execution (more secure and scalable):

```python
from swerex.deployment.modal import ModalDeployment

# Initialize deployment
deployment = ModalDeployment(
    image="python:3.12",
    startup_timeout=60,
    deployment_timeout=3600,
)

# Pass deployment to reward function
# (requires modifying frontierco_grpo_megatron.py to pass deployment to workflow)
```

**Reward Function Features**:
- Binary rewards (1.0 if all tests pass, 0.0 otherwise)
- Code extraction from markdown blocks
- Library installation support (via `# LIBRARIES:` comment)
- Timeout handling
- Fallback to local execution if swerex unavailable

## Output Locations

Generated code samples are saved to:
- **SFT evaluation**: `~/areal_experiments/logs/ubuntu/frontierco-sft-megatron/{trial_name}/`
- **RL training**: `~/areal_experiments/logs/ubuntu/frontierco-grpo/{trial_name}/generated/`
- **RL evaluation**: `~/areal_experiments/logs/ubuntu/frontierco-grpo/{trial_name}/generated-eval/`
- **Checkpoints**: `~/areal_experiments/checkpoints/ubuntu/frontierco-*/{trial_name}/default/`

## Configuration Files

### `frontierco_sft_megatron.yaml`
- Model: Qwen/Qwen3-8B (can be changed to other models)
- Dataset path: `frontierco`
- Max length: 8192 tokens
- Optimizer: Adam with cosine LR schedule

### `frontierco_grpo_megatron.yaml`
- Initializes from SFT checkpoint
- Multiple samples per prompt (n_samples=8)
- Reward-based policy optimization
- KL divergence control (disabled by default)

### `frontierco_eval_config.yaml`
- Optimized for 8-GPU evaluation
- Uses all GPUs for maximum throughput
- Smaller batch size (16) for memory efficiency

## Architecture Overview

```
examples/frontierco/
├── reward.py                          # Code execution & reward computation
├── frontierco_sft_megatron.py         # SFT training script
├── frontierco_sft_megatron.yaml       # SFT configuration
├── frontierco_grpo_megatron.py        # GRPO training script
├── frontierco_grpo_megatron.yaml      # GRPO configuration
├── frontierco_eval.py                 # Evaluation script
├── frontierco_eval_config.yaml        # Evaluation configuration
├── PLAN.md                            # Detailed implementation plan
└── README.md                          # This file

areal/dataset/
└── frontierco.py                      # Dataset loader for SFT and RL
```

## Key Metrics

### Training Metrics:
- **Loss**: Cross-entropy loss (SFT) or policy gradient loss (GRPO)
- **Perplexity**: Language modeling quality
- **Learning rate**: Tracks optimizer schedule
- **Throughput**: Tokens/samples per second

### Evaluation Metrics:
- **pass@k**: Percentage of problems solved with ≥1 correct solution in k samples
- **Reward**: Binary pass/fail per test case
- **Execution success rate**: Percentage of samples that compile and run
- **Average tokens**: Length of generated solutions

## Tips & Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Enable `gradient_checkpointing: true`
- Reduce `max_length` or `max_new_tokens`

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Increase `max_concurrent_rollouts` for RL
- Verify allocation mode uses all GPUs

### Low Code Quality
- Verify test cases are correct
- Check reward function is working (print rewards)
- Try different `n_samples` and `temperature`
- Increase training epochs

### Test Execution Failures
- Ensure test format matches reward function expectations
- Validate test cases can execute independently
- Check for timeout issues (increase timeout in reward.py)

## References

- **PLAN.md**: Comprehensive implementation plan and design decisions
- **GSM8K Examples**: Similar structure in `examples/math/`
- **AReaL Documentation**: https://github.com/inclusionAI/AReaL

## Next Steps

See `PLAN.md` for:
- Detailed architecture explanation
- Implementation phases
- Dataset integration guide
- Swerex integration instructions
- Advanced configuration options
