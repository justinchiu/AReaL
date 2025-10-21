# CoBench/FrontierCO Integration Guide

## Overview

This guide shows how to integrate **AReaL-trained models** with your existing **CoBench evaluation framework**.

**Pipeline**:
1. **Train** on OR-Instruct-Data-3K using AReaL (Megatron SFT/GRPO)
2. **Evaluate** on CoBench/FrontierCO using your evaluation framework

## Setup

### 1. Download CoBench Data

```bash
# Download CO-Bench dataset (637MB)
huggingface-cli download CO-Bench/CO-Bench --repo-type dataset --local-dir data/comb_opt

# Download FrontierCO dataset
huggingface-cli download CO-Bench/FrontierCO --repo-type dataset --local-dir data/frontierco
```

### 2. Install AReaL Agent in Your CoBench Framework

Copy the AReaL agent to your CoBench repo:

```bash
# From AReaL repo
cp examples/frontierco/areal_cobench_agent.py /path/to/your/cobench/repo/evals/agents/

# Or create a symlink
ln -s $(pwd)/examples/frontierco/areal_cobench_agent.py /path/to/your/cobench/repo/evals/agents/
```

Register the agent in your framework's `evals/agents/__init__.py`:

```python
from evals.agents.areal_cobench_agent import AReaLAgent, AReaLAgentConfig

# Add to your agent registry
__all__ = [
    ...
    "AReaLAgent",
    "AReaLAgentConfig",
]
```

The agent implementation is already complete in `areal_cobench_agent.py`. It:

- ✅ Extends your `Agent` base class
- ✅ Implements `async run_problem(problem: Problem) -> Solution` interface
- ✅ Formats prompts to match OR-Instruct training
- ✅ Generates code with AReaL model
- ✅ Executes code and extracts solutions
- ✅ Returns `Solution` objects with output dict compatible with CoBenchTask scoring
- ✅ Output dict format: `{"solution": result, "code": code}` - merges cleanly with case_data

**Key features**:
- Automatic prompt formatting matching OR-Instruct style
- Configurable generation parameters (temperature, max_tokens, etc.)
- Handles generator functions (for anytime algorithms)
- Robust error handling and fallbacks
- Extracts code from markdown blocks

**How it works with CoBenchTask evaluation**:
1. CoBenchTask calls `agent.run_problem(problem)` where problem contains case_data
2. Agent generates code and executes it with case_data to get solution
3. Agent returns `Solution(output={"solution": result, "code": code})`
4. CoBenchTask merges: `{**case_data, **output}` → `{**case_data, "solution": result, "code": code}`
5. CoBenchTask calls `eval_func(**merged_dict)` to compute score
6. Score is returned as `{"raw_score": float}` or `{"raw_score": float, "normalized_score": float}`

## Running Evaluation

### Option 1: Direct Evaluation

After training with AReaL, evaluate directly:

```bash
# 1. Train with AReaL
python3 -m areal.launcher.local examples/frontierco/frontierco_sft_megatron.py \
    --config examples/frontierco/frontierco_sft_megatron.yaml \
    model.path=Qwen/Qwen3-8B \
    trial_name=qwen3-8b

# 2. Copy agent to your CoBench repo
cp examples/frontierco/areal_cobench_agent.py /path/to/cobench/evals/agents/

# 3. Evaluate with CoBench
cd /path/to/your/cobench/repo
python -m evals.tasks.cobench \
    agent=evals.agents.areal_cobench_agent.AReaLAgentConfig \
    agent.model_path=/path/to/areal/checkpoint \
    task_names='["Job shop scheduling", "Aircraft landing"]' \
    max_files_per_task=3 \
    output_path=areal_cobench_results.jsonl
```

### Option 2: Batch Evaluation on All Tasks

```bash
# Evaluate on multiple CoBench tasks
python -m evals.tasks.cobench \
    agent=evals.agents.areal_cobench_agent.AReaLAgentConfig \
    agent.model_path=/path/to/areal/checkpoint \
    agent.temperature=0.8 \
    agent.max_new_tokens=2048 \
    task_names='[
        "Job shop scheduling",
        "Aircraft landing",
        "Assignment problem",
        "Bin packing - one-dimensional",
        "Vehicle routing problem",
        "Traveling salesman problem",
        "Knapsack problem",
        "Graph coloring"
    ]' \
    max_files_per_task=5 \
    max_problems_per_task=2 \
    output_path=areal_full_results.jsonl
```

### Option 3: Compare Models

Compare base model, SFT, and RL checkpoints:

```bash
# Base model
python -m evals.tasks.cobench \
    agent=evals.agents.areal_cobench_agent.AReaLAgentConfig \
    agent.model_path=Qwen/Qwen3-8B \
    task_names='["Job shop scheduling"]' \
    output_path=base_results.jsonl

# SFT checkpoint
python -m evals.tasks.cobench \
    agent=evals.agents.areal_cobench_agent.AReaLAgentConfig \
    agent.model_path=/path/to/sft/checkpoint \
    task_names='["Job shop scheduling"]' \
    output_path=sft_results.jsonl

# RL checkpoint
python -m evals.tasks.cobench \
    agent=evals.agents.areal_cobench_agent.AReaLAgentConfig \
    agent.model_path=/path/to/rl/checkpoint \
    task_names='["Job shop scheduling"]' \
    output_path=rl_results.jsonl
```

## Checkpoint Path

AReaL saves checkpoints to:
```
~/areal_experiments/checkpoints/ubuntu/frontierco-sft-megatron/{trial_name}/default/epoch*
```

Example:
```bash
CHECKPOINT=/home/ubuntu/areal_experiments/checkpoints/ubuntu/frontierco-sft-megatron/qwen3-8b/default/epoch0epochstep100globalstep100

python -m evals.tasks.cobench \
    agent.agent_name=areal_agent \
    agent.agent_params.model_path=$CHECKPOINT \
    output_path=results.jsonl
```

## Visualizing Results

Use your existing visualizer:

```bash
streamlit run cobench_visualizer.py
```

This will show:
- Success rates per task
- Generated code (syntax highlighted)
- Solution quality metrics
- Comparison across checkpoints

## Integration Script Template

Create `evaluate_areal_on_cobench.py`:

```python
#!/usr/bin/env python3
"""
Evaluate AReaL-trained model on CoBench/FrontierCO.
"""
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(experiment_name, trial_name):
    """Find latest AReaL checkpoint."""
    base_path = Path.home() / "areal_experiments/checkpoints/ubuntu"
    checkpoint_dir = base_path / experiment_name / trial_name / "default"

    if not checkpoint_dir.exists():
        raise ValueError(f"No checkpoints found at {checkpoint_dir}")

    # Find latest epoch checkpoint
    checkpoints = list(checkpoint_dir.glob("epoch*"))
    if not checkpoints:
        raise ValueError(f"No epoch checkpoints in {checkpoint_dir}")

    latest = sorted(checkpoints)[-1]
    return str(latest)


def run_cobench_eval(checkpoint_path, output_path, task_names=None):
    """Run CoBench evaluation."""
    if task_names is None:
        task_names = [
            "Job shop scheduling",
            "Aircraft landing",
            "Bin packing - one-dimensional",
        ]

    cmd = [
        "python", "-m", "evals.tasks.cobench",
        f"agent.agent_name=areal_agent",
        f"agent.agent_params.model_path={checkpoint_path}",
        f"agent.agent_params.temperature=0.8",
        f"agent.agent_params.max_new_tokens=2048",
        f"task_names={task_names}",
        f"max_files_per_task=3",
        f"max_problems_per_task=2",
        f"output_path={output_path}",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Find latest checkpoint
    checkpoint = find_latest_checkpoint(
        experiment_name="frontierco-sft-megatron",
        trial_name="qwen3-8b",
    )

    print(f"Found checkpoint: {checkpoint}")

    # Run evaluation
    run_cobench_eval(
        checkpoint_path=checkpoint,
        output_path="areal_cobench_results.jsonl",
    )

    print("Evaluation complete! View results with:")
    print("  streamlit run cobench_visualizer.py")
```

## Expected Results Format

Your CoBench framework outputs JSONL with:
```json
{
  "task": "Job shop scheduling",
  "file": "ft06.txt",
  "problem_id": 0,
  "success": true,
  "solution_quality": 55,
  "optimal": 55,
  "gap": 0.0,
  "generated_code": "def solve(...)...",
  "execution_time": 12.3
}
```

## Performance Tracking

Track improvements across training stages:

| Model | Task | Success Rate | Avg Gap | Avg Time |
|-------|------|--------------|---------|----------|
| Base (Qwen3-8B) | Job shop | TBD | TBD | TBD |
| + SFT (OR-Instruct) | Job shop | TBD | TBD | TBD |
| + RL (GRPO) | Job shop | TBD | TBD | TBD |

## Notes

### Prompt Format Alignment
- **AReaL training** uses OR-Instruct format: "Below is an operations research question..."
- **Agent prompt** should match this format for best results
- See `_format_problem_prompt()` in agent implementation

### Checkpoint Compatibility
- AReaL saves in standard HuggingFace format (compatible with `transformers`)
- Megatron checkpoints are automatically converted
- Use the checkpoint path directly, no conversion needed

### Optimization for CoBench
- **Temperature**: 0.8-1.0 for diverse solutions
- **Max tokens**: 2048+ for complete optimization code
- **Sampling**: Enable for variety (anytime algorithms benefit from multiple attempts)

## Troubleshooting

### Issue: "No module named 'areal'"
**Solution**: Make sure you're running from your CoBench repo, not AReaL repo.

### Issue: Checkpoint not found
**Solution**: Check the checkpoint path format. AReaL uses `epoch{N}epochstep{M}globalstep{K}` naming.

### Issue: Out of memory
**Solution**: Reduce batch size or use smaller model variant (Qwen3-1.7B instead of 8B).

### Issue: Generated code doesn't match CoBench format
**Solution**: Verify prompt format matches OR-Instruct training data. Check `_format_problem_prompt()`.

## Next Steps

1. ✅ Train on OR-Instruct-Data-3K with AReaL
2. ⬜ Implement `COBenchAReaLAgent` in your evaluation framework
3. ⬜ Run evaluation on CoBench tasks
4. ⬜ Analyze results with visualizer
5. ⬜ Iterate: RL training with quality-based rewards
6. ⬜ Final evaluation on FrontierCO benchmark

## Resources

- **AReaL Training**: See `examples/frontierco/README.md`
- **Your CoBench Framework**: README in your repo
- **Original CO-Bench**: https://github.com/sunnweiwei/CO-Bench
