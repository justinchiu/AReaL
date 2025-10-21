# FrontierCO Evaluation

## Overview

**Training**: CardinalOperations/OR-Instruct-Data-3K (3,000 OR problems with `coptpy` code)
**Evaluation**: CO-Bench/FrontierCO (challenging combinatorial optimization benchmark)

FrontierCO is a benchmark of real-world, challenging combinatorial optimization problems where solvers must **yield progressively improved solutions over time** (anytime algorithms).

## Setup

### 1. Download FrontierCO Dataset

```bash
# Install huggingface CLI if needed
pip install huggingface_hub

# Download FrontierCO dataset
huggingface-cli download CO-Bench/FrontierCO --repo-type dataset --local-dir ~/frontierco_data
```

Or in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download("CO-Bench/FrontierCO", repo_type="dataset", local_dir="~/frontierco_data")
```

### 2. Install CO-Bench Evaluation Framework

```bash
git clone https://github.com/sunnweiwei/CO-Bench.git
cd CO-Bench
pip install -e .
```

## Evaluation Approach

### Key Differences from Standard Code Evaluation

| Aspect | Standard Code Eval | FrontierCO Eval |
|--------|-------------------|-----------------|
| **Output** | Single solution | Generator yielding improved solutions |
| **Metric** | Pass/Fail (correctness) | Solution quality (optimality) |
| **Timeout** | Hard limit (fails after) | Anytime (last yielded before timeout) |
| **Problems** | Code completion | Combinatorial optimization |

### Evaluation Process

1. **Generate Code**: Model generates a `solve()` function
2. **Execute with Timeout**: Run with configurable timeout (e.g., 300s dev, 3600s final)
3. **Collect Solutions**: Capture all yielded solutions
4. **Measure Quality**: Use problem-specific metrics (e.g., tour length for TSP)
5. **Report Best**: Return the last/best solution before timeout

## Integration with AReaL

### Option 1: External Evaluation (Recommended)

Train with AReaL, then evaluate separately using CO-Bench's evaluator:

```bash
# 1. Train with AReaL
python3 -m areal.launcher.local examples/frontierco/frontierco_sft_megatron.py \
    --config examples/frontierco/frontierco_sft_megatron.yaml

# 2. Generate predictions on FrontierCO
python3 examples/frontierco/generate_frontierco_predictions.py \
    --checkpoint /path/to/checkpoint \
    --output predictions.jsonl

# 3. Evaluate with CO-Bench
cd CO-Bench
python evaluate.py \
    --predictions ../predictions.jsonl \
    --dataset FrontierCO \
    --timeout 3600
```

### Option 2: Custom AReaL Reward Function

For RL training with FrontierCO-style rewards:

```python
def frontierco_optimization_reward(prompt, completions, prompt_ids, completion_ids, **kwargs):
    """
    Reward based on solution quality for optimization problems.

    Returns:
        float: Normalized quality score (higher is better)
    """
    # 1. Extract generated code
    code = extract_code_block(completions)

    # 2. Execute with timeout and collect yielded solutions
    solutions = execute_with_yield(code, problem_instance, timeout=300)

    # 3. Evaluate best solution quality
    if solutions:
        best_solution = solutions[-1]  # Last yielded
        quality = evaluate_solution_quality(best_solution, problem_instance)
        # Normalize to [0, 1] range
        reward = normalize_quality(quality, problem_type)
        return reward
    else:
        return 0.0  # No valid solution
```

## Example FrontierCO Problem

```python
# Problem: Traveling Salesman Problem (TSP)
def solve(node_coords: List[Tuple[float, float]]) -> Generator[List[int], None, None]:
    """
    Generate progressively better TSP tours.

    Args:
        node_coords: List of (x, y) coordinates for each node

    Yields:
        List[int]: Tour as list of node indices (0-indexed)
    """
    n = len(node_coords)

    # Initial greedy solution
    tour = greedy_tsp(node_coords)
    yield tour

    # Improve with local search
    for iteration in range(1000):
        improved_tour = two_opt_improve(tour, node_coords)
        if tour_length(improved_tour) < tour_length(tour):
            tour = improved_tour
            yield tour  # Yield each improvement
```

## Metrics

FrontierCO uses problem-specific quality metrics:

- **TSP**: Tour length (minimize)
- **Knapsack**: Total value (maximize)
- **Scheduling**: Makespan or total time (minimize)
- **Routing**: Total distance/cost (minimize)

## Evaluation Script Template

See `generate_frontierco_predictions.py` for a complete implementation that:
1. Loads trained AReaL checkpoint
2. Generates code for each FrontierCO problem
3. Executes with timeout
4. Saves predictions in CO-Bench format

## Resources

- **FrontierCO Dataset**: https://huggingface.co/datasets/CO-Bench/FrontierCO
- **CO-Bench Repository**: https://github.com/sunnweiwei/CO-Bench
- **Paper/Documentation**: Check CO-Bench repo for evaluation details

## Training Strategy

For best results on FrontierCO:

1. **SFT on OR-Instruct-Data-3K**: Learn basic OR modeling + code generation
2. **Fine-tune on FrontierCO**: Adapt to specific problem formats (if training data available)
3. **RL with Quality Rewards**: Use solution quality as reward signal
4. **Prompt Engineering**: Guide model to generate generator-style code

## Next Steps

1. ✅ Train SFT model on OR-Instruct-Data-3K (already configured)
2. ⬜ Download FrontierCO dataset
3. ⬜ Install CO-Bench evaluation framework
4. ⬜ Create `generate_frontierco_predictions.py` script
5. ⬜ Run evaluation on FrontierCO benchmark
6. ⬜ (Optional) Implement RL training with quality-based rewards
