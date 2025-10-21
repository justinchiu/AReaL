#!/usr/bin/env python3
"""
Run evaluation for all trained Qwen3 model sizes.
"""
import subprocess
import sys
from pathlib import Path

# Configuration
STORAGE_PATH = "/home/ubuntu/areal_experiments"
CHECKPOINT_BASE = f"{STORAGE_PATH}/checkpoints/ubuntu"
EXPERIMENT_NAME = "gsm8k-sft-megatron"
EVAL_SCRIPT = "examples/math/gsm8k_eval.py"
EVAL_CONFIG = "examples/math/gsm8k_eval_config.yaml"  # Use 8-GPU eval config

# Model configurations
MODELS = [
    {"path": "Qwen/Qwen3-1.7B", "trial_name": "qwen3-1.7b"},
    {"path": "Qwen/Qwen3-8B", "trial_name": "qwen3-8b"},
    {"path": "Qwen/Qwen3-14B", "trial_name": "qwen3-14b"},
    {"path": "Qwen/Qwen3-30B-A3B", "trial_name": "qwen3-30b-a3b"},
    {"path": "Qwen/Qwen3-32B", "trial_name": "qwen3-32b"},
]


def find_checkpoint_path(trial_name):
    """Find the checkpoint path for a trained model."""
    # The actual structure is: checkpoints/ubuntu/gsm8k-sft-megatron/{trial_name}/default/epoch*
    trial_path = Path(CHECKPOINT_BASE) / EXPERIMENT_NAME / trial_name / "default"

    if not trial_path.exists():
        print(f"Warning: Trial path not found: {trial_path}")
        return None

    # Look for saved checkpoints - typically in format: epoch0epochstep58globalstep58
    checkpoint_dirs = list(trial_path.glob("epoch*"))

    if not checkpoint_dirs:
        print(f"Warning: No checkpoints found in {trial_path}")
        return None

    # Sort by name and get the latest
    checkpoint_dirs.sort()
    latest_checkpoint = checkpoint_dirs[-1]

    print(f"Found checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def run_single_eval(model_path_or_checkpoint, trial_name, experiment_name):
    """Run a single evaluation and return results."""
    import re

    # Build command
    cmd = [
        "python3",
        "-m",
        "areal.launcher.local",
        EVAL_SCRIPT,
        "--config",
        EVAL_CONFIG,
        f"actor.path={model_path_or_checkpoint}",
        f"valid_dataset.batch_size=16",
        f"gconfig.n_samples=1",
        f"gconfig.max_new_tokens=512",
        f"cluster.fileroot={STORAGE_PATH}",
        f"experiment_name={experiment_name}",
        f"trial_name={trial_name}",
    ]

    # Run evaluation - note: this may raise JobException but still complete successfully
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check the log file for results instead of relying on return code
        log_path = f"{STORAGE_PATH}/logs/ubuntu/{experiment_name}/{trial_name}/trainer.log"

        try:
            with open(log_path, 'r') as f:
                log_content = f.read()

            # Extract reward/accuracy from log
            reward_match = re.search(r'eval-rollout/reward\s+│\s+([\d.e+-]+)', log_content)

            if reward_match:
                reward = float(reward_match.group(1))
                accuracy_pct = reward * 100
                return {"success": True, "accuracy": accuracy_pct, "reward": reward}
            else:
                return {"success": False, "accuracy": None, "reward": None}

        except FileNotFoundError:
            return {"success": False, "accuracy": None, "reward": None}

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "accuracy": None, "reward": None}


def run_evaluation(model_config):
    """Run evaluation for both base and fine-tuned models."""
    trial_name = model_config["trial_name"]
    model_path = model_config["path"]

    print("\n" + "=" * 70)
    print(f"Evaluating {trial_name}")
    print("=" * 70 + "\n")

    results = {}

    # 1. Evaluate base model
    print(f"[1/2] Evaluating BASE model ({model_path})...")
    base_result = run_single_eval(
        model_path,
        f"{trial_name}-base",
        "gsm8k-base-eval"
    )

    if base_result["success"]:
        print(f"  ✓ Base model: {base_result['accuracy']:.2f}% accuracy")
    else:
        print(f"  ✗ Base model evaluation failed")
    results["base"] = base_result

    # 2. Evaluate fine-tuned model (if checkpoint exists)
    print(f"\n[2/2] Evaluating FINE-TUNED model...")
    checkpoint_path = find_checkpoint_path(trial_name)

    if checkpoint_path is None:
        print(f"  ⊘ No checkpoint found - skipping fine-tuned evaluation")
        results["finetuned"] = None
    else:
        finetuned_result = run_single_eval(
            checkpoint_path,
            trial_name,
            f"{EXPERIMENT_NAME}_eval"
        )

        if finetuned_result["success"]:
            print(f"  ✓ Fine-tuned model: {finetuned_result['accuracy']:.2f}% accuracy")

            # Calculate delta
            if base_result["success"]:
                delta = finetuned_result['accuracy'] - base_result['accuracy']
                print(f"  → Delta: {delta:+.2f}%")
        else:
            print(f"  ✗ Fine-tuned model evaluation failed")
        results["finetuned"] = finetuned_result

    return results


def main():
    # Note: We don't require training outputs to exist since we can evaluate base models
    print("=" * 80)
    print("GSM8K Evaluation - Base vs Fine-tuned Models")
    print("=" * 80)

    # Run evaluation for each model
    results = {}
    for model_config in MODELS:
        success = run_evaluation(model_config)
        results[model_config["trial_name"]] = success

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Base Accuracy':<15} {'Fine-tuned':<15} {'Delta':<15}")
    print("-" * 80)

    for trial_name, result in results.items():
        model_name = trial_name

        if result is None:
            print(f"{model_name:<20} {'N/A':<15} {'No checkpoint':<15} {'N/A':<15}")
        else:
            base_result = result.get("base", {})
            ft_result = result.get("finetuned")

            # Base accuracy
            if base_result.get("success"):
                base_acc = f"{base_result['accuracy']:.2f}%"
            else:
                base_acc = "Failed"

            # Fine-tuned accuracy
            if ft_result is None:
                ft_acc = "No checkpoint"
                delta = "N/A"
            elif ft_result.get("success"):
                ft_acc = f"{ft_result['accuracy']:.2f}%"

                # Calculate delta
                if base_result.get("success"):
                    delta_val = ft_result['accuracy'] - base_result['accuracy']
                    delta = f"{delta_val:+.2f}%"
                else:
                    delta = "N/A"
            else:
                ft_acc = "Failed"
                delta = "N/A"

            print(f"{model_name:<20} {base_acc:<15} {ft_acc:<15} {delta:<15}")

    print("=" * 80)
    print(f"\nBase model logs: {STORAGE_PATH}/logs/ubuntu/gsm8k-base-eval/")
    print(f"Fine-tuned logs: {STORAGE_PATH}/logs/ubuntu/{EXPERIMENT_NAME}_eval/")


if __name__ == "__main__":
    main()
