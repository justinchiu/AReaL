#!/usr/bin/env python3
"""
Run training for multiple Qwen3 model sizes with separate logging.
"""
import subprocess
import sys
from pathlib import Path

# Configuration
STORAGE_PATH = "/home/ubuntu/areal_experiments"
CONFIG_FILE = "examples/math/gsm8k_sft_megatron.yaml"
SCRIPT_FILE = "examples/math/gsm8k_sft_megatron.py"

# Model configurations
MODELS = [
    {"path": "Qwen/Qwen3-1.7B", "trial_name": "qwen3-1.7b"},
    {"path": "Qwen/Qwen3-8B", "trial_name": "qwen3-8b"},
    {"path": "Qwen/Qwen3-14B", "trial_name": "qwen3-14b"},
    {"path": "Qwen/Qwen3-30B-A3B", "trial_name": "qwen3-30b-a3b"},
    {"path": "Qwen/Qwen3-32B", "trial_name": "qwen3-32b"},
]

# Optional: Model-specific hyperparameters
# Uncomment and modify if you need different settings per model
MODEL_SPECIFIC_PARAMS = {
    "Qwen/Qwen3-1.7B": {},
    "Qwen/Qwen3-8B": {},
    "Qwen/Qwen3-14B": {},
    "Qwen/Qwen3-30B-A3B": {},
    "Qwen/Qwen3-32B": {},
}


def download_model(model_path):
    """Download model if not already cached."""
    print(f"Downloading {model_path}...")
    cmd = [
        "python",
        "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{model_path}', "
        f"allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model'])",
    ]
    subprocess.run(cmd, check=True)


def run_training(model_config):
    """Run training for a single model."""
    model_path = model_config["path"]
    trial_name = model_config["trial_name"]

    print("\n" + "=" * 60)
    print(f"Training {model_path} (trial: {trial_name})")
    print("=" * 60 + "\n")

    # Build command
    cmd = [
        "python3",
        "-m",
        "areal.launcher.local",
        SCRIPT_FILE,
        "--config",
        CONFIG_FILE,
        f"model.path={model_path}",
        f"trial_name={trial_name}",
        f"cluster.fileroot={STORAGE_PATH}",
    ]

    # Add model-specific parameters
    if model_path in MODEL_SPECIFIC_PARAMS:
        for key, value in MODEL_SPECIFIC_PARAMS[model_path].items():
            cmd.append(f"{key}={value}")

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully completed training for {model_path}")
        return {"training": True, "evaluation": None}
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed for {model_path}: {e}")
        return {"training": False, "evaluation": None}


def main():
    # Create storage directory
    Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)

    # Optional: Download all models first
    print("Downloading all models...")
    for model_config in MODELS:
        try:
            download_model(model_config["path"])
        except Exception as e:
            print(f"Warning: Failed to download {model_config['path']}: {e}")

    # Run training for each model
    results = {}
    for model_config in MODELS:
        result = run_training(model_config)
        results[model_config["path"]] = result

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for model_path, result in results.items():
        status = "✓ SUCCESS" if result["training"] else "✗ FAILED"
        print(f"{status}: {model_path}")

    print(f"\nResults saved to: {STORAGE_PATH}/gsm8k-sft-megatron/")
    print(f"\nTo evaluate the trained models, run:")
    print(f"  ./eval_all_qwen3_sizes.py")

    # Exit with error code if any training failed
    if not all(r["training"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
