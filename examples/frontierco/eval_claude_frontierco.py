#!/usr/bin/env python3
"""
Evaluate Claude models on FrontierCO test problems using CO-Bench framework.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add CO-Bench to path
sys.path.insert(0, str(Path.home() / "CO-Bench"))

from evaluation import YieldingEvaluator, get_new_data


def generate_code_with_claude(problem_description, model="claude-sonnet-4-20250514", temperature=0.8, max_tokens=4096):
    """Generate code using Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Format prompt
    prompt = (
        f"You are an expert in Operations Research and combinatorial optimization. "
        f"Solve the following problem by implementing the required solve function.\n\n"
        f"{problem_description}\n\n"
        f"Important requirements:\n"
        f"1. Your solve function MUST use Python's yield keyword to progressively return better solutions\n"
        f"2. The function should continuously improve solutions over time\n"
        f"3. Enclose all code within ```python ... ``` markers\n"
        f"4. You may use standard Python packages (numpy, scipy, etc.) but avoid commercial solvers like Gurobi\n"
        f"5. The solver will run with a timeout - focus on finding good solutions quickly\n"
        f"6. Use practical heuristics like nearest neighbor, 2-opt, etc. NOT brute force\n\n"
        f"Generate the complete implementation:"
    )

    print(f"Calling Claude API ({model})...")

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = response.content[0].text
    return generated_text


def extract_code_blocks(text):
    """Extract Python code from markdown code blocks."""
    import re

    # Try to find ```python blocks
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return "\n\n".join(matches)

    # Try generic ``` blocks
    pattern = r"```\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return "\n\n".join(matches)

    # Return full text if no blocks found
    return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate Claude on FrontierCO")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["TSP"],
        help="Tasks to evaluate (default: TSP)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/AReaL/data/frontierco",
        help="Directory containing frontierco data"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for solver execution (seconds)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./frontierco_eval_claude",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Evaluating Claude {args.model} on FrontierCO")
    print(f"Tasks: {args.tasks}")
    print(f"Timeout: {args.timeout}s")

    all_results = {}

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        try:
            # Load task data
            data = get_new_data(task, src_dir=args.data_dir, data_dir=args.data_dir)

            # Generate code
            generated_text = generate_code_with_claude(
                data.problem_description,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            # Save raw response
            response_file = os.path.join(args.output_dir, f"{task}_response.txt")
            with open(response_file, "w") as f:
                f.write(generated_text)
            print(f"Raw response saved to: {response_file}")

            # Extract code
            code = extract_code_blocks(generated_text)

            # Save generated code
            code_file = os.path.join(args.output_dir, f"{task}_generated.py")
            with open(code_file, "w") as f:
                f.write(code)
            print(f"Generated code saved to: {code_file}")

            # Evaluate
            print(f"Evaluating with timeout={args.timeout}s...")
            evaluator = YieldingEvaluator(data, timeout=args.timeout)
            feedback = evaluator.evaluate(code)

            # Store results
            all_results[task] = {
                "test_score": feedback.test_score,
                "test_feedback": feedback.test_feedback,
                "dev_score": feedback.dev_score,
                "dev_feedback": feedback.dev_feedback,
            }

            print(f"\nResults for {task}:")
            print(f"  Dev Score: {feedback.dev_score}")
            print(f"  Test Score: {feedback.test_score}")
            print(f"  Test Feedback: {feedback.test_feedback}")

        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {"error": str(e)}

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    for task, result in all_results.items():
        if "error" in result:
            print(f"  {task}: ERROR - {result['error']}")
        else:
            print(f"  {task}: Test Score = {result['test_score']}")


if __name__ == "__main__":
    main()
