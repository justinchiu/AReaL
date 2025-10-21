"""
AReaL Agent for CoBench evaluation.

This agent wraps AReaL-trained models for evaluation on combinatorial optimization problems.
"""
from __future__ import annotations

import ast
import re
from typing import Any

import torch
from chz import chz, field
from transformers import AutoModelForCausalLM, AutoTokenizer

from evals.agents.agent import Agent, AgentConfig
from evals.models import Problem, Solution


@chz
class AReaLAgentConfig(AgentConfig):
    """Configuration for AReaL agent."""

    agent_name: str = "areal_agent"

    # Model configuration
    model_path: str = field(
        default="Qwen/Qwen3-8B"
    )  # Path to AReaL checkpoint or HuggingFace model
    device: str = "cuda"  # Device to use (cuda/cpu)
    dtype: str = "bfloat16"  # Model dtype (bfloat16/float16/float32)

    # Generation parameters
    temperature: float = 0.8  # Sampling temperature
    max_new_tokens: int = 2048  # Maximum tokens to generate
    do_sample: bool = True  # Whether to use sampling
    top_p: float | None = None  # Nucleus sampling parameter
    top_k: int | None = None  # Top-k sampling parameter

    # Prompt formatting
    include_solve_signature: bool = True  # Include function signature in prompt
    include_solve_source: bool = True  # Include reference solve function
    include_task_description: bool = True  # Include task description


class AReaLAgent(Agent):
    """Agent that uses AReaL-trained models to solve CoBench problems."""

    config_cls = AReaLAgentConfig

    def __init__(self, config: AReaLAgentConfig):
        super().__init__(config)
        self.config = config

        print(f"Loading model from {config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        # Set dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(config.dtype, torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=dtype,
            device_map=config.device,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully!")

    async def run_problem(self, problem: Problem) -> Solution:
        """
        Solve a single CoBench problem.

        Args:
            problem: Problem containing task information and case data

        Returns:
            Solution: Generated code and extracted solution
        """
        # Format prompt
        prompt = self._format_problem_prompt(problem)

        # Generate solution
        generated_code = await self._generate_code(prompt)

        # Extract solution from generated code
        solution_data = self._extract_solution(generated_code, problem)

        return Solution(
            problem_id=problem.problem_id,
            output=solution_data,
            metadata={
                "generated_code": generated_code,
                "prompt_length": len(prompt),
            },
        )

    def _format_problem_prompt(self, problem: Problem) -> str:
        """
        Format problem into prompt matching OR-Instruct training format.

        Args:
            problem: Problem to format

        Returns:
            str: Formatted prompt
        """
        data = problem.data

        # Extract problem information
        task_name = data.get("task_name", "Unknown task")
        task_description = data.get("task_description", "")
        case_data = data.get("case_data", {})
        solve_signature = data.get("solve_signature", "")
        solve_source = data.get("solve_function_source", "")

        # Build prompt components
        prompt_parts = [
            "Below is an operations research question. "
            "Build a mathematical model and corresponding Python code that appropriately addresses the question.",
            "",
            f"# Task: {task_name}",
        ]

        if self.config.include_task_description and task_description:
            prompt_parts.extend(["", "# Task Description:", task_description])

        # Add problem instance data
        prompt_parts.extend(
            ["", "# Problem Instance:", self._format_case_data(case_data)]
        )

        # Add solve function information
        if self.config.include_solve_signature and solve_signature:
            prompt_parts.extend(["", "# Expected Function Signature:", solve_signature])

        if self.config.include_solve_source and solve_source:
            prompt_parts.extend(
                ["", "# Reference Implementation:", f"```python\n{solve_source}\n```"]
            )

        # Add instructions
        prompt_parts.extend([
            "",
            "# Instructions:",
            "1. Provide a complete Python `solve` function that solves this problem",
            "2. The function should match the expected signature",
            "3. Include necessary imports and helper functions",
            "4. For optimization problems, yield progressively better solutions over time",
            "5. Wrap your code in ```python``` markers",
        ])

        return "\n".join(prompt_parts)

    def _format_case_data(self, case_data: dict[str, Any]) -> str:
        """Format case data for inclusion in prompt."""
        lines = []
        for key, value in case_data.items():
            # Format value nicely
            if isinstance(value, (list, dict)):
                value_str = str(value)
                # Truncate if too long
                if len(value_str) > 500:
                    value_str = value_str[:500] + "..."
            else:
                value_str = str(value)
            lines.append(f"- {key}: {value_str}")
        return "\n".join(lines)

    async def _generate_code(self, prompt: str) -> str:
        """
        Generate code using the model.

        Args:
            prompt: Formatted prompt

        Returns:
            str: Generated code
        """
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return generated

    def _extract_solution(self, generated_code: str, problem: Problem) -> dict[str, Any]:
        """
        Extract solution from generated code by executing it.

        Args:
            generated_code: Generated code containing solve function
            problem: Original problem

        Returns:
            dict: Solution data ready for evaluation
        """
        # Extract Python code from markdown blocks
        code = self._extract_code_blocks(generated_code)

        if not code:
            # If no code blocks, try to use the whole response
            code = generated_code

        # Try to execute the code and extract solution
        try:
            # Get case data
            case_data = problem.data.get("case_data", {})

            # Create execution namespace
            namespace = {}

            # Execute the generated code to define functions
            exec(code, namespace)

            # Check if solve function exists
            if "solve" not in namespace:
                return {
                    "error": "No solve function found in generated code",
                    "code": code,
                }

            solve_func = namespace["solve"]

            # Call solve function with case data
            # Try different parameter passing strategies
            try:
                # Strategy 1: Unpack case_data as kwargs
                result = solve_func(**case_data)
            except TypeError:
                try:
                    # Strategy 2: Pass individual arguments based on signature
                    import inspect
                    sig = inspect.signature(solve_func)
                    params = {k: case_data.get(k) for k in sig.parameters.keys() if k in case_data}
                    result = solve_func(**params)
                except Exception:
                    # Strategy 3: Pass case_data as single argument
                    result = solve_func(case_data)

            # If result is a generator, collect all yielded values
            if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict, list)):
                try:
                    results = list(result)
                    # Use last yielded result (best solution)
                    if results:
                        result = results[-1]
                except Exception:
                    pass

            return {
                "solution": result,
                "code": code,
            }

        except Exception as e:
            return {
                "error": f"Execution failed: {str(e)}",
                "code": code,
            }

    def _extract_code_blocks(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        # Try to find ```python blocks
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Return all code blocks concatenated
            return "\n\n".join(matches)

        # Try generic ``` blocks
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return "\n\n".join(matches)

        return text  # Return full text if no blocks found
