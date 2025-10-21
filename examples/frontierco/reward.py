"""
Reward function for code generation tasks using swerex for execution.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional


def extract_code_block(message: str) -> dict | None:
    """Extract Python code block from markdown format."""
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, message, re.DOTALL)

    if matches:
        code = matches[0].strip()

        # Extract libraries from LIBRARIES comment if present
        libraries = []
        lines = code.split("\n")
        for line in lines:
            if line.strip().startswith("# LIBRARIES:"):
                lib_str = line.replace("# LIBRARIES:", "").strip()
                libraries = [lib.strip() for lib in lib_str.split(",")]
                break

        return {"code": code, "libraries": libraries}
    return None


async def run_code_with_tests(
    deployment,
    code: str,
    test_cases: list,
    entry_point: str = "",
    libraries: list = None,
) -> tuple[bool, str]:
    """
    Execute code with test cases in swerex deployment.

    Args:
        deployment: AbstractDeployment instance
        code: Python code to execute
        test_cases: List of test case dicts with test_code
        entry_point: Function name to test
        libraries: Optional list of pip packages to install

    Returns:
        (success, output): Boolean indicating if all tests passed, and output string
    """
    from swerex.runtime.abstract import BashAction, CreateBashSessionRequest

    await deployment.start()
    runtime = deployment.runtime

    # Create bash session
    await runtime.create_session(CreateBashSessionRequest())

    # Install required libraries
    if libraries:
        for lib in libraries:
            result = await runtime.run_in_session(BashAction(command=f"pip install {lib}"))
            if result.exit_code != 0:
                return False, f"Failed to install {lib}: {result.stderr}"

    # For HumanEval format: code is a function definition, tests call that function
    # Write the solution code
    write_code_cmd = f"cat > /tmp/solution.py << 'EOF'\n{code}\nEOF"
    result = await runtime.run_in_session(BashAction(command=write_code_cmd))
    if result.exit_code != 0:
        return False, f"Error writing code: {result.stderr}"

    # Run each test case
    passed = 0
    for i, test_case in enumerate(test_cases):
        # Get test code
        test_code = test_case.get("test_code", "")

        if not test_code:
            continue

        # Create a test script that imports solution and runs tests
        test_script = f"""
{code}

{test_code}

# Run the test
check({entry_point})
"""

        # Write test script
        write_test_cmd = f"cat > /tmp/test_{i}.py << 'EOF'\n{test_script}\nEOF"
        result = await runtime.run_in_session(BashAction(command=write_test_cmd))
        if result.exit_code != 0:
            return False, f"Error writing test {i}: {result.stderr}"

        # Execute test
        exec_cmd = f"python /tmp/test_{i}.py"
        result = await runtime.run_in_session(BashAction(command=exec_cmd))

        if result.exit_code == 0:
            passed += 1
        else:
            # Test failed
            return False, f"Test {i} failed: {result.stderr}"

    all_passed = passed == len(test_cases)
    return all_passed, f"Passed {passed}/{len(test_cases)} tests"


def frontierco_reward_fn(
    prompt,
    completions,
    prompt_ids,
    completion_ids,
    test_cases,
    entry_point="",
    deployment=None,
    **kwargs,
):
    """
    Reward function using swerex for code execution.

    This function is called synchronously from the RL training loop,
    but wraps async swerex calls.

    Args:
        prompt: The problem prompt
        completions: Generated code solution
        prompt_ids: Tokenized prompt
        completion_ids: Tokenized completion
        test_cases: List of test case dicts
        entry_point: Function name being tested
        deployment: Swerex deployment instance
        **kwargs: Additional arguments

    Returns:
        float: Binary reward (1.0 if all tests pass, 0.0 otherwise)
    """
    # Extract code from completion
    code_block = extract_code_block(completions)

    if not code_block:
        # Try to extract code without markers (raw code)
        code_block = {"code": completions.strip(), "libraries": []}

    code = code_block.get("code", "")
    if not code:
        return 0.0

    # Handle deployment
    if deployment is None:
        # Fall back to local execution without swerex
        return frontierco_reward_fn_fallback(
            code=code, test_cases=test_cases, entry_point=entry_point
        )

    try:
        # Run async code execution
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    run_code_with_tests(
                        deployment=deployment,
                        code=code,
                        test_cases=test_cases,
                        entry_point=entry_point,
                        libraries=code_block.get("libraries", []),
                    ),
                )
                success, output = future.result(timeout=30)  # 30 second timeout
        except RuntimeError:
            # No event loop running, can use asyncio.run directly
            success, output = asyncio.run(
                run_code_with_tests(
                    deployment=deployment,
                    code=code,
                    test_cases=test_cases,
                    entry_point=entry_point,
                    libraries=code_block.get("libraries", []),
                )
            )

        return 1.0 if success else 0.0

    except Exception as e:
        # Execution error
        print(f"[FrontierCo] Error executing code: {e}")
        return 0.0


def frontierco_reward_fn_fallback(code: str, test_cases: list, entry_point: str = ""):
    """
    Fallback reward function using local Python execution.
    Used when swerex deployment is not available.

    Args:
        code: Python code to execute
        test_cases: List of test case dicts
        entry_point: Function name being tested

    Returns:
        float: Binary reward (1.0 if all tests pass, 0.0 otherwise)
    """
    import sys
    from io import StringIO

    try:
        # Create namespace for execution
        namespace = {}

        # Execute the code to define the function
        exec(code, namespace)

        # Run each test
        passed = 0
        for test_case in test_cases:
            test_code = test_case.get("test_code", "")
            if not test_code:
                continue

            try:
                # Execute test in the same namespace
                exec(test_code, namespace)
                # Run the check function
                exec(f"check({entry_point})", namespace)
                passed += 1
            except AssertionError:
                # Test failed
                return 0.0
            except Exception as e:
                # Execution error
                return 0.0

        return 1.0 if passed == len(test_cases) else 0.0

    except Exception as e:
        return 0.0
