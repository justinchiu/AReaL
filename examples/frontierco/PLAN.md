# FrontierCo Setup Plan

## Overview
This directory contains training and evaluation scripts for code generation tasks using the AReaL framework. The setup follows the GSM8K example structure but adapted for code execution and verification.

## Architecture

### 1. Dataset Layer (`areal/dataset/frontierco.py`)
**Purpose**: Load and format code generation datasets for SFT and RL training

**Required Functions**:
- `get_frontierco_sft_dataset()` - Format dataset for supervised fine-tuning
  - Apply chat templates with user/assistant roles
  - Create loss masks (only train on code generation, not prompts)
  - Support multiple dataset formats (HumanEval, MBPP, custom)

- `get_frontierco_rl_dataset()` - Format dataset for RL training
  - Return prompt-only format for model to generate completions
  - Include test cases and metadata for verification
  - Support code execution backends (local or remote)

**Key Features**:
- Multi-language support (Python primary, extensible to others)
- Test case validation and formatting
- Configurable prompt templates
- Code extraction from markdown blocks

---

### 2. Reward Function (`examples/frontierco/reward.py`)
**Purpose**: Evaluate generated code correctness via test execution

**Implementation Options**:

#### Option A: Local Execution (Fast, Simpler)
```python
def frontierco_reward_fn(prompt, completions, prompt_ids, completion_ids, test_cases, **kwargs):
    from functioncall.code.local_verify import code_verify
    # Extract code from completion
    code = extract_code(completions)
    # Run test cases locally
    results = code_verify(code, test_cases, language="PYTHON")
    return 1 if all(results) else 0
```

#### Option B: Remote Execution (Scalable, Secure)
```python
def frontierco_reward_fn(prompt, completions, prompt_ids, completion_ids, test_cases, **kwargs):
    from functioncall.code.verify import code_verify
    # Extract code from completion
    code = extract_code(completions)
    # Submit to remote FaaS service
    results = code_verify(code, test_cases, language="PYTHON")
    return 1 if all(results) else 0
```

#### Option C: Hybrid with Partial Credit
```python
def frontierco_reward_fn_partial(prompt, completions, prompt_ids, completion_ids, test_cases, **kwargs):
    # Return percentage of passed tests (0.0 to 1.0)
    code = extract_code(completions)
    results = code_verify(code, test_cases, language="PYTHON")
    return sum(results) / len(results)
```

**Key Decisions**:
- Binary vs partial credit rewards
- Local vs remote execution
- Timeout configuration
- Error handling strategy
- Support for multiple test cases per problem

---

### 3. Training Scripts

#### 3.1 Supervised Fine-Tuning (`frontierco_sft.py`)
**Based on**: `examples/math/gsm8k_sft.py`

**Key Components**:
```python
- FSDPLMEngine for model training
- SFTConfig for configuration
- get_frontierco_sft_dataset() for data loading
- Standard supervised training loop
- Perplexity evaluation on validation set
```

**Training Flow**:
1. Load pretrained model (Qwen/DeepSeek-Coder/CodeLlama)
2. Apply chat templates to code + solution pairs
3. Train with cross-entropy loss on code generation
4. Evaluate perplexity on held-out set
5. Save checkpoints periodically

**Configuration** (`frontierco_sft.yaml`):
- Model path and dtype
- Batch size and learning rate
- Max sequence length (code can be long!)
- Gradient checkpointing
- FSDP/Megatron backend selection

---

#### 3.2 Reinforcement Learning (`frontierco_grpo.py`)
**Based on**: `examples/math/gsm8k_grpo.py`

**Key Components**:
```python
- FSDPPPOActor for policy training
- RemoteSGLangEngine for inference
- RLVRWorkflow with frontierco_reward_fn
- GRPO (Group Relative Policy Optimization)
```

**Training Flow**:
1. Load SFT checkpoint as initialization
2. Generate code completions via SGLang
3. Execute code and compute rewards
4. Update policy with PPO/GRPO objective
5. Periodic evaluation on validation set

**Configuration** (`frontierco_grpo.yaml`):
- Base from SFT checkpoint
- Sampling params (temperature, top_p, n_samples)
- Reward scaling and normalization
- KL penalty coefficient
- Rollout concurrency

---

### 4. Evaluation Scripts

#### 4.1 Single Model Evaluation (`frontierco_eval.py`)
**Based on**: `examples/math/gsm8k_eval.py`

**Purpose**: Evaluate a trained model on test set

**Process**:
1. Load model checkpoint
2. Generate code for each test problem
3. Execute generated code against test cases
4. Compute pass@1, pass@5, pass@10 metrics
5. Save generated code samples
6. Log results and statistics

**Metrics**:
- **pass@k** - Percentage of problems with ≥1 correct solution in k samples
- **Execution success rate** - % of samples that compile/run
- **Average test pass rate** - Mean % of tests passed per problem
- **Timeout rate** - % of samples exceeding time limit

---

#### 4.2 Batch Evaluation (`eval_all_frontierco_sizes.py`)
**Based on**: `examples/math/eval_all_qwen3_sizes.py`

**Purpose**: Evaluate multiple model sizes (1.7B, 8B, 14B, 32B)

**Features**:
- Compare base vs fine-tuned models
- Automatic checkpoint discovery
- Summary tables with deltas
- Parallel evaluation on 8 GPUs

---

### 5. Code Execution Backend

**Leverage Existing Infrastructure**:

#### Local Execution Path:
```
functioncall/code/local_verify.py
  └─> functioncall/code/function/testing_util.py
       └─> RuntimeModule + reliability_guard()
```

#### Remote Execution Path:
```
functioncall/code/verify.py
  └─> functioncall/base/call.py
       └─> async_invoke_function() → HTTP POST to FaaS
```

**Security Considerations**:
- Use `reliability_guard()` to disable dangerous operations
- Set resource limits (memory, CPU time)
- Isolate in subprocess with timeout
- Sanitize test case inputs

---

## Implementation Steps

### Phase 1: Dataset Setup
1. Create `areal/dataset/frontierco.py` with:
   - Support for HumanEval format initially
   - Chat template application
   - Test case parsing and validation
2. Add dataset config to `areal/dataset/__init__.py`

### Phase 2: Reward Function
1. Create `examples/frontierco/reward.py`:
   - Implement code extraction from markdown
   - Wrap existing code_verify functions
   - Handle timeout and error cases
2. Test reward function standalone

### Phase 3: SFT Training
1. Create `examples/frontierco/frontierco_sft.py`:
   - Copy from gsm8k_sft.py
   - Minimal modifications (import dataset)
2. Create `examples/frontierco/frontierco_sft.yaml`:
   - Start with Qwen/Qwen3-8B or DeepSeek-Coder-6.7B
   - Batch size: 64-128
   - Learning rate: 1e-5 to 5e-5
   - Max length: 4096 tokens
3. Create `examples/frontierco/frontierco_sft_megatron.py` for larger models

### Phase 4: RL Training
1. Create `examples/frontierco/frontierco_grpo.py`:
   - Copy from gsm8k_grpo.py
   - Use frontierco_reward_fn
2. Create `examples/frontierco/frontierco_grpo.yaml`:
   - Initialize from SFT checkpoint
   - n_samples: 4-8 for pass@k
   - Temperature: 0.8-1.0 for diversity
   - Reward scaling: tune based on reward distribution

### Phase 5: Evaluation
1. Create `examples/frontierco/frontierco_eval.py`:
   - Load checkpoint
   - Generate with n_samples for pass@k
   - Execute and compute metrics
2. Create `examples/frontierco/frontierco_eval_config.yaml`:
   - 8-GPU allocation for speed
3. Create batch evaluation scripts:
   - `run_all_frontierco_sizes.py`
   - `eval_all_frontierco_sizes.py`

### Phase 6: Documentation
1. Update `examples/frontierco/README.md` with:
   - Quick start guide
   - Configuration examples
   - Evaluation metrics explanation
   - Troubleshooting tips

---

## Key Design Decisions

### Dataset Choice
**Recommended**: Start with HumanEval
- Well-established benchmark (164 problems)
- Clean Python format
- Standardized test cases
- Easy to parse and validate

**Alternatives**:
- MBPP (More Basic Python Problems) - 974 problems, simpler
- APPS (Competitive programming) - More challenging
- Custom internal datasets

### Model Selection
**Recommended**: Qwen3-8B or DeepSeek-Coder-6.7B
- Good balance of size/performance
- Strong base coding ability
- Efficient for 8-GPU setup

### Execution Backend
**Recommended**: Local execution initially
- Faster development iteration
- No external dependencies
- Easier debugging
- Scale to remote later if needed

### Reward Structure
**Recommended**: Binary (pass/fail) initially
- Simpler to implement
- Clear signal
- Can add partial credit later if needed

---

## Expected File Structure

```
examples/frontierco/
├── PLAN.md (this file)
├── README.md
├── reward.py
├── frontierco_sft.py
├── frontierco_sft.yaml
├── frontierco_sft_megatron.py
├── frontierco_sft_megatron.yaml
├── frontierco_grpo.py
├── frontierco_grpo.yaml
├── frontierco_eval.py
├── frontierco_eval_config.yaml
├── run_all_frontierco_sizes.py
└── eval_all_frontierco_sizes.py

areal/dataset/
└── frontierco.py (new)
```

---

## Configuration Templates

### SFT Config Highlights
```yaml
experiment_name: frontierco-sft-megatron
model:
  path: Qwen/Qwen3-8B
  dtype: bfloat16
  max_length: 4096
train_dataset:
  batch_size: 128
  path: openai/humaneval  # or custom dataset
  type: sft
  max_length: 4096
```

### GRPO Config Highlights
```yaml
experiment_name: frontierco-grpo
actor:
  path: /path/to/sft/checkpoint
  reward_scaling: 5.0
  kl_ctl: 0.01
gconfig:
  n_samples: 8  # for pass@8
  temperature: 1.0
  max_new_tokens: 2048
```

---

## Metrics & Baselines

### Expected Performance (HumanEval)

| Model | Base | After SFT | After RL |
|-------|------|-----------|----------|
| Qwen3-1.7B | 30-40% | 40-50% | 50-60% |
| Qwen3-8B | 50-60% | 60-70% | 70-80% |
| Qwen3-14B | 60-70% | 70-75% | 75-85% |

*Note: These are rough estimates. Actual performance depends on training data, hyperparameters, and evaluation setup.*

### Key Metrics to Track
- **Training**: Loss, perplexity, learning rate, throughput
- **Evaluation**: pass@1, pass@5, pass@10, execution success rate
- **Reward**: Mean reward, reward distribution, timeout rate

---

## References

### Internal Code References
- **GSM8K Example**: `examples/math/gsm8k_*.py` - Template structure
- **Code Execution**: `functioncall/code/` - Core verification logic
- **TIR Example**: `examples/tir/` - Multi-turn tool execution
- **Math Parser**: `areal/reward/math_parser.py` - Answer verification patterns

### External Resources
- HumanEval: https://github.com/openai/human-eval
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- DeepSeek-Coder: https://github.com/deepseek-ai/DeepSeek-Coder

---

## Next Steps
1. Review this plan with team
2. Decide on dataset (HumanEval recommended)
3. Implement Phase 1 (dataset layer)
4. Test reward function standalone
5. Run SFT training on single model size
6. Evaluate and iterate
