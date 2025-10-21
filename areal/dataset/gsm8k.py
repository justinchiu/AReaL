from typing import Optional

from datasets import load_dataset

question_suffix = "\nPlease format computation in <<{computation}>> and your answer as #### {answer}."


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        # Create messages format with user question and assistant answer
        messages = [
            {"role": "user", "content": sample["question"] + question_suffix},
            {"role": "assistant", "content": sample["answer"]},
        ]

        # Apply chat template to get the full sequence with special tokens
        seq_token = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,  # We have the assistant response
        )

        # To compute loss mask, we need to know where the assistant response starts
        # Tokenize just the user message to find the prompt length
        user_only = [{"role": "user", "content": sample["question"]}]
        prompt_token = tokenizer.apply_chat_template(
            user_only,
            tokenize=True,
            add_generation_prompt=True,  # Adds <|im_start|>assistant
        )

        # Loss mask: 0 for prompt (user + assistant start), 1 for assistant response
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))

        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"] + question_suffix
                #+ "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
