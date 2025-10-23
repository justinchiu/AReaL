import os
import sys

import torch.distributed as dist

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

from examples.frontierco.reward import frontierco_reward_fn


def main(args):
    # Use eval-specific config by default if not overridden
    if '--config' not in ' '.join(args):
        default_config = os.path.join(
            os.path.dirname(__file__), 'frontierco_eval_config.yaml'
        )
        args = ['--config', default_config] + args
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    dist.init_process_group("gloo")
    # Create a group for stats all-reduce.
    group = dist.new_group()

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    # Create dataset and dataloaders
    # Note: Using "train" split as OR-Instruct-Data-3K doesn't have a separate test split
    valid_dataset = get_custom_dataset(
        split="train", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=rank,
        world_size=world_size,
        dataset_config=config.valid_dataset,
    )

    # Initialize inference engine
    config.rollout.max_head_offpolicyness = int(1e12)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize()

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=frontierco_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    # Run evaluation.
    cnt = 0
    for data in valid_dataloader:
        for item in data:
            eval_rollout.submit(item, workflow)
            cnt += 1
    eval_rollout.wait(cnt, timeout=None)

    eval_rollout_stats = stats_tracker.export_all(reduce_group=group)
    print(f"Evaluation results:\n{tabulate_stats(eval_rollout_stats)}")
    eval_rollout.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
