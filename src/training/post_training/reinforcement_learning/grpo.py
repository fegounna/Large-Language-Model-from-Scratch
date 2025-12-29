from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
import random
import copy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class RolloutGroup:
    """Stores one prompt with K responses and their metrics."""

    prompt: str
    ground_truth: str
    responses: List[str]
    rewards: torch.Tensor  # (K,)
    advantages: torch.Tensor  # (K,)
    old_logp_sums: torch.Tensor  # (K,)
    ref_logp_sums: torch.Tensor  # (K,)


class ReplayBuffer:
    """Circular buffer for storing rollout groups."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[RolloutGroup] = []
        self.position = 0

    def add(self, group: RolloutGroup):
        if len(self.buffer) < self.capacity:
            self.buffer.append(group)
        else:
            self.buffer[self.position] = group
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[RolloutGroup]:
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)

    def __len__(self) -> int:
        return len(self.buffer)


def normalize_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def compute_ppo_loss(
    advantages: torch.Tensor,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    clip_range: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    unclipped_loss = ratio * advantages
    clipped_loss = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    with torch.no_grad():
        clip_fraction = (clipped_loss < unclipped_loss).float().mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()

    metrics = {
        "policy_loss": policy_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "clip_fraction": clip_fraction.item(),
        "approx_kl": approx_kl.item(),
    }

    return policy_loss, metrics


class GRPOTrainer:
    """Group Relative Policy Optimization trainer with experience replay."""

    def __init__(
        self,
        policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_fn: Callable[[str, str], float],
        learning_rate: float = 1e-5,
        clip_range: float = 0.2,
        kl_penalty: float = 0.02,
        grad_clip: float = 1.0,
        buffer_size: int = 2000,
    ):
        self.policy = policy
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.clip_range = clip_range
        self.kl_penalty = kl_penalty
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.ref_policy = self._create_reference_policy()

    def _create_reference_policy(self) -> PreTrainedModel:
        ref = copy.deepcopy(self.policy)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad_(False)
        return ref

    @torch.no_grad()
    def collect_rollouts(
        self,
        prompts: List[Tuple[str, str]],  # (prompt, ground_truth) pairs
        group_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ):
        self.policy.eval()
        device = next(self.policy.parameters()).device

        self.ref_policy.to(device)

        for prompt, ground_truth in prompts:
            responses = generate_responses(
                self.policy,
                self.tokenizer,
                prompt,
                n_samples=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            rewards = torch.tensor(
                [self.reward_fn(response, ground_truth) for response in responses],
                dtype=torch.float32,
            )

            advantages = normalize_advantages(rewards)

            prompts_batch = [prompt] * group_size
            old_logp = compute_logprobs(
                self.policy, self.tokenizer, prompts_batch, responses
            )
            ref_logp = compute_logprobs(
                self.ref_policy, self.tokenizer, prompts_batch, responses
            )

            group = RolloutGroup(
                prompt=prompt,
                ground_truth=ground_truth,
                responses=responses,
                rewards=rewards.cpu(),
                advantages=advantages.cpu(),
                old_logp_sums=old_logp.cpu(),
                ref_logp_sums=ref_logp.cpu(),
            )
            self.replay_buffer.add(group)

        self.policy.train()

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        groups = self.replay_buffer.sample(batch_size)
        if not groups:
            return {}

        prompts, responses, advantages_list, old_logp_list, ref_logp_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for group in groups:
            k = len(group.responses)
            prompts.extend([group.prompt] * k)
            responses.extend(group.responses)
            advantages_list.append(group.advantages)
            old_logp_list.append(group.old_logp_sums)
            ref_logp_list.append(group.ref_logp_sums)

        device = next(self.policy.parameters()).device
        advantages = torch.cat(advantages_list).to(device)
        old_logp = torch.cat(old_logp_list).to(device)
        ref_logp = torch.cat(ref_logp_list).to(device)

        # This is the trick on why doing off-policy, because computing log probs is cheaper than sampling because it is parallelized
        new_logp = compute_logprobs(self.policy, self.tokenizer, prompts, responses)

        policy_loss, metrics = compute_ppo_loss(
            advantages, new_logp, old_logp, self.clip_range
        )

        kl_divergence = new_logp - ref_logp
        kl_loss = self.kl_penalty * kl_divergence.mean()

        total_loss = policy_loss + kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        metrics.update(
            {
                "total_loss": total_loss.item(),
                "kl_loss": kl_loss.item(),
                "kl_mean": kl_divergence.mean().item(),
                "grad_norm": grad_norm.item(),
                "buffer_size": len(self.replay_buffer),
            }
        )

        return metrics

    def train(
        self,
        train_prompts: List[Tuple[str, str]],
        n_iterations: int = 1000,
        rollout_interval: int = 10,
        rollouts_per_collection: int = 64,
        batch_size: int = 32,
        group_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ):
        initial_sample = random.sample(
            train_prompts, min(rollouts_per_collection, len(train_prompts))
        )
        self.collect_rollouts(
            initial_sample,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        for iteration in range(1, n_iterations + 1):
            if iteration % rollout_interval == 0:
                sample = random.sample(
                    train_prompts, min(rollouts_per_collection, len(train_prompts))
                )
                self.collect_rollouts(
                    sample,
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            metrics = self.train_step(batch_size)

            if iteration % 10 == 0 and metrics:
                print(
                    f"[Iter {iteration:04d}] "
                    f"loss={metrics['total_loss']:.4f} "
                    f"pg_loss={metrics['policy_loss']:.4f} "
                    f"kl={metrics['kl_mean']:.4f} "
                    f"ratio={metrics['ratio_mean']:.3f} "
                    f"clip_frac={metrics['clip_fraction']:.2f} "
                    f"buffer={metrics['buffer_size']}"
                )


@torch.no_grad()
def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> List[str]:
    """Generate multiple responses for a single prompt."""
    pass


@torch.no_grad()
def compute_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    responses: List[str],
) -> torch.Tensor:
    """
    Compute sum of log probabilities for response tokens.
    Returns: (batch_size,) tensor of logp sums
    """
    pass
