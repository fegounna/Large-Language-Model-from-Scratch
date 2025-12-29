import torch
import torch.nn.functional as F


def tokenize_prompt_and_out(
    prompt_strs: list[str], output_strs: list[str], tokenizer
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).

    Parameters
    ----------
    prompt_strs: list[str]
        List of prompt strings.
    output_strs: list[str]
        List of output strings.
    tokenizer: PreTrainedTokenizer
        Tokenizer to use for tokenization.

    Returns
    -------
    dict[str, torch.Tensor]
        Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
        1): a mask on the response tokens in the labels.
    """
    batch_size = len(prompt_strs)
    assert batch_size == len(output_strs), (
        "prompt_strs and output_strs must have the same length"
    )

    prompt_tokens_list = []
    output_tokens_list = []
    prompt_and_output_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)

        prompt_tokens_list.append(prompt_tokens)
        output_tokens_list.append(output_tokens)

        prompt_and_output_lens.append(len(prompt_tokens) + len(output_tokens))

    max_len = max(prompt_and_output_lens)
    seq_len = max_len - 1

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i in range(batch_size):
        combined_tokens = prompt_tokens_list[i] + output_tokens_list[i]
        input_ids_seq = combined_tokens[:-1]
        labels_seq = combined_tokens[1:]
        actual_len = len(input_ids_seq)

        input_ids[i, :actual_len] = torch.tensor(input_ids_seq, dtype=torch.long)
        labels[i, :actual_len] = torch.tensor(labels_seq, dtype=torch.long)

        prompt_len = len(prompt_tokens_list[i])
        response_start_idx = prompt_len - 1
        response_end_idx = actual_len

        if response_end_idx > response_start_idx:
            response_mask[i, response_start_idx:response_end_idx] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Parameters
    ----------
    logits: torch.Tensor
        Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.

    Returns
    -------
    torch.Tensor
        Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    logsumexp_logits = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - logsumexp_logits
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens) from a causal language model,
    and optionally the entropy of the model's next-token distribution.

    Parameters
    ----------
    model: PreTrainedModel
    input_ids: torch.Tensor
        Shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method.
    labels: torch.Tensor
        Shape (batch_size, sequence_length), labels as produced by your tokenization method.
    return_token_entropy: bool
        If True, also return per-token entropy by calling compute_entropy.

    Returns
    -------
    dict[str, torch.Tensor].
        "log_probs": shape (batch_size, sequence_length), conditional log-probabilities log p_θ(x_t | x_{<t}).
        "token_entropy": optional, shape (batch_size, sequence_length), per-token entropy for each position
        (present only if return_token_entropy=True).
    """
    logits = model(input_ids).logits  # (batch_size, sequence_length, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)

    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )

    result = {"log_probs": log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over tensor elements and normalize by a constant while respecting a boolean mask.

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to normalize.
    mask: torch.Tensor
        Boolean mask (or 0/1 tensor) indicating which elements to include.
        Must be broadcastable with tensor.
    normalize_constant: float
        The constant to divide by for normalization.
    dim: int | None
        Dimension(s) along which to sum. If None, sum over all dimensions.

    Returns
    -------
    torch.Tensor
        The masked sum divided by normalize_constant.
    """
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim)
    result = masked_sum / normalize_constant

    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.

    Parameters
    ----------
    policy_log_probs: torch.Tensor
        Shape (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
    response_mask: torch.Tensor
        Shape (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
    gradient_accumulation_steps: int
        Number of microbatches per optimizer step.
    normalize_constant: float
        The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation.
        metadata: Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """

    nll_loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )

    loss = nll_loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "nll_loss": nll_loss.detach(),
    }

    return loss, metadata  # for logging
