from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

MODEL_NAME = "mistral"
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM(MODEL_NAME, device=DEVICE, dtype=DTYPE)


@torch.no_grad()
def top_p_sampling(
    prompt: str,
    model,
    tokenizer,
    p: float,
    temperature: float,
    max_new_token: int = 128,
    device: torch.device | None = None,
):
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(max_new_token):
        outputs = model(input_ids=token_ids)  # B, seq_len , vocab_size

        logits = outputs.logits[:, -1, :]  # B,vocab_size

        probas = F.softmax(logits / temperature, dim=-1)

        sorted, indices = torch.sort(probas, dim=-1, descending=True)
        cumSum = torch.cumsum(sorted, dim=-1, dtype=sorted.dtype)

        mask = cumSum <= p

        selected_proba = mask * sorted
        selected_proba = selected_proba / selected_proba.sum(dim=-1, keepdim=True)

        pos = torch.multinomial(selected_proba, 1)

        next_token_id = torch.gather(indices, dim=-1, index=pos)

        token_ids = torch.cat([token_ids, next_token_id], dim=-1)

        if next_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(token_ids[0])


# no kv cahing for the moment fix model later
# add short sequences penalty
@torch.no_grad()
def beam_search(
    prompt,
    model,
    tokenizer,
    num_beams=4,
    max_new_tokens=50,
    device=None,
):
    input_ids: torch.Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(
        device
    )  # B,T
    batch_size, T = input_ids.shape
    input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)  # B,beams,T
    scores = torch.zeros(batch_size, num_beams, device=device)

    done = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        if done.all():
            break

        B, beams, seq_len = input_ids.shape
        input_ids_flat = input_ids.view(B * beams, seq_len)
        next_token_logits = model(input_ids_flat).logits[:, -1, :]
        next_token_logits = next_token_logits.view(B, beams, -1)  # B,beams,V
        next_token_scores = torch.nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # B,beams,V

        next_token_scores[done] = float("-inf")
        next_token_scores[done, eos_token_id] = 0

        next_token_scores = scores.unsqueeze(-1) + next_token_scores  # B,beams,V
        next_token_scores = next_token_scores.view(B, -1)  # B, beams*V

        top_k_scores, top_k_idx = torch.topk(next_token_scores, k=num_beams, dim=-1)

        vocab_size = next_token_logits.size(-1)
        next_beam_idx = (
            top_k_idx // vocab_size
        )  # B,beams  which OLD beam each new beam came from
        next_token_idx = top_k_idx % vocab_size  # B,beams  which token was chosen

        input_ids = torch.gather(
            input_ids,
            dim=1,
            index=next_beam_idx.unsqueeze(-1).expand(-1, -1, input_ids.size(-1)),
        )
        input_ids = torch.cat([input_ids, next_token_idx.unsqueeze(-1)], dim=-1)

        done = torch.gather(done, dim=1, index=next_beam_idx)
        done = done | (next_token_idx == eos_token_id)

        scores = top_k_scores

    return input_ids[:, 0, :]
