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
    max_len: int = 512,
    device: torch.device | None = None,
):
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(max_len):
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
