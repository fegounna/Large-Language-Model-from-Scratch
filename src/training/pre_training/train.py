from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
from dataclasses import dataclass
import numpy as np

from model.architecture import TransformerLM
from training.trainning_utils import cross_entropy, cosine_annealing_lr_schedule,gradient_clipping,AdamW

ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    print("DDP NOT SET UP")

SEED = 1337
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


@dataclass
class config:
    micro_batch_size = 16
    grad_accum_steps = 1
    seq_len = 512
    path = ""
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_step = 10
    weight_decay = 0.01
    max_steps = 1000
    gradient_clipping = 1.0


cfg = config()


class Dataloader:
    def __init__(self, path, B, T, device, num_worker):
        self.data = np.memmap(
            path, dtype=np.uint16, mode="r"
        )  # uint16 because our tokenizer only 32K
        self.B = B
        self.T = T
        self.device = device
        self.len = len(self.data)

        base_seed = 199
        self.g = np.random.default_rng(base_seed + num_worker)

    def get_batch(self):
        ix = self.g.integers(0, self.len - (self.T + 1), size=(self.B,))

        batch = torch.tensor(
            np.stack([self.data[i : i + self.T + 1] for i in ix]), dtype=torch.long
        )

        x = batch[:, :-1].to(device=self.device, non_blocking=True)
        y = batch[:, 1:].to(device=self.device, non_blocking=True)

        return x, y


train_loader = Dataloader(
    path=cfg.path,
    B=cfg.micro_batch_size,
    T=cfg.seq_len,
    device=device,
    num_worker=ddp_rank,
)


model = TransformerLM().to(device)
torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = AdamW(params=model.parameters, weight_decay=cfg.weight_decay)

for step in range(cfg.max_steps):
    for micro_step in range(cfg.grad_accum_steps):
        optimizer.zero_grad()
        x, y = train_loader.get_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
            loss = cross_entropy(logits, y) / cfg.grad_accum_steps
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == cfg.grad_accum_steps - 1
                )
            loss.backward()

    norm = gradient_clipping(model.parameters, cfg.gradient_clipping)
    lr = cosine_annealing_lr_schedule(
        step,
        alpha_max=cfg.max_lr,
        alpha_min=cfg.min_lr,
        T_w=cfg.warmup_step,
        T_c=cfg.max_steps,
    )
    for param in optimizer.param_groups:
        param["lr"] = lr
    optimizer.step()
