# minimal_accelerate_commhook.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from accelerate import Accelerator

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def allreduce_avg_hook(process_group, bucket: dist.GradBucket):
    buf = bucket.buffer()
    print(f"Process {dist.get_rank()}: Executing allreduce_avg_hook on bucket with {buf.numel()} elements")
    fut = dist.all_reduce(buf, op=dist.ReduceOp.AVG, group=process_group, async_op=True).get_future()
    output = fut.then(lambda f: f.value()[0])
    print(f"Process {dist.get_rank()}: Output: {output}")
    return output

def main():
    accelerator = Accelerator()
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    # Wrap with Accelerate (adds DDP when multi-process)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Register comm hook if we’re in DDP
    if hasattr(model, "register_comm_hook"):
        print(f"Process {dist.get_rank()}: Registering communication hook")
        pg = getattr(model, "process_group", dist.group.WORLD)
        model.register_comm_hook(pg, allreduce_avg_hook)

    # One synthetic step
    x = torch.randn(20, 10, device=accelerator.device)
    y = torch.randn(20, 5, device=accelerator.device)
    loss = nn.MSELoss()(model(x), y)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    if accelerator.is_local_main_process:
        print(f"loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
