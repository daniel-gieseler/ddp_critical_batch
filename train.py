# minimal_accelerate_commhook.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from hooks.gns import GradientNoiseScaleState, gns_hook
from hooks.default import allreduce_avg_hook

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def get_dataset(sample_size):
    x = torch.randn(sample_size, 10)
    y = torch.randn(sample_size, 5)
    return TensorDataset(x, y)


def main(comm_hook):
    set_seed(42)
    accelerator = Accelerator()
    LOCAL_BATCH_SIZE = 20
    GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * accelerator.num_processes
    
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    dataloader = DataLoader(get_dataset(LOCAL_BATCH_SIZE), batch_size=LOCAL_BATCH_SIZE, shuffle=False)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if comm_hook != "none":
        assert hasattr(model, "register_comm_hook"), "Model must be wrapped in torch.nn.parallel.DistributedDataParallel"
        if comm_hook == "default":
            model.register_comm_hook(state=None, hook=allreduce_avg_hook)
        elif comm_hook == "gns":
            gns_state = GradientNoiseScaleState(device=accelerator.device)
            model.register_comm_hook(state=gns_state, hook=gns_hook)

    for x, y in dataloader:
        loss = nn.MSELoss()(model(x), y)
        accelerator.backward(loss)
        
        if comm_hook == "gns":
            gns_state.update(LOCAL_BATCH_SIZE, GLOBAL_BATCH_SIZE)
        
        # optimizer.step()
        # optimizer.zero_grad()
        break  # Just one batch

    if accelerator.is_local_main_process:
        print(f"loss: {loss.item():.8f}")
        if comm_hook == "gns":
            print(f"gradient_noise_scale: {gns_state.gns}")
    
    accelerator.end_training()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train with different communication hooks")
    parser.add_argument("--hook", choices=["none", "default", "gns"], default="none",
                       help="Communication hook type: none (no hook), default (allreduce_avg), or gns (gradient noise scale)")
    args = parser.parse_args()
    
    main(comm_hook=args.hook)
