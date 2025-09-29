import torch

def gns_hook(state, bucket):
    squared_norm = lambda x: x.pow(2).sum()
    buffer = bucket.buffer()
    state.local_sq_norms += squared_norm(buffer)
    fut = torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.AVG, async_op=True).get_future()
    def update_global_sq_norms(fut):
        buffer = fut.value()[0]
        state.global_sq_norms += squared_norm(buffer)
        return buffer
    return fut.then(update_global_sq_norms)

class GradientNoiseScaleState:
    def __init__(self, device, window_n=9999):
        self.local_sq_norms = torch.tensor(0., device=device)
        self.global_sq_norms = torch.tensor(0., device=device)
        self.ema_alpha = 2 / (window_n + 1)
        self.ema = {'sq_norm': 0., 'var': 0.}
        self.gns = float('nan')

    def _average_local_sq_norms(self):
        # TODO: maybe I can just reduce to rank 0
        torch.distributed.all_reduce(self.local_sq_norms, op=torch.distributed.ReduceOp.AVG)

    def _solve_estimates(self, local_sq_norm: float, global_sq_norm: float, local_batch_size: int, global_batch_size: int):
        """Compute unbiased estimates of squared gradient norm and variance.
        Based on "An Empirical Model of Large-Batch Training" (2018).
        Notice the aggregation symmetry:
          - local_sq_norm:  average of the squared norm of the gradient
          - global_sq_norm: squared norm of the average of the gradient
        """
        est_sq_norm = (global_batch_size * global_sq_norm - local_batch_size * local_sq_norm) / (global_batch_size - local_batch_size)
        est_var = (local_sq_norm - global_sq_norm) / (1 / local_batch_size - 1 / global_batch_size)
        return est_sq_norm, est_var

    def _update_ema(self, name: str, value: float):
        """Update the exponential moving average of the given variable."""
        self.ema[name] = (1 - self.ema_alpha) * self.ema[name] + self.ema_alpha * value

    def update(self, local_batch_size: int, global_batch_size: int):
        self._average_local_sq_norms()
        #
        est_sq_norm, est_var = self._solve_estimates(self.local_sq_norms.item(), self.global_sq_norms.item(), local_batch_size, global_batch_size)
        self._update_ema('sq_norm', est_sq_norm)
        self._update_ema('var', est_var)
        self.gns = max(self.ema['var'], 1e-8) / max(self.ema['sq_norm'], 1e-8)
        #
        self.local_sq_norms.zero_()
        self.global_sq_norms.zero_()