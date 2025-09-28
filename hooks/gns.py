import torch

def gns_hook(state, bucket):
    squared_norm = lambda x: x.pow(2).sum()
    buf = bucket.buffer()
    state.local_sq_norms.append(squared_norm(buf))
    fut = torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.AVG, async_op=True).get_future()
    def callback(fut):
        buf = fut.value()[0]
        state.global_sq_norms.append(squared_norm(buf))
        return buf
    return fut.then(callback)

class GradientNoiseScaleState:
    def __init__(self, window_n=9999, eps=1e-8):
        self.window_n = window_n
        self.eps = eps
        self.ema = {'sq_norm': 0., 'var': 0.}
        self.gradient_noise_scale = float('nan')
        self.local_sq_norms = []
        self.global_sq_norms = []

    # TODO: get rid of this
    def update_from_buckets(self, local_batch_size, global_batch_size):
        """Updates GNS state from bucket statistics and returns the current gradient noise scale."""
        local_sq_norm, global_sq_norm = self.get_bucket_stats()
        return self.update(local_sq_norm, global_sq_norm, local_batch_size, global_batch_size)

    def get_bucket_stats(self):
        """Returns the current gradient statistics and clears the bucket lists."""
        local_sq_norm = sum(self.local_sq_norms)
        global_sq_norm = sum(self.global_sq_norms)
        self.local_sq_norms = []
        self.global_sq_norms = []
        stats = torch.stack([local_sq_norm, global_sq_norm])
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.AVG)
        return stats[0].item(), stats[1].item()

    def _unbiased_estimates(self, sq_small, sq_large, n_small, n_large):
        """Compute unbiased estimates of squared gradient norm and variance.
        Based on "An Empirical Model of Large-Batch Training" (2018).
        """
        est_sq_norm = (n_large * sq_large - n_small * sq_small) / (n_large - n_small)
        est_var = (sq_small - sq_large) / (1 / n_small - 1 / n_large)
        return est_sq_norm, est_var

    def _calculate_gradient_noise_scale(self, est_sq_norm, est_var):
        """Calculate gradient noise scale with stability:
           - EMA: Updates moving averages for gradient norm and variance
           - eps: Prevents division by zero in final ratio
        """
        alpha = 2 / (self.window_n + 1)
        self.ema['sq_norm'] = (1 - alpha) * self.ema['sq_norm'] + alpha * est_sq_norm
        self.ema['var'] = (1 - alpha) * self.ema['var'] + alpha * est_var
        return max(self.ema['var'], self.eps) / max(self.ema['sq_norm'], self.eps)

    def update(self, local_sq_norm, global_sq_norm, local_batch_size, global_batch_size):
        est_sq_norm, est_var = self._unbiased_estimates(local_sq_norm, global_sq_norm, local_batch_size, global_batch_size)
        self.gradient_noise_scale = self._calculate_gradient_noise_scale(est_sq_norm, est_var)
        return self.gradient_noise_scale