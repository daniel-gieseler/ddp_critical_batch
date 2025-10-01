# Intro

When scaling training across multiple GPUs, the hyperparameter batch size stands out as a first-class concern above all other hyperparameters. It becomes the primary axis of scaling, since in distributed data parallelism, adding more GPUs translates directly to increasing the effective batch size.

The logic behind training speedup is straightforward. Larger batch sizes give a clearer signal of the right gradient direction, making it possible to take bigger steps with learning rates that would otherwise be unstable.

Although changing batch size may seem like a simple quantitative adjustment, in practice it represents a qualitative shift in the training regime. In the literature, this shift is even deserving of its own name: Large-Batch Training. With it come a whole new set of challenges such as the generalization gap, demanding hyperparameter tuning, training instability, and diminishing returns as batch size grows.

In fact, for many years, large-batch training was thought to be impractical, and small batches remained the default choice. Only recently have we developed the tools to tame this beast - recovering the convergence seen in small-batch training while also benefiting from the reduced training times of heavy parallelism.

# Theory

One of those tools is the critical batch size $$B_{\mathrm{crit}}$$, the batch size at which further increases yield diminishing returns: below it, increasing batch size speeds up learning nearly linearly (in terms of wall-time), but above it, extra samples are mostly wasted. Here we are following the theory developed in the paper “An Empirical Model of Large-Batch Training” (2018), where $$B_{\mathrm{crit}}$$ is approximately equal to the gradient noise scale (GNS).

$$
B_{\mathrm{crit}} \approx \frac{S}{|G|^2}
$$

$${|G|}$$ is the norm of the true gradient and $${S}$$ is its variance. One can think of it as comparing the magnitude of noise to signal in the gradients. But we can't measure these value directly. Instead, we can an estimate $$|G_{\mathrm{est}}|$$ given a batch size $$B_{\mathrm{est}}$$, and relate to the true gradient by the following relation:

$$
|G_{\mathrm{est}}|^2 = |G|^2 + \frac{1}{B_{\mathrm{est}}} \ S
$$

If we had two estimates at two different batch sizes we could actually solve it for $${|G|}$$ and $${S}$$. In the paper, they propose a method (Appendix A.1) that is computationally efficient and fits naturally into a Distributed Data Parallel (DDP) setup, where we already see two sizes of batches during a training cycle: the local batch on each GPU and the global batch size of their sum. In this case our two measuments would be: $$(B_{\mathrm{local}} \, G_{\mathrm{local}})$$ and $$(B_{\mathrm{global}} \, G_{\mathrm{global}})$$. Then we can solve for the desired values like this:

$$
|G|^2 = \frac{1}{B_{\mathrm{global}} - B_{\mathrm{local}}}
\Bigl( B_{\mathrm{global}}\|G_{\mathrm{global}}|^2 \-\ B_{\mathrm{local}}\|G_{\mathrm{local}}|^2 \Bigr)
$$

$$
\mathcal{S} = \frac{1}{\tfrac{1}{B_{\mathrm{local}}} - \tfrac{1}{B_{\mathrm{global}}}}
\bigl( |G_{\mathrm{local}}|^2 - |G_{\mathrm{global}}|^2 \bigr)
$$

One every step of training we can calculate these values. But they are still too noisy. So, before doing their ratio, an exponential moving average is kept for both values throughout training. 

Notice that it is expected that $$B_{\mathrm{crit}}$$ will increase during training as the model becomes better at the task.

# Practice

To make the measurements at both local and global batch sizes, we need to intercept gradients being calculated during the backwards pass, before they are `all_reduce`. Luckily Pytorch exposes a way in with Communication Hooks. We can register a hook with `register_comm_hook`. It expects a `state` to persist between hook calls and a function `hook`. Here is a minimal training script and the edits necessary.

```diff
+ gns_state = GradientNoiseScaleState(device=accelerator.device)
+ model.register_comm_hook(state=gns_state, hook=gns_hook)

for x, y in dataloader:
    loss = nn.MSELoss()(model(x), y)
    accelerator.backward(loss)
+   gns_state.update(LOCAL_BATCH_SIZE, GLOBAL_BATCH_SIZE)
    optimizer.step()
    optimizer.zero_grad()
```

In the above code, the hook is being triggered multiple times as buckets of gradients become ready during the backwards pass. The hook is storing those gradients for us in the state. Notice that when a comm hook is registered, we are now responsible for doing the `all_reduce average` ourselves.

```python
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
```

Finally, once the backwards pass is complete, we trigger update on the `state` which performs all the calculations described in the previous session.

```python
class GradientNoiseScaleState:
    def __init__(self, device, window_n=9999):
        self.local_sq_norms = torch.tensor(0., device=device)
        self.global_sq_norms = torch.tensor(0., device=device)
        self.ema_alpha = 2 / (window_n + 1)
        self.ema = {'sq_norm': 0., 'var': 0.}
        self.gns = float('nan')

    def _average_local_sq_norms(self):
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
```

# Bonus

TODO: add training speed cost of using gns_hook as opposed to no hook.

TODO: add a study on the paper concerns about learning rate being well-tuned
