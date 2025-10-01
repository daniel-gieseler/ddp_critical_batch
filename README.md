# Introduction

When scaling training across multiple GPUs, the hyperparameter batch size stands out as a first-class concern above all other hyperparameters. It becomes the primary axis of scaling, since in distributed data parallelism, adding more GPUs translates directly to increasing the effective batch size.

The logic behind training speedup is straightforward. Larger batch sizes give a clearer signal of the right gradient direction, making it possible to take bigger steps with learning rates that would otherwise be unstable.

Although changing batch size may seem like a simple quantitative adjustment, in practice it represents a qualitative shift in the training regime. In the literature, this shift is even deserving of its own name: Large-Batch Training. With it come a whole new set of challenges such as the generalization gap, demanding hyperparameter tuning, training instability, and diminishing returns as batch size grows.

In fact, for many years, large-batch training was thought to be impractical, and small batches remained the default choice. Only recently have we developed the tools to tame this beast - recovering the convergence seen in small-batch training while also benefiting from the reduced training times of heavy parallelism.

# Critical Batch Size

One of those tools is the critical batch size $$B_{\mathrm{crit}}$$, the batch size at which further increases yield diminishing returns: below it, increasing batch size speeds up learning nearly linearly (in terms of wall-time), but above it, extra samples are mostly wasted. Here we are following the theory developed in the paper “An Empirical Model of Large-Batch Training” (2018), where $$B_{\mathrm{crit}}$$ is approximately equal to the gradient noise scale.

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

# Implementation



```diff
// Before → After (pseudo-code)
- const cache = {};            // old line removed
+ const cache = new Map();     // new line added
  cache.set('user', id);
```
