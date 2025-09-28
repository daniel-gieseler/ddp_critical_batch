from torch import distributed as dist

def allreduce_avg_hook(state, bucket: dist.GradBucket):
    buf = bucket.buffer()
    print(f"Process {dist.get_rank()}: Executing allreduce_avg_hook on bucket with {buf.numel()} elements")
    fut = dist.all_reduce(buf, op=dist.ReduceOp.AVG, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0])