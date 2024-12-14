# Performance Profiling Guide

## PyTorch Profiler Usage

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input)
print(prof.key_averages().table())
```

## Common Bottlenecks

1. Data loading
2. Model forward pass
3. Backward pass
4. GPU memory transfers

## Optimization Tips

- Use appropriate batch sizes
- Enable GPU memory pinning
- Optimize data loading pipeline
- Consider mixed precision training

[Add specific profiling results and recommendations for your model]
