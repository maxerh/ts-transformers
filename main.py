import csv
import tqdm
import torch
import numpy as np

from model.attentions import (StandardAttention, SwinAttention, SparseAttention, LogSparseAttention, ChunkedAttention,
                              FocalAttention, LinearAttention)


def performance_test(attn_factory, device, iterations=10, batch_size=16, seq_length=512, dim=128):
    """
    Measures average inference time per iteration and peak memory usage for a given attention module.
    """
    # Create an instance of the attention module.
    module = attn_factory(dim).to(device)
    x = torch.randn(batch_size, seq_length, dim, device=device)

    # Warm-up passes.
    for _ in range(10):
        _ = module(x)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        _ = module(x)
    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / iterations  # ms per iteration
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)    # MB allocated
    #print(f"Avg Inference Time: {avg_time:6.3f} ms, Peak Memory: {max_mem:6.3f} MB")
    return avg_time, max_mem


if __name__ == '__main__':
    attention_mechanisms = {
        "StandardAttention": lambda dim: StandardAttention(dim, heads=8),                                   # 2017, google
        "SwinAttention": lambda dim: SwinAttention(dim, window_size=8, shift_size=0, heads=8),              # 2021, microsoft
        "ChunkedAttention": lambda dim: ChunkedAttention(dim, window_size=8, overlap=0, heads=8),            # 2018, google
        #"FocalAttention": lambda dim: FocalAttention(dim, radius=8, heads=8),                              # 2021, microsoft
        "LinearAttention": lambda dim: LinearAttention(dim, heads=8),                                       # 2020, Idiap Research Institute, Switzerland
        #"SparseAttention": lambda dim: SparseAttention(dim, heads=8, window_size=16, global_stride=16),    # 2025, DeepSeek
        "LogSparseAttention": lambda dim: LogSparseAttention(dim, heads=8, local_window=8, num_exps=4),     # 2019, univ. of california
    }
    devices = ['cuda', 'cpu']
    start = 32
    step = 32
    stop = 2048
    T = np.linspace(start, stop, int((stop - start) / step + 1), dtype=int)

    # Run test for each device.
    for device in devices:
        print(device)
        tabletime = []
        tablemem = []
        tabletime.append(['window_size'] + list(attention_mechanisms.keys()))
        tablemem.append( ['window_size'] + list(attention_mechanisms.keys()))
        # Run tests for different sequence lengths.
        for seq_len in tqdm.tqdm(T):
            tableT = [seq_len]
            tableMB = [seq_len]
            # Run tests for each attention mechanism.
            for name, factory in attention_mechanisms.items():
                time, memory = performance_test(factory, device, iterations=500, batch_size=16, seq_length=seq_len, dim=128)
                tableT.append(time)
                tableMB.append(memory)
            tabletime.append(tableT)
            tablemem.append(tableMB)

        with open(f'duration_ms_{device}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(tabletime)
        if device != "cpu":
            with open(f'memory_alloc_mb_{device}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(tablemem)


