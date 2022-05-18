# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from cProfile import label
import itertools
import pprint
from typing import Dict
import argparse
import pickle
from collections import defaultdict
import os

import torch
from torch.utils import benchmark
import nvtx
import xformers.ops


def ref_attention(q, k, v):
    scale = 1.0
    scale  = (1.0 / q.shape[-1] ** 0.5)
    q = q * scale
    return (q @ k.transpose(-2, -1)).softmax(-1) @ v


min_run_time = 2
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = list(
    itertools.product([8], [64, 127, 128, 256, 2048, 4096], [8, 16, 32, 128, 256])
    # itertools.product([1, 8, 32, 256], [127, 128, 512, 513, 1023, 1024], [16, 32])
    # itertools.product([8], [127, 128, 512, 513, 1023, 1024], [32])
)

results = []
mem_use: Dict[str, Dict[str, float]] = defaultdict(dict)
def on_exception_enter_postmortem(f):
    import pdb
    import sys
    from functools import wraps
    import traceback

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as exception:
            traceback.print_exc()
            pdb.post_mortem(sys.exc_info()[2])
            raise

    return wrapper

@on_exception_enter_postmortem
def benchmark_forward(args):
    optimized_label =  "optimized" if args.label is None else args.label
    results = []
    if args.compare is not None:
        for name in args.compare.split(","):
            with open(f"{name}.pkl", "rb") as fd:
                results += pickle.load(fd)

    print(f"Processing {len(SHAPES)} cases")
    for num_threads in NUM_THREADS:
        for shape in SHAPES:
            print(f"===== {shape} =====")
            B, M, K = shape
            q = torch.rand(shape, device=device)
            v = torch.rand(shape, device=device)
            k = torch.rand(shape, device=device)
            sub_label = f"B={B}, M={M}, K={K}"

            if True:
                # q = torch.tensor([[1., 0],[0, 0],[0, 1],[0, 0]]).unsqueeze(0).to(device)
                # k = torch.tensor([[1., 0],[0, 1],[0, 0],[0, 0]]).unsqueeze(0).to(device)
                # v = torch.tensor([[1., 0],[0, 0],[0, 1],[1, 1]]).unsqueeze(0).to(device)
                # q, k, v = q, q, q

                with nvtx.annotate(sub_label, color="purple"):
                    r = xformers.ops.memory_efficient_attention2(q, k, v)
                    torch.cuda.synchronize()

                with nvtx.annotate(f"{sub_label}_vanilla", color="purple"):
                    rr = ref_attention(q, k, v)
                    torch.cuda.synchronize()
                assert (r - rr).abs().max() < 1e-5
                # continue

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            results.append(
                benchmark.Timer(
                    stmt="fn(q, q, q)",
                    globals={
                        "q": q,
                        "fn": xformers.ops.memory_efficient_attention2,
                    },
                    label="attention",
                    description=optimized_label,
                    sub_label=sub_label,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
            )
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_use[optimized_label][sub_label] = memory
            memory_str = f"Memory used: {memory} MB"

            print(optimized_label, memory_str)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            results.append(
                benchmark.Timer(
                    stmt="fn(q, q, q)",
                    globals={
                        "q": q,
                        "fn": ref_attention,
                    },
                    label="attention",
                    description="vanilla",
                    sub_label=sub_label,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
            )

            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_use["vanilla"][sub_label] = memory
            memory_str = f"Memory used: {memory} MB"
            print("Vanilla", memory_str)

    compare = benchmark.Compare(results)
    compare.print()

    if args.label is not None:
        with open(f"{args.label}.pkl", "wb+") as fd:
            pickle.dump([
                r for r in results if r.description == optimized_label
            ], fd)
    pprint.pprint(mem_use)


def benchmark_backward():
    print(f"Processing {len(SHAPES)} cases")
    print("Backward")
    for num_threads in NUM_THREADS:
        for shape in SHAPES:
            print(f"===== {shape} =====")
            B, M, K = shape
            q = torch.rand(shape, device=device, requires_grad=True)
            sub_label = f"B={B}, M={M}, K={K}"

            if True:
                r = xformers.ops.memory_efficient_attention(q, q, q)
                r.backward(torch.ones_like(q))

                grad = q.grad
                q.grad = None

                rr = ref_attention(q, q, q)
                rr.backward(torch.ones_like(q))
                assert (grad - q.grad).abs().max() < 1e-5

            out = xformers.ops.memory_efficient_attention(q, q, q)
            grad = torch.ones_like(q)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            results.append(
                benchmark.Timer(
                    stmt="out.backward(grad, retain_graph=True)",
                    globals={
                        "out": out,
                        "grad": grad,
                    },
                    label="attention",
                    description="optimized",
                    sub_label=sub_label,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
            )
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_use["optimized"][sub_label] = memory
            memory_str = f"Memory used: {memory} MB"

            print("Optimized", memory_str)

            out = ref_attention(q, q, q)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            results.append(
                benchmark.Timer(
                    stmt="out.backward(grad, retain_graph=True)",
                    globals={
                        "out": out,
                        "grad": grad,
                    },
                    label="attention",
                    description="vanilla",
                    sub_label=sub_label,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
            )

            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_use["vanilla"][sub_label] = memory
            memory_str = f"Memory used: {memory} MB"
            print("Vanilla", memory_str)

    compare = benchmark.Compare(results)
    compare.print()

    pprint.pprint(mem_use)

@on_exception_enter_postmortem
def benchmark_matmull(args):
    def ref_implem(a, b):
        return torch.matmul(a, b.transpose(2, 1))

    a = torch.tensor([[1.0, 0], [-1, 0], [0, 100]], device=device).unsqueeze(0)
    b = torch.tensor([[2.0, 3], [5, 7]], device=device).transpose(1, 0).contiguous().unsqueeze(0)
    r1 = ref_implem(a, b)
    r2 = torch.ops.xformers.dhaziza_custom_matmull(a, b)
    print("r1\n", r1)
    print("r2\n", r2)
    assert (r1 - r2).abs().max() < 1e-4, (r1 - r2).abs().max()

    optimized_label =  "optimized" if args.label is None else args.label
    results = []
    if args.compare is not None:
        for name in args.compare.split(","):
            with open(f"{name}.pkl", "rb") as fd:
                results += pickle.load(fd)
                for r in results:
                    r.description = name


    print(f"Processing {len(SHAPES)} cases")
    print("Forward")
    for num_threads in NUM_THREADS:
        for shape in SHAPES:
            print(f"===== {shape} =====")
            B, M, K = shape
            q = torch.rand(shape, device=device)
            sub_label = f"B={B}, M={M}, K={K}"

            if True:
                r = torch.ops.xformers.dhaziza_custom_matmull(q, q)
                rr = ref_implem(q, q)
                assert (r - rr).abs().max() < 1e-4, (r - rr).abs().max()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            results.append(
                benchmark.Timer(
                    stmt="fn(q, q)",
                    globals={
                        "q": q,
                        "fn": torch.ops.xformers.dhaziza_custom_matmull,
                    },
                    label="attention",
                    description=optimized_label,
                    sub_label=sub_label,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
            )
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_use[optimized_label][sub_label] = memory
            memory_str = f"Memory used: {memory} MB"

            print(optimized_label, memory_str)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            results.append(
                benchmark.Timer(
                    stmt="fn(q, q)",
                    globals={
                        "q": q,
                        "fn": ref_implem,
                    },
                    label="attention",
                    description="vanilla_matmull",
                    sub_label=sub_label,
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
            )

            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_use["vanilla"][sub_label] = memory
            memory_str = f"Memory used: {memory} MB"
            print("Vanilla", memory_str)

    compare = benchmark.Compare(results)
    compare.print()

    if args.label is not None:
        with open(f"{args.label}.pkl", "wb+") as fd:
            pickle.dump([
                r for r in results if r.description == optimized_label
            ], fd)
    pprint.pprint(mem_use)

parser = argparse.ArgumentParser()
parser.add_argument("--label", default=None, type=str, help="Store results to a file")
parser.add_argument("--compare", default=None, type=str, help="Compare to a previously stored benchmark")
args = parser.parse_args()

print(f"cuda-gdb -p {os.getpid()}")
# input("Enter any key to start ...")

benchmark_forward(args)
# benchmark_backward()
# benchmark_matmull(args)