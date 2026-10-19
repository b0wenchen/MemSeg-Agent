#!/usr/bin/env python
"""Aggregate client static-memory checkpoints without transmitting SAM2."""

from __future__ import annotations

import argparse

import torch

from memseg_agent.federated import fedavg_static_memories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", nargs="*", type=float)
    args = parser.parse_args()
    states = []
    for path in args.inputs:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        states.append(checkpoint["static_memory"] if "static_memory" in checkpoint else checkpoint)
    state = fedavg_static_memories(states, args.weights)
    torch.save({"static_memory": state, "num_clients": len(states)}, args.output)
    print(f"wrote memory-only FedAvg checkpoint to {args.output}")


if __name__ == "__main__":
    main()
