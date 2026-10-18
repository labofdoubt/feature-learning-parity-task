"""Weight-norm structure at the final checkpoint vs initialization (one-block model).

3 x 2 grid: rows = embedding column norms (one per input bit), linear-layer row
norms (one per hidden unit, histogram), readout entry magnitudes (histogram,
vector norm in the title); columns = final checkpoint, init. Init is rebuilt by
running the trainer for zero steps with the run's own config (same seed).

Usage: python scripts/analyze_weight_norms_init_vs_final.py --run-dir runs/<name> [--block 0]
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from parity_net.checkpoint import load_checkpoint


def init_state(run_dir: Path) -> dict:
    """Bit-exact init weights: train for 0 steps into a temp dir with the same config."""
    import subprocess
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="_init_", dir=run_dir.parent))
    cfg["training"].update(num_steps=0, progress_every=0, output_dir=str(tmp), device="cpu")
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run([sys.executable, "-m", "parity_net.train", "--config", str(tmp / "config.yaml")],
                   check=True, capture_output=True, cwd=str(_ROOT))
    state = torch.load(tmp / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)["model_state"]
    shutil.rmtree(tmp)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    final = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    states = {"final checkpoint (step %d)" % int(final.get("step", 0)): final["model_state"],
              "init": init_state(run_dir)}
    b = args.block
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    data = {}
    for j, (label, s) in enumerate(states.items()):
        E = s["embedding.weight"].double()                        # N x d
        V = s[f"blocks.{b}.linear.weight"].double()              # N x N
        key = "readout.weight" if "readout.weight" in s else sorted(k for k in s if k.startswith("layerwise_readouts"))[0]
        w = s[key].double().reshape(-1)
        data[label] = dict(emb=E.norm(dim=0).numpy(), rows=V.norm(dim=1).numpy(), w=w.abs().numpy(), wnorm=float(w.norm()))
    emb_max = max(d["emb"].max() for d in data.values()) * 1.1
    row_bins = np.linspace(0, max(d["rows"].max() for d in data.values()) * 1.05, 60)
    w_bins = np.linspace(0, max(d["w"].max() for d in data.values()) * 1.05, 60)
    for j, (label, d) in enumerate(data.items()):
        ax = axes[0][j]; ax.bar(range(len(d["emb"])), d["emb"]); ax.set_ylim(0, emb_max)
        ax.set_title(f"embedding column norms — {label}", fontsize=10); ax.set_xlabel("input bit"); ax.set_ylabel("||E[:, i]||")
        ax = axes[1][j]; ax.hist(d["rows"], bins=row_bins); ax.set_yscale("log")
        ax.set_title(f"linear-layer row norms (block {b}) — {label}  mean {d['rows'].mean():.3f}, max {d['rows'].max():.3f}", fontsize=10)
        ax.set_xlabel("||V[j, :]||"); ax.set_ylabel("hidden units")
        ax = axes[2][j]; ax.hist(d["w"], bins=w_bins); ax.set_yscale("log")
        ax.set_title(f"readout |w_j| — {label}   ||w|| = {d['wnorm']:.4g}", fontsize=10)
        ax.set_xlabel("|w_j|"); ax.set_ylabel("hidden units")
    fig.suptitle(f"{run_dir.name}: weight norms, final vs init", fontsize=11)
    fig.tight_layout()
    out = run_dir / "analysis"; out.mkdir(exist_ok=True)
    fig.savefig(out / "weight_norms_init_vs_final.png", dpi=140); print(out / "weight_norms_init_vs_final.png")
    for label, d in data.items():
        print(f"{label}: emb col norm mean {d['emb'].mean():.4f} (min {d['emb'].min():.4f}, max {d['emb'].max():.4f}) | "
              f"row norm mean {d['rows'].mean():.4f} max {d['rows'].max():.4f} | ||w|| {d['wnorm']:.4g}")


if __name__ == "__main__":
    main()
