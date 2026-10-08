"""Configuration dataclass for the backward Walsh-spectrum analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BackwardWalshConfig:
    run: str                          # e.g. "wd3_k0_N2048"
    run_dir: Path
    checkpoint_dir: Path
    output_dir: Path
    # Walsh mode index sets (sorted tuples of input-bit indices)
    target_mode: tuple[int, ...]      # S — the degree-16 target
    child_a: tuple[int, ...]          # A — first  degree-8 child of S
    child_b: tuple[int, ...]          # B — second degree-8 child of S
    # Input dimension info (needed to generate noise-variable controls)
    input_dim: int = 32
    relevant_dim: int = 16
    # Number of alternative (C_k, D_k) partitions of S used as d8 controls
    n_partition_controls: int = 15
    # Degree-16 control modes: number per replacement class (1,2,4 relevant→noise)
    n_d16_controls_per_class: int = 5
    # Dataset / computation
    batch_size: int = 2048
    # star_block: block index l* identified as computing S from A,B.
    # If None it will be auto-detected as the block where H(S) rises the most.
    star_block: int | None = None
