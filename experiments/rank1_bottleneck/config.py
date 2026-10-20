from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class Rank1Config:
    # ---- task: inputs are the d singleton modes x_1..x_d plus their product; target = product
    d: int = 16
    # Input modes (Walsh characters) fed to the embedding. A named set:
    #   "singletons+product"  x_1..x_d and prod_i x_i            (d+1 modes, the original setup)
    #   "singletons"          x_1..x_d only
    #   "singletons+pairs"    x_1..x_d and the adjacent pairs x_1x_2, x_3x_4, ...   (d + d/2 modes)
    # or an explicit list of supports as 0-based index lists, e.g. [[0],[1],[0,1]].
    input_modes: str | list = "singletons+product"
    # Deprecated alias: False is the same as input_modes = "singletons".
    include_product_input: bool = True
    # Number of bottleneck directions K: code s = W_enc h in R^K, b = W_dec s.
    bottleneck_k: int = 1
    # False removes the bottleneck entirely: the ReLU layer reads h(x) directly.
    use_bottleneck: bool = True
    # Residual connection around the ReLU layer after the bottleneck:
    #   a(x) = b(x) + ReLU(W b(x))  instead of  a(x) = ReLU(W b(x)),
    # so the readout also sees the bottleneck output directly.
    residual_after_bottleneck: bool = False
    # ---- model
    d_hid: int = 1024
    embedding_variance: float | None = None      # None -> 1 / d_hid
    # Multiplies the initial embedding rows of modes of the given degree, e.g. {1: 3.0}
    # makes every singleton embedding 3x longer than the standard draw (degree-2 rows
    # untouched), i.e. Q_1/Q_2 = 3 in norm.
    embedding_scale_by_degree: dict | None = None
    hidden_variance: float | None = None         # linear layer W (d_hid x d_hid); None -> 1 / d_hid
    readout_variance: float | None = None        # None -> 1 / d_hid
    bottleneck_variance: float | None = None     # w_enc and w_dec entries; None -> 1 / d_hid
    # w_enc initialization:
    #   "random"      N(0, bottleneck_variance) entries
    #   "linear_sum"  aligned with v_1 + ... + v_d (sum of the singleton embeddings)
    #   "product"     aligned with v_{x_1 ... x_d} (embedding of the product mode)
    #   "random_no_product"  random draw with its component along v_prod projected out
    # with cosine `enc_cosine` to that direction (the rest is a random orthogonal component),
    # at the norm a random draw would have, sqrt(d_hid * bottleneck_variance).
    enc_init: Literal["random", "linear_sum", "product", "random_no_product"] = "random"
    enc_cosine: float = 1.0
    # ---- optimization (SGD with momentum, as in the main repo)
    lr: float = 1e-3
    weight_decay: float = 1e-3
    # Weight decay for the readout only (None -> same as weight_decay); every other
    # parameter keeps weight_decay.
    weight_decay_readout: float | None = None
    momentum: float = 0.9
    num_steps: int = 100_000
    batch_size: int = 512
    # ---- evaluation / bookkeeping
    test_samples: int = 65_536                   # full enumeration when d <= 16, else Monte Carlo
    validate_every: int = 1000
    checkpoint_every: int = 10_000
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/rank1_example"

    def __post_init__(self) -> None:
        if self.enc_init not in ("random", "linear_sum", "product", "random_no_product"):
            raise ValueError(f"enc_init must be random, linear_sum, product or random_no_product, got {self.enc_init!r}")
        if not self.include_product_input and self.input_modes == "singletons+product":
            self.input_modes = "singletons"
        if self.bottleneck_k < 1:
            raise ValueError("bottleneck_k must be >= 1")
        self.mode_supports()  # validates

    def readout_wd(self) -> float:
        return self.weight_decay if self.weight_decay_readout is None else self.weight_decay_readout

    def mode_supports(self) -> list[tuple[int, ...]]:
        """The input modes as tuples of 0-based bit indices, in embedding-row order."""
        d = self.d
        if isinstance(self.input_modes, str):
            singles = [(i,) for i in range(d)]
            if self.input_modes == "singletons":
                return singles
            if self.input_modes == "singletons+product":
                return singles + [tuple(range(d))]
            if self.input_modes == "singletons+pairs":
                if d % 2:
                    raise ValueError("singletons+pairs needs even d")
                return singles + [(i, i + 1) for i in range(0, d, 2)]
            raise ValueError(f"unknown input_modes {self.input_modes!r}")
        supports = [tuple(int(i) for i in m) for m in self.input_modes]
        for m in supports:
            if not m or any(i < 0 or i >= d for i in m) or len(set(m)) != len(m):
                raise ValueError(f"bad mode support {m} for d={d}")
        return supports

    def mode_names(self) -> list[str]:
        return ["".join(f"x{i + 1}" for i in m) if len(m) < self.d else "product" for m in self.mode_supports()]

    def product_mode_index(self) -> int | None:
        """Row of the full-product mode in the embedding, or None if it is not an input."""
        full = tuple(range(self.d))
        sup = self.mode_supports()
        return sup.index(full) if full in sup else None
        if not -1.0 <= self.enc_cosine <= 1.0:
            raise ValueError("enc_cosine must be in [-1, 1]")
        if self.d < 1 or self.d_hid < 1:
            raise ValueError("d and d_hid must be positive")

    @property
    def emb_var(self) -> float: return 1.0 / self.d_hid if self.embedding_variance is None else self.embedding_variance
    @property
    def hid_var(self) -> float: return 1.0 / self.d_hid if self.hidden_variance is None else self.hidden_variance
    @property
    def ro_var(self) -> float: return 1.0 / self.d_hid if self.readout_variance is None else self.readout_variance
    @property
    def bn_var(self) -> float: return 1.0 / self.d_hid if self.bottleneck_variance is None else self.bottleneck_variance


def to_dict(cfg: Rank1Config) -> dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: Rank1Config, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))


def load_config(path: str | Path) -> Rank1Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Rank1Config(**raw)
