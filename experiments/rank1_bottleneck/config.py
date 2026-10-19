from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class Rank1Config:
    # ---- task: inputs are the d singleton modes x_1..x_d plus their product; target = product
    d: int = 16
    # ---- model
    d_hid: int = 1024
    embedding_variance: float | None = None      # None -> 1 / d_hid
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
