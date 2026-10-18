from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 32
    relevant_dim: int = 16
    N: int = 1024
    L: int = 4
    activation: Literal["relu", "gelu", "tanh", "silu", "half-tanh", "square"] = "silu"
    # Fixed, non-learnable multiplier c on the activation: phi(x) -> c * phi(x).
    activation_scale: float = 1.0
    use_readout_barrier: bool = True
    embedding_weight_variance: float | None = None
    freeze_embedding: bool = True
    hidden_weight_variance: float = 1.0
    readout_weight_variance: float = 1e-4
    use_layerwise_readouts: bool = False
    use_post_activation_linear: bool = False
    # False removes every skip connection, turning the stack into a plain deep MLP.
    use_skip_connections: bool = True
    # None means the post-activation linear reuses hidden_weight_variance.
    post_activation_linear_variance: float | None = None
    bias: bool = False
    use_attention: bool = False
    sequence_mixing: Literal["attention", "uniform", "uniform_vo", "none"] = "attention"
    num_heads: int = 1
    attention_logit_scale: Literal["1/sqrt(d)", "1/d"] = "1/sqrt(d)"
    autoregressive_feedback: Literal["raw", "sign"] = "raw"
    use_kv_cache: bool = True
    # Optional sparse bottleneck on the residual stream. When enabled, every block
    # ends with  h -> W_dec TopK(W_enc h)  applied right after its residual addition
    # (under use_attention only the MLP residual addition gets one), so there are
    # exactly L bottlenecks, one per nonlinearity.
    use_bottleneck: bool = False
    # Width of the bottleneck code (d_bottleneck). Required when use_bottleneck.
    bottleneck_dim: int | None = None
    # Number of active code entries per sample: one int shared by every bottleneck,
    # or a list of L ints, one per block. Required when use_bottleneck.
    bottleneck_k: int | list[int] | None = None
    # True keeps the K largest |code| entries (either sign); False keeps the K
    # largest signed entries, with no ReLU applied.
    bottleneck_abs_topk: bool = True
    # Per-entry init variances of the encoder (N -> d_bottleneck) and decoder
    # (d_bottleneck -> N) weights. None means 1/N for the encoder and 1/K for the
    # decoder (per block, using that block's K), which keeps the residual-stream
    # scale roughly unchanged at init.
    bottleneck_encoder_weight_variance: float | None = None
    bottleneck_decoder_weight_variance: float | None = None
    # Whether the encoder and decoder linears carry biases (independent of `bias`).
    bottleneck_bias: bool = False
    # Magnitude-direction decoupling (arXiv:2606.25971) for the bottleneck encoder
    # and decoder only. Each matrix is W = diag(g_row) W_hat diag(g_col) with the
    # direction W_hat held on a fixed Frobenius sphere (radius = its init norm,
    # projected back after every optimizer step) and learnable per-row/per-column
    # softplus gains that start at 1. Its own init overrides the bottleneck
    # variances: W_hat ~ N(0, 1/N) entrywise, then scaled to exactly
    # c_F = sqrt(d_out d_in / N). The bottleneck group trains without weight decay
    # (the sphere is the regularizer); wd_bottleneck must stay unset / 0.
    bottleneck_md_decouple: bool = False
    # How the K passing code entries are chosen.
    #   "activation"  per input, the K largest (|z| or z) encoder outputs: an
    #                 input-dependent gate that can mix Walsh modes.
    #   "scores"      a learnable score vector s (one scalar per code entry, no
    #                 input dependence): the K largest (|s| or s) entries pass on
    #                 every input and are multiplied by their score, so the
    #                 bottleneck is a pure filter W_dec diag(m) W_enc with exactly
    #                 K nonzero gains m_j = s_j.
    bottleneck_selection: Literal["activation", "scores"] = "activation"
    # Gradient regime for the scores (bottleneck_selection="scores" only):
    #   "hard"         exact gradient: only the K selected scores learn (through
    #                  the value they multiply); the selection changes only when
    #                  a selected score is overtaken.
    #   "laplace_ste"  straight-through: hard mask in the forward, and in the
    #                  backward the selection indicator 1[r_j >= theta] (theta =
    #                  midpoint between the K-th and (K+1)-th ranked score) is
    #                  differentiated as the Laplace CDF with scale
    #                  bottleneck_score_temperature, i.e. every score within a
    #                  few temperatures of the threshold gets gradient with weight
    #                  exp(-|r_j - theta| / tau) / (2 tau).
    #   "gumbel"       Gumbel-TopK: in training mode the ranks are perturbed once
    #                  per forward by tau * Gumbel(0, 1) noise (shared across the
    #                  batch, so the mask is still input-independent) before the
    #                  hard top-K, and the kept scores get the exact gradient as
    #                  in "hard". The noise lets currently dropped channels be
    #                  selected and trained; in eval mode the selection is the
    #                  deterministic top-K.
    #   "gumbel_reinforce"  "gumbel" plus a REINFORCE (score-function) term on the
    #                  selection: the Gumbel-top-K draw is a Plackett-Luce sample
    #                  with logits r/tau, and the training loop adds
    #                  reinforce_scale * (loss - baseline) * log p(support) to the
    #                  objective (baseline = EMA of the loss, decay 0.99), so every
    #                  score, kept or not, receives the policy-gradient estimate of
    #                  d E[loss] / d s on top of the exact gain gradient.
    bottleneck_score_grad: Literal["hard", "laplace_ste", "gumbel", "gumbel_reinforce"] = "hard"
    # Multiplier on the REINFORCE term (gumbel_reinforce only). Note d log p / d s
    # is O(1/tau), so with small tau this term is large; scale it down if the
    # scores jitter.
    bottleneck_reinforce_scale: float = 1.0
    # tau, in score units (scores start at 1 +- init std): the Laplace surrogate's
    # scale under laplace_ste, the Gumbel noise magnitude under gumbel.
    bottleneck_score_temperature: float = 0.01
    # Scores start at 1 + N(0, std^2): gains ~1 like a 0/1 mask, random ranking.
    bottleneck_score_init_std: float = 0.01
    # True: readouts (final and layerwise) read the residual stream right after the
    # residual addition, before that block's bottleneck; the bottleneck output only
    # feeds the next block, so the last bottleneck is inert. False: readouts see the
    # post-bottleneck stream. Residual-net only.
    readout_before_bottleneck: bool = False

    def bottleneck_ks(self) -> list[int]:
        """Per-block K list, validating the bottleneck fields. Empty when disabled."""
        if not self.use_bottleneck:
            return []
        if self.bottleneck_dim is None or self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be a positive int when use_bottleneck=True")
        if self.bottleneck_k is None:
            raise ValueError("bottleneck_k must be set when use_bottleneck=True")
        ks = [self.bottleneck_k] * self.L if isinstance(self.bottleneck_k, int) else list(self.bottleneck_k)
        if len(ks) != self.L:
            raise ValueError(
                f"bottleneck_k must be an int or a list of L={self.L} ints, got {self.bottleneck_k!r}"
            )
        for k in ks:
            if not isinstance(k, int) or k <= 0 or k > self.bottleneck_dim:
                raise ValueError(
                    f"each bottleneck_k must be an int in [1, bottleneck_dim={self.bottleneck_dim}], got {k!r}"
                )
        for name in ("bottleneck_encoder_weight_variance", "bottleneck_decoder_weight_variance"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
            if value is not None and self.bottleneck_md_decouple:
                raise ValueError(
                    f"{name} has no effect under bottleneck_md_decouple=True (the MD init "
                    "draws N(0, 1/N) and projects to the c_F sphere); leave it null")
        if self.bottleneck_selection not in ("activation", "scores"):
            raise ValueError(f"bottleneck_selection must be 'activation' or 'scores', got {self.bottleneck_selection!r}")
        if self.bottleneck_score_grad not in ("hard", "laplace_ste", "gumbel", "gumbel_reinforce"):
            raise ValueError(
                "bottleneck_score_grad must be 'hard', 'laplace_ste', 'gumbel' or 'gumbel_reinforce', "
                f"got {self.bottleneck_score_grad!r}")
        if self.bottleneck_score_grad in ("laplace_ste", "gumbel", "gumbel_reinforce") \
                and self.bottleneck_score_temperature <= 0:
            raise ValueError(f"bottleneck_score_temperature must be positive for {self.bottleneck_score_grad}")
        if self.bottleneck_score_init_std < 0:
            raise ValueError("bottleneck_score_init_std must be non-negative")
        return ks


@dataclass
class TaskConfig:
    input_dim: int = 32
    relevant_dim: int = 16
    exclude_targets: list[str] = field(default_factory=list)
    # Hierarchical non-uniform distribution parameter (0 = uniform).
    data_rho: float = 0.0
    # Which data distribution to use when data_rho > 0.
    #   "hierarchical"                   – asymmetric biased tree all the way to individual bits.
    #   "hierarchical_degree2"           – asymmetric biased tree stops at degree-2 latents;
    #                                      each z_i expanded uniformly, so E[S*x_j]=0.
    #   "hierarchical_symmetric"         – symmetric biased tree all the way to bits; all
    #                                      nodes at the same depth are statistically equivalent;
    #                                      E[S*y_d] = (rho/2)^log2(k/d).
    #   "hierarchical_symmetric_degree2" – symmetric biased tree stops at degree-2 latents;
    #                                      each z_i expanded uniformly, so E[S*x_j]=0.
    #   "walsh_single_mode"              – p(x) = 2^{-d} (1 + rho * chi_A(x)) with a single
    #                                      Walsh mode A given by data_walsh_mode; the only
    #                                      nonzero moment is E[chi_A] = rho. Allows rho in
    #                                      [-1, 1] (hierarchical variants require [0, 1)).
    #   "walsh_clusters"                 – latent mixture of single-Walsh-mode clusters from
    #                                      data_walsh_clusters; the marginal is
    #                                      2^{-d}[1 + sum_j alpha_j rho_j chi_{A_j}(x)].
    #                                      Per-cluster rho values apply; data_rho is ignored.
    #   "uniform"                        – always sample uniformly regardless of data_rho.
    # Default is "hierarchical" so that existing configs continue to behave as before.
    data_distribution: str = "hierarchical"
    # Walsh support A for walsh_single_mode, as ZERO-BASED input coordinates
    # (e.g. [2, 3, 4, 5] means chi_A = x_2*x_3*x_4*x_5 in 0-based / x3*x4*x5*x6 in
    # 1-based notation). Required nonempty for walsh_single_mode; ignored otherwise.
    data_walsh_mode: list[int] | None = None
    # Cluster list for walsh_clusters: each entry is {mode: [zero-based indices],
    # rho: float in [-1, 1], weight: float >= 0}. Weights are relative and
    # normalized internally (alpha_j = w_j / sum w); at least one must be positive.
    # Modes may overlap or repeat (duplicate supports add their alpha_j*rho_j).
    # Required nonempty for walsh_clusters; ignored otherwise.
    data_walsh_clusters: list[dict] | None = None
    # For uniform exhaustive eval: independent irrelevant-bit draws per relevant config.
    eval_noise_repeats: int = 2
    # When True, training loss covers only the root parity (degree relevant_dim).
    # All lower-degree auxiliary targets are excluded from the loss even if they appear
    # in exclude_targets; this makes non-uniform experiments explicit and safe.
    train_only_root: bool = False
    # Reuse-star task parameters (only used when task_type == "reuse_star").
    #   task_type        – "tree_parity" (default) or "reuse_star"
    #   num_reuse_targets– m in {1,2,3}: how many of [S1,S2,S3] are active in the loss
    task_type: str = "tree_parity"
    num_reuse_targets: int = 3


@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adamw"] = "adamw"
    lr: float = 1e-3
    lr_embedding: float | None = None
    lr_hidden: float | None = None
    lr_readout: float | None = None
    # Bottleneck encoder/decoder weights form their own group (not "hidden").
    lr_bottleneck: float | None = None
    weight_decay: float = 0.0
    wd_embedding: float | None = None
    wd_hidden: float | None = None
    wd_readout: float | None = None
    wd_bottleneck: float | None = None
    momentum: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class TrainingConfig:
    num_steps: int = 10_000
    test_samples: int = 20_000
    train_samples: int | None = None
    batch_size: int = 512
    seed: int = 0
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    validate_every: int = 100
    progress_every: int = 0
    checkpoint_every: int = 1_000
    # When true, checkpoints follow a logarithmic step grid instead of
    # checkpoint_every: {1..9}x10, {1..9}x100, ... up to num_steps (num_steps
    # always included). Useful for analyses that need dense early-training
    # coverage. checkpoint_every is ignored in this mode.
    checkpoint_log_scale: bool = False
    output_dir: str = "runs/parity"
    barrier_c: float | None = None
    barrier_lambda: float = 10.0
    matmul_precision: Literal["highest", "high", "medium"] = "highest"
    teacher_forcing_ratio: float = 1.0
    curriculum: bool = False
    curriculum_mse_threshold: float = 0.01
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # Re-draw the readout weights from their own RNG seed right after the model is
    # built (every other weight and the data stay bit-identical to `seed`).
    # readout_seed_keys restricts it to given layerwise-readout keys (block indices
    # as strings, e.g. ["1"] = the readout after block 1); None re-draws all readouts.
    readout_seed: int | None = None
    readout_seed_keys: list[str] | None = None
    # Likewise re-draw every bottleneck's encoder and decoder weights (MD direction
    # or plain weights, with their configured init) from this seed; selection
    # scores and everything else stay bit-identical to `seed`.
    bottleneck_seed: int | None = None
    # Residual blocks (0-based) whose parameters, bottleneck included, are frozen for
    # this training stage: requires_grad=False, so they are left out of the optimizer.
    # Resume with a different list to unfreeze; the optimizer state is then rebuilt.
    freeze_blocks: list[int] | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        default_task = TaskConfig()
        if self.task == default_task and (
            self.model.input_dim != default_task.input_dim
            or self.model.relevant_dim != default_task.relevant_dim
        ):
            self.task = TaskConfig(
                input_dim=self.model.input_dim,
                relevant_dim=self.model.relevant_dim,
            )
        else:
            self.model.input_dim = self.task.input_dim
            self.model.relevant_dim = self.task.relevant_dim


def normalize_training_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept training sections written before `log_every` became `validate_every`.

    Old run configs and saved checkpoints still carry `log_every`, so it is mapped
    across rather than rejected.
    """
    raw = dict(raw)
    if "log_every" in raw:
        legacy = raw.pop("log_every")
        raw.setdefault("validate_every", legacy)
    return raw


def to_dict(config: Any) -> dict[str, Any]:
    data = asdict(config)
    if "training" in data:
        data["training"]["optimizer"]["betas"] = list(data["training"]["optimizer"]["betas"])
    return data


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(to_dict(config), f, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    task_raw = raw.pop("task", None)
    model_raw = raw["model"]
    if task_raw is None:
        task = TaskConfig(
            input_dim=model_raw.get("input_dim", 32),
            relevant_dim=model_raw.get("relevant_dim", 16),
        )
    else:
        task = TaskConfig(**task_raw)
        model_raw["input_dim"] = task.input_dim
        model_raw["relevant_dim"] = task.relevant_dim
    training_raw = normalize_training_raw(raw["training"])
    opt = OptimizerConfig(**training_raw.pop("optimizer"))
    if isinstance(opt.betas, list):
        opt.betas = tuple(opt.betas)
    return ExperimentConfig(
        model=ModelConfig(**model_raw),
        task=task,
        training=TrainingConfig(optimizer=opt, **training_raw),
    )


def write_default_config(path: str | Path) -> None:
    save_config(ExperimentConfig(), path)
