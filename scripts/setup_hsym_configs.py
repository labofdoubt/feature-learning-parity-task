"""Create configs for hierarchical_symmetric tree_parity experiments."""
import os
import yaml
from pathlib import Path

ROOT = Path("/root/parity")

BASE_MODEL = dict(
    input_dim=32,
    relevant_dim=16,
    L=4,
    activation="half-tanh",
    activation_scale=1.0,
    use_readout_barrier=False,
    embedding_weight_variance=0.03125,   # 1/input_dim
    freeze_embedding=False,
    use_layerwise_readouts=False,
    use_post_activation_linear=False,
    use_skip_connections=True,
    bias=False,
    use_attention=False,
    sequence_mixing="attention",
    num_heads=1,
    attention_logit_scale="1/sqrt(d)",
    autoregressive_feedback="raw",
    use_kv_cache=True,
)

BASE_TASK = dict(
    input_dim=32,
    relevant_dim=16,
    exclude_targets=[],
    data_distribution="hierarchical_symmetric",
    eval_noise_repeats=2,
    train_only_root=True,
    task_type="tree_parity",
    num_reuse_targets=3,
)

BASE_TRAINING = dict(
    num_steps=500000,
    test_samples=100000,
    train_samples=100000,
    batch_size=512,
    seed=0,
    device="cuda",
    dtype="float32",
    validate_every=1000,
    progress_every=100,
    checkpoint_every=10000,
    barrier_c=None,
    barrier_lambda=10.0,
    matmul_precision="high",
    teacher_forcing_ratio=1.0,
    curriculum=False,
    curriculum_mse_threshold=0.001,
    optimizer=dict(
        name="sgd",
        lr=0.001,
        lr_embedding=0.008,
        lr_hidden=0.001,
        lr_readout=0.000125,
        weight_decay=0.001,
        wd_embedding=0.000125,
        wd_hidden=0.001,
        wd_readout=0.008,
        momentum=0.9,
        betas=[0.9, 0.999],
    ),
)

EXPERIMENTS = [
    dict(name="hsym_N2048_rho01", N=2048, rho=0.1),
    dict(name="hsym_N2048_rho03", N=2048, rho=0.3),
    dict(name="hsym_N512_rho01",  N=512,  rho=0.1),
    dict(name="hsym_N512_rho03",  N=512,  rho=0.3),
]

for exp in EXPERIMENTS:
    N, rho, name = exp["N"], exp["rho"], exp["name"]
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = dict(
        BASE_MODEL,
        N=N,
        hidden_weight_variance=1.0 / N,
        readout_weight_variance=1.0 / (N * N),
        post_activation_linear_variance=1.0 / N,
    )
    task = dict(BASE_TASK, data_rho=rho)
    training = dict(BASE_TRAINING, output_dir=str(run_dir))

    config = dict(model=model, task=task, training=training)

    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"  wrote {cfg_path}")

print("Done.")
