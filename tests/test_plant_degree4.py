import json
import math

import pytest

torch = pytest.importorskip("torch")

from experiments.rank1_bottleneck.config import Rank1Config
from experiments.rank1_bottleneck.model import Rank1BottleneckNet
from experiments.rank1_bottleneck.plant_degree4 import Setup, forward_parts

MODES = [[0], [1], [2], [3], [4], [5], [6], [7], [0, 1, 2, 3], [4, 5, 6, 7]]


def _make_source(tmp_path, d_hid=48):
    cfg = Rank1Config(d=8, d_hid=d_hid, input_modes=MODES, use_bottleneck=False, weight_decay=0.01,
                      embedding_scale_by_degree={1: 8.0}, output_dir=str(tmp_path))
    torch.manual_seed(3)
    model = Rank1BottleneckNet(cfg)
    with torch.no_grad():                       # make a few units exactly silent
        model.readout.weight[0, :6] = 0.0
        model.linear.weight[:6] = 0.0
    path = tmp_path / "final.pt"
    torch.save(dict(step=50_000, model_state=model.state_dict(), config=cfg.__dict__), path)
    return path


def test_construction_identities(tmp_path):
    src = _make_source(tmp_path)
    s = Setup(src, M=4, intervention_seed=11)
    assert s.ip == 8 and s.iq == 9 and s.singles == list(range(8))
    assert s.ortho_checks["VU_max"] < 1e-10 and s.ortho_checks["UtU_err"] < 1e-12
    assert s.selection["yJ_rms"] < 1e-12                      # silent units chosen
    assert s.prep_checks["preact_change_nonplanted_max"] < 1e-10
    assert s.prep_checks["output_identity_err"] < 1e-10
    for rho in (0.15, 0.3):
        v = s.validate(rho)
        f = v["f64"]
        assert f["nonplanted_preact_err"] < 1e-10
        assert f["planted_preact_err"] < 1e-12
        assert f["output_err"] < 1e-12                       # y_rho - y_base = rho^3/4 chi_8
        assert f["other_walsh_max"] < 1e-12
        assert abs(f["sqnorm_increase"] - 3 * rho ** 2) < 1e-10
        assert f["dL_err"] < 1e-10
        g = v["gradient_check"]
        assert g["err_V"] < 1e-9 and g["err_W"] < 1e-9 and g["err_r"] < 1e-9
        assert abs(v["orth_control"]["injected_target_coef"]) < 1e-12
        assert abs(v["orth_control"]["readout_norm_J"] - v["orth_control"]["coherent_readout_norm_J"]) < 1e-12
        t = v["train_dtype"]
        assert t["output_err"] < 1e-4 and t["dL_err"] < 1e-4
    # templates have unit Frobenius norm and projections equal rho right after planting
    assert abs(float(s.TV.norm()) - 1) < 1e-12 and abs(float(s.TW.norm()) - 1) < 1e-12 and abs(float(s.tr.norm()) - 1) < 1e-12
    V, W, r = s.plant(0.28)
    m = s.measure(V, W, r, s.J)
    assert abs(m["b_par"] - 0.28) < 1e-12 and abs(m["w_par"] - 0.28) < 1e-12 and abs(m["r_par"] - 0.28) < 1e-12
    assert abs(m["a_planted"] - 0.28 ** 3 / 4) < 1e-12 and abs(m["a_planted_minus_pred"]) < 1e-12
    json.dumps(s.metadata())                                  # serializable


def test_replicated_units(tmp_path):
    src = _make_source(tmp_path, d_hid=64)
    with torch.no_grad():
        pass
    s = Setup(src, M=8, intervention_seed=5)
    assert len(s.J) == 8 and s.S.shape == (8, 2) and abs(float(s.S.sum())) < 1e-12
    f = s.validate(0.3)["f64"]
    assert f["output_err"] < 1e-12 and abs(f["sqnorm_increase"] - 3 * 0.09) < 1e-10


def test_arm_tensors_and_short_training(tmp_path):
    from experiments.rank1_bottleneck.plant_degree4 import train_arm
    src = _make_source(tmp_path)
    s = Setup(src, M=4, intervention_seed=11)
    V, W, r, rho = s.arm_tensors("orth0.3")
    assert rho == 0.3 and abs(float(r[s.J].norm()) - 0.3) < 1e-12
    out = tmp_path / "arm"
    train_arm(s, "rho0.3", out, steps=12, data_seed=1, device=torch.device("cpu"))
    assert (out / "analysis" / "trajectory.csv").exists() and (out / "checkpoints" / "final.pt").exists()
    meta = json.loads((out / "analysis" / "intervention.json").read_text())
    assert meta["validation"]["f64"]["output_err"] < 1e-12
    fin = json.loads((out / "analysis" / "final.json").read_text())
    assert set(fin["ablation"]) == {"only_degree4", "only_singletons"}
