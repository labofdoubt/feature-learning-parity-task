"""Neuron Walsh Analysis — Streamlit viewer.

Loads precomputed artifacts from <run_dir>/analysis/neuron_walsh/step_*.pt
(created by analysis.neuron_walsh.compute); never reruns the model.

Run:  streamlit run analysis/neuron_walsh/app.py -- --runs-root /root/parity/runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from analysis.neuron_walsh import metrics as M

st.set_page_config(page_title="Neuron Walsh Analysis", layout="wide")


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="/root/parity/runs")
    a, _ = p.parse_known_args()
    return a


ARGS = _args()
RUNS_ROOT = Path(ARGS.runs_root)


@st.cache_data(show_spinner=False, ttl=60)
def list_runs(root: str) -> list[str]:
    root = Path(root)
    return sorted(d.name for d in root.iterdir()
                  if (d / "analysis" / "neuron_walsh").is_dir())


@st.cache_data(show_spinner=False)
def list_steps(root: str, run: str) -> list[int]:
    d = Path(root) / run / "analysis" / "neuron_walsh"
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("step_*.pt"))


@st.cache_data(show_spinner="loading checkpoint artifact...", max_entries=8)
def load_step(root: str, run: str, step: int) -> dict:
    """Lazily load one artifact; tensors -> numpy."""
    p = Path(root) / run / "analysis" / "neuron_walsh" / f"step_{step:08d}.pt"
    raw = torch.load(p, map_location="cpu", weights_only=False)
    out = dict(step=raw["step"], modes=raw["modes"], num_samples=raw["num_samples"],
               locations={})
    for loc in raw["locations"]:
        out["locations"][loc["name"]] = dict(
            C=loc["coefficients"].numpy().astype(np.float64),
            R=loc["correlations"].numpy().astype(np.float64),
            mean=loc["activation_mean"].numpy(),
            var=loc["activation_variance"].numpy().astype(np.float64),
            layer_power=np.asarray(loc["layer_power"], dtype=np.float64),
            block=loc["block"],
        )
    return out


@st.cache_data(show_spinner="scanning steps for y-range...")
def mode_rank_extremes(root: str, run: str, loc_name: str, mode_reg_idx: int,
                       topn: int) -> tuple[float, float]:
    """(ymin, ymax) of the sorted neuron-power curve across ALL saved steps."""
    d = Path(root) / run / "analysis" / "neuron_walsh"
    ymin, ymax = np.inf, 0.0
    for p in sorted(d.glob("step_*.pt")):
        raw = torch.load(p, map_location="cpu", weights_only=False)
        loc = next(l for l in raw["locations"] if l["name"] == loc_name)
        pw = np.sort(loc["coefficients"].numpy()[mode_reg_idx].astype(np.float64) ** 2)[::-1]
        ymax = max(ymax, float(pw[0]))
        ymin = min(ymin, float(pw[min(topn, len(pw)) - 1]))
    return max(ymin, 1e-14), max(ymax, 1e-14)


# ── sidebar: global controls ─────────────────────────────────────────────────

runs = list_runs(str(RUNS_ROOT))
if not runs:
    st.error(f"No runs with analysis/neuron_walsh under {RUNS_ROOT}")
    st.stop()

st.sidebar.title("Neuron Walsh Analysis")
run = st.sidebar.selectbox("Run", runs)
steps = list_steps(str(RUNS_ROOT), run)
step = st.sidebar.select_slider("Training step", steps, value=steps[-1])
data = load_step(str(RUNS_ROOT), run, step)
loc_names = list(data["locations"].keys())
loc_name = st.sidebar.selectbox("Block / nonlinearity", loc_names)

all_degrees = sorted({m["degree"] for m in data["modes"]})
_default_degrees = [d for d in all_degrees if d != 1]  # tree targets only
degrees = st.sidebar.multiselect(
    "Mode degrees (add 1 to include singletons)", all_degrees,
    default=_default_degrees)
rep = st.sidebar.radio("Coefficient representation",
                       ["Power |a|^2", "Raw signed coefficient", "Absolute coefficient",
                        "Fraction of neuron tracked power"],
                       index=0)
order_ref = st.sidebar.radio("Neuron ordering reference",
                             ["fixed neuron index", "current checkpoint",
                              "final checkpoint"], index=0)
min_power = st.sidebar.number_input("Noise floor: min neuron tracked power",
                                    value=1e-8, format="%.1e")

# tracked modes = tree modes filtered by selected degrees (default: all 15)
mode_sel = [m for m in data["modes"] if m["degree"] in degrees]
mode_idx = np.array([m["index"] for m in mode_sel])
mode_names = [m["name"] for m in mode_sel]

loc = data["locations"][loc_name]
C = loc["C"][mode_idx]           # [M, N]
R = loc["R"][mode_idx]
var = loc["var"]
N = C.shape[1]

# neuron metrics on the tracked set
P = C ** 2
tot_i = M.neuron_tracked_power(C)
meff = M.m_eff(C)
pur = M.purity(C)
cov = M.coverage(C, var)
dom = M.dominant_mode(C)
dom_deg = np.array([mode_sel[d]["degree"] for d in dom])
low_power = tot_i < min_power

hp = M.mode_power(C)
neff = M.n_eff(C)

if "sel_mode" not in st.session_state:
    st.session_state.sel_mode = mode_names[-1]
if "sel_neuron" not in st.session_state:
    st.session_state.sel_neuron = int(np.argmax(tot_i))
# apply a scatter-click selection BEFORE the bound widgets are instantiated
if "pending_neuron" in st.session_state:
    st.session_state.sel_neuron = int(st.session_state.pop("pending_neuron"))
# keep selections valid when the degree filter / location changes
if st.session_state.sel_mode not in mode_names:
    st.session_state.sel_mode = mode_names[-1]
if not (0 <= int(st.session_state.sel_neuron) < N):
    st.session_state.sel_neuron = 0


def rep_matrix():
    if rep == "Power |a|^2":
        return P, "power"
    if rep == "Raw signed coefficient":
        return C, "a"
    if rep == "Absolute coefficient":
        return np.abs(C), "|a|"
    return P / (tot_i[None, :] + 1e-30), "p(S|i)"


def neuron_order(kind: str, ref_C: np.ndarray, ref_var: np.ndarray) -> np.ndarray:
    rP = ref_C ** 2
    rtot = rP.sum(0)
    rdom = rP.argmax(0)
    if kind == "original neuron index":
        return np.arange(N)
    if kind == "dominant mode (sorted by shown coefficient)":
        # groups in hierarchy order (d2_0..d2_7, d4_*, d8_*, d16_0); inside a
        # group neurons sort by the MAGNITUDE of their own dominant-mode value
        # in the currently selected coefficient representation
        if rep == "Raw signed coefficient" or rep == "Absolute coefficient":
            V = np.abs(ref_C)
        elif rep == "Fraction of neuron tracked power":
            V = rP / (rtot + 1e-30)
        else:  # Power |a|^2
            V = rP
        vsel = V[rdom, np.arange(N)]
        return np.lexsort((-vsel, rdom))
    if kind == "dominant mode":
        return np.lexsort((-rtot, rdom))
    if kind == "dominant mode (tree only, ignores d1)":
        # dominance and grouping computed over degree>=2 modes only, so adding
        # the d1 rows to the display does not reshuffle the staircase
        tmask = np.array([m["degree"] > 1 for m in mode_sel])
        if not tmask.any():
            return np.lexsort((-rtot, rdom))
        rPt = rP[tmask]
        return np.lexsort((-rPt.sum(0), rPt.argmax(0)))
    if kind == "dominant mode degree":
        rdeg = np.array([mode_sel[d]["degree"] for d in rdom])
        return np.lexsort((-rtot, rdeg))
    if kind == "total tracked power":
        return np.argsort(-rtot)
    if kind == "purity":
        return np.argsort(-(rP / (rtot + 1e-30)).max(0))
    if kind == "M_eff":
        p = rP / (rtot + 1e-30)
        return np.argsort(1.0 / ((p ** 2).sum(0) + 1e-30))
    if kind == "coverage":
        return np.argsort(-(rtot / (ref_var + 1e-30)))
    if kind == "selected mode coefficient":
        mi = mode_names.index(st.session_state.sel_mode)
        return np.argsort(-rP[mi])
    # clustered mode fingerprint: dominant mode, then descending power
    return np.lexsort((-rtot, rdom))


tabs = st.tabs(["Overview", "Mode view", "Neuron view", "Layer population",
                "Mode interactions", "Training dynamics", "Hierarchy"])

# ── Overview: neuron x mode heatmap ──────────────────────────────────────────
with tabs[0]:
    c1, c2, c3 = st.columns([2, 2, 3])
    n_sort = c1.selectbox("Neuron sorting", [
        "original neuron index", "dominant mode (sorted by shown coefficient)",
        "dominant mode", "dominant mode (tree only, ignores d1)",
        "dominant mode degree",
        "total tracked power", "purity", "M_eff", "coverage",
        "selected mode coefficient", "clustered mode fingerprint"])
    m_sort = c2.selectbox("Mode sorting",
                          ["hierarchy", "degree", "total mode power", "N_eff"])
    drop_lowvar = c3.checkbox("remove low-variance neurons", value=False,
                              key="drop_lowvar")
    var_exp = c3.slider("variance threshold: 10^", -6.0, 0.0, -4.0, step=0.1,
                        key="var_exp", disabled=not drop_lowvar)

    # ordering reference: which checkpoint supplies the sort keys.
    # "fixed neuron index" + a sort uses FINAL-checkpoint keys so the order
    # stays stable while scrubbing the step slider.
    if order_ref == "current checkpoint":
        ref, ref_var = C, var
    else:  # "final checkpoint", or "fixed neuron index" with a sort chosen
        fin = load_step(str(RUNS_ROOT), run, steps[-1])["locations"][loc_name]
        ref, ref_var = fin["C"][mode_idx], fin["var"]
    nord = (np.arange(N) if n_sort == "original neuron index"
            else neuron_order(n_sort, ref, ref_var))
    if drop_lowvar:
        keep_n = var[nord] >= 10.0 ** var_exp
        nord = nord[keep_n]
        if len(nord) == 0:
            st.warning("threshold removes every neuron — lower it")
            st.stop()

    if m_sort == "hierarchy":
        mord = np.arange(len(mode_sel))
    elif m_sort == "degree":
        mord = np.argsort([m["degree"] for m in mode_sel], kind="stable")
    elif m_sort == "total mode power":
        mord = np.argsort(-hp)
    else:
        mord = np.argsort(neff)

    Z, zlabel = rep_matrix()
    Z = Z[np.ix_(mord, nord)]
    # x must be POSITIONAL: with numeric x=nord, plotly places columns at the
    # coordinate value, silently undoing any neuron permutation.
    hover = ("neuron %{customdata[2]:.0f}<br>mode %{y}<br>" + zlabel + " %{z:.3e}<br>"
             "purity %{customdata[0]:.2f} · coverage %{customdata[1]:.2f} · "
             "M_eff %{customdata[3]:.2f}")
    cd = np.stack([pur[nord], cov[nord], nord.astype(float), meff[nord]],
                  axis=-1)[None, :, :].repeat(len(mord), 0)
    fig = go.Figure(go.Heatmap(
        z=Z, y=[mode_names[i] for i in mord], x=np.arange(len(nord)),
        customdata=cd, hovertemplate=hover, colorscale="Viridis",
        colorbar=dict(title=zlabel)))
    fig.update_layout(height=max(420, 90 + 24 * len(mode_sel)),
                      margin=dict(l=10, r=10, t=25, b=10),
                      xaxis_title="neuron (sorted position — hover for id)",
                      yaxis_title="mode")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{loc_name} @ step {step} — {len(mode_sel)} tracked tree modes × "
               f"{len(nord)}/{N} neurons shown"
               + (f" (σ² ≥ 10^{var_exp:.1f} filter removed {N - len(nord)})" if drop_lowvar else "") + ". "
               f"Sanity: layer power H(S)=Σ_i a² holds by construction; see compute log for "
               f"independent checks.")

# ── Mode view ────────────────────────────────────────────────────────────────
with tabs[1]:
    sel_mode = st.selectbox("Mode", mode_names, key="sel_mode")
    mi = mode_names.index(sel_mode)
    md = mode_sel[mi]
    p_sorted = np.sort(P[mi])[::-1]
    order = np.argsort(-P[mi])
    cum = M.cumulative_power(C, mi)
    cols = st.columns(2)
    cols[0].metric("H(S)", f"{hp[mi]:.3e}")
    cols[1].metric("N_eff", f"{neff[mi]:.1f}")
    st.caption(f"support={md['support']} degree={md['degree']} "
               f"parent={md['parent']} children={md['children']}")
    c1, c2 = st.columns(2)
    fix_y = c1.checkbox("fix y-scale across steps", value=False, key="fix_y_rank")
    f1 = go.Figure(go.Scatter(y=p_sorted[:512], mode="lines"))
    f1.update_layout(title="neuron power a² vs rank (top 512)", height=300,
                     yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10))
    if fix_y:
        ymin, ymax = mode_rank_extremes(str(RUNS_ROOT), run, loc_name,
                                        int(mode_idx[mi]), 512)
        f1.update_yaxes(range=[np.log10(ymin * 0.8), np.log10(ymax * 1.5)])
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure(go.Scatter(x=np.arange(1, len(cum) + 1), y=cum, mode="lines"))
    f2.update_layout(title="cumulative power C_S(k)", height=300,
                     xaxis_type="log", xaxis_title="k (neurons)",
                     margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(f2, use_container_width=True)
    topn = st.slider("signed coefficients of strongest neurons", 5, 50, 20)
    f3 = go.Figure(go.Bar(x=[str(i) for i in order[:topn]], y=C[mi, order[:topn]],
                          marker_color=np.where(C[mi, order[:topn]] >= 0,
                                                "steelblue", "indianred")))
    f3.update_layout(title="signed a_iS, top neurons", height=280,
                     xaxis_title="neuron", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(f3, use_container_width=True)

# ── Neuron view ──────────────────────────────────────────────────────────────
with tabs[2]:
    ni = int(st.number_input("Neuron", 0, N - 1, key="sel_neuron"))
    cols = st.columns(5)
    flags = " ⚠ insufficient tracked-mode power" if low_power[ni] else ""
    cols[0].metric("M_eff", f"{meff[ni]:.2f}")
    cols[1].metric("purity", f"{pur[ni]:.2f}")
    cols[2].metric("coverage", f"{cov[ni]:.3f}")
    cols[3].metric("tracked power", f"{tot_i[ni]:.3e}")
    cols[4].metric("variance", f"{var[ni]:.3e}")
    if flags:
        st.warning("Below noise floor — semanticity metrics unreliable." + flags)
    show = st.radio("Modes shown", ["top N", "all tracked"], horizontal=True)
    topn = st.slider("N", 5, len(mode_sel), min(15, len(mode_sel))) \
        if show == "top N" else len(mode_sel)
    order = np.argsort(-P[:, ni])[:topn]
    frac = P[:, ni] / (tot_i[ni] + 1e-30)
    f = go.Figure(go.Bar(
        y=[f"{mode_names[j]} (d{mode_sel[j]['degree']})" for j in order][::-1],
        x=P[order, ni][::-1], orientation="h",
        customdata=np.stack([C[order, ni], frac[order], R[order, ni]], -1)[::-1],
        hovertemplate="power %{x:.3e}<br>a=%{customdata[0]:.3e}<br>"
                      "frac=%{customdata[1]:.2f}<br>corr=%{customdata[2]:.3f}"))
    f.update_layout(title=f"neuron {ni}: mode spectrum (power)", height=420,
                    margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(f, use_container_width=True)

# ── Layer population ─────────────────────────────────────────────────────────
with tabs[3]:
    ax_opts = {"M_eff": meff, "purity": pur, "coverage": cov,
               "total tracked power": tot_i,
               "max |correlation|": np.abs(R).max(0),
               "dominant mode degree": dom_deg.astype(float)}
    c1, c2, c3 = st.columns(3)
    xk = c1.selectbox("x", list(ax_opts), index=0)
    yk = c2.selectbox("y", list(ax_opts), index=2)
    minc = c3.number_input("min coverage filter (histograms)", 0.0, 1.0, 0.0)
    keep = ~low_power
    fig = go.Figure(go.Scattergl(
        x=ax_opts[xk][keep], y=ax_opts[yk][keep], mode="markers",
        marker=dict(size=4 + 20 * (tot_i[keep] / (tot_i.max() + 1e-30)),
                    opacity=0.5),
        customdata=np.stack([np.arange(N)[keep], tot_i[keep]], -1),
        hovertemplate="neuron %{customdata[0]}<br>x=%{x:.3f} y=%{y:.3f}"
                      "<br>tracked power %{customdata[1]:.2e}"))
    fig.update_layout(height=420, xaxis_title=xk, yaxis_title=yk,
                      margin=dict(l=10, r=10, t=25, b=10))
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                         key="pop_scatter")
    try:
        pts = ev.selection.points
        if pts:
            st.session_state.pending_neuron = int(pts[0]["customdata"][0])
            st.info(f"selected neuron {st.session_state.pending_neuron} — see Neuron view")
            st.rerun()
    except Exception:
        pass
    hsel = keep & (cov >= minc)
    c1, c2, c3, c4 = st.columns(4)
    for col, (name, vals) in zip(
            (c1, c2, c3),
            [("M_eff", meff[hsel]), ("purity", pur[hsel]), ("coverage", cov[hsel])]):
        h = go.Figure(go.Histogram(x=vals, nbinsx=50))
        h.update_layout(title=name, height=250, margin=dict(l=10, r=10, t=30, b=10))
        col.plotly_chart(h, use_container_width=True)
    degs, cnts = np.unique(dom_deg[hsel], return_counts=True)
    h = go.Figure(go.Bar(x=[f"d{d}" for d in degs], y=cnts))
    h.update_layout(title="dominant mode degree", height=250,
                    margin=dict(l=10, r=10, t=30, b=10))
    c4.plotly_chart(h, use_container_width=True)

# ── Mode interactions ────────────────────────────────────────────────────────
with tabs[4]:
    signed = st.toggle("Signed coefficient alignment (default: power co-localization)")
    K = M.mode_colocalization(C, signed=signed)
    f = go.Figure(go.Heatmap(z=K, x=mode_names, y=mode_names,
                             colorscale="RdBu_r" if signed else "Viridis",
                             zmin=-1 if signed else 0, zmax=1))
    f.update_layout(height=max(520, 120 + 20 * len(mode_names)),
                    title="Signed coefficient alignment" if signed
                    else "Power co-localization  K_ST",
                    margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(f, use_container_width=True)

# ── Training dynamics ────────────────────────────────────────────────────────
with tabs[5]:
    sel_mode = st.selectbox("Mode (trajectory)", mode_names,
                            index=mode_names.index(st.session_state.sel_mode),
                            key="traj_mode")
    ni = st.number_input("Neuron (trajectory)", 0, N - 1,
                         value=int(st.session_state.sel_neuron), key="traj_neuron")
    Hs, Ns, t1, nmet = [], [], [], {"top power": [], "M_eff": [], "purity": [],
                                    "coverage": []}
    for s in steps:
        d = load_step(str(RUNS_ROOT), run, s)["locations"][loc_name]
        Cs = d["C"][mode_idx]
        mi = mode_names.index(sel_mode)
        Hs.append(M.mode_power(Cs)[mi])
        Ns.append(M.n_eff(Cs)[mi])
        t1.append(M.topk_fractions(Cs, (1,))[1][mi])
        Ps = Cs ** 2
        nmet["top power"].append(Ps[:, ni].max())
        nmet["M_eff"].append(M.m_eff(Cs)[ni])
        nmet["purity"].append(M.purity(Cs)[ni])
        nmet["coverage"].append(M.coverage(Cs, d["var"])[ni])
    c1, c2 = st.columns(2)
    f1 = go.Figure()
    f1.add_scatter(x=steps, y=Hs, name="H(S,t)")
    f1.add_scatter(x=steps, y=Ns, name="N_eff(S,t)", yaxis="y2")
    f1.add_scatter(x=steps, y=t1, name="top-1 frac", yaxis="y2")
    f1.update_layout(title=f"mode {sel_mode}", height=350, xaxis_type="log",
                     yaxis=dict(title="H", type="log"),
                     yaxis2=dict(overlaying="y", side="right", title="N_eff / frac"),
                     margin=dict(l=10, r=10, t=30, b=10))
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure()
    for k, v in nmet.items():
        f2.add_scatter(x=steps, y=v, name=k)
    f2.update_layout(title=f"neuron {ni}", height=350, xaxis_type="log",
                     yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(f2, use_container_width=True)

# ── Hierarchy ────────────────────────────────────────────────────────────────
with tabs[6]:
    xs, ys, sizes, texts, hovers = [], [], [], [], []
    ex, ey = [], []
    pos = {}
    for m in mode_sel:
        lvl = m["level"]
        siblings = [q for q in mode_sel if q["degree"] == m["degree"]]
        j = siblings.index(m)
        xpos = (j + 0.5) / len(siblings)
        pos[m["name"]] = (xpos, lvl)
    for m in mode_sel:
        x0, y0 = pos[m["name"]]
        if m["parent"] in pos:
            x1, y1 = pos[m["parent"]]
            ex += [x0, x1, None]; ey += [y0, y1, None]
    fe = go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="gray", width=1),
                    hoverinfo="skip")
    mi_by_name = {n: i for i, n in enumerate(mode_names)}
    for m in mode_sel:
        i = mi_by_name[m["name"]]
        x0, y0 = pos[m["name"]]
        xs.append(x0); ys.append(y0)
        sizes.append(12 + 40 * hp[i] / (hp.max() + 1e-30))
        texts.append(m["name"])
        hovers.append(f"{m['name']} d{m['degree']}<br>H={hp[i]:.3e}"
                      f"<br>N_eff={neff[i]:.1f}<br>top1={M.topk_fractions(C,(1,))[1][i]:.2f}")
    fn = go.Scatter(x=xs, y=ys, mode="markers+text", text=texts,
                    textposition="bottom center",
                    marker=dict(size=sizes, color=neff, colorscale="Viridis",
                                colorbar=dict(title="N_eff")),
                    hovertext=hovers, hoverinfo="text")
    f = go.Figure([fe, fn])
    f.update_layout(height=420, title="tree: size=H(S), color=N_eff(S)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(f, use_container_width=True)

    rows = []
    for ln in loc_names:
        d = data["locations"][ln]
        Cl = d["C"][mode_idx]
        mi = mode_names.index(st.session_state.sel_mode)
        tk = M.topk_fractions(Cl, (1, 10))
        rows.append(dict(location=ln, H=M.mode_power(Cl)[mi],
                         N_eff=M.n_eff(Cl)[mi],
                         top1=tk[1][mi], top10=tk[10][mi]))
    st.caption(f"mode {st.session_state.sel_mode} across locations")
    st.dataframe(rows, use_container_width=True)
