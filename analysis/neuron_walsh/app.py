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
    # a run is hidden from the app (not deleted) by `touch <run>/.hide_from_streamlit`
    names = sorted(d.name for d in root.iterdir()
                   if (d / "analysis" / "neuron_walsh").is_dir()
                   and not (d / ".hide_from_streamlit").exists())
    # optional <runs-root>/.streamlit_order: one run name per line, listed runs
    # come first in that order, the rest keep alphabetical order
    order_file = root / ".streamlit_order"
    if order_file.exists():
        wanted = [l.strip() for l in order_file.read_text().splitlines() if l.strip()]
        rank = {n: i for i, n in enumerate(wanted)}
        names.sort(key=lambda n: (rank.get(n, len(rank)), n))
    return names


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


@st.cache_data(show_spinner="loading input-Walsh artifact...", max_entries=8)
def load_input_walsh(root: str, run: str, step: int) -> dict | None:
    """Optional exact pre-activation spectrum (analysis.neuron_walsh.input_walsh)."""
    p = (Path(root) / run / "analysis" / "neuron_walsh" / "input_walsh"
         / f"step_{step:08d}.pt")
    if not p.exists():
        return None
    raw = torch.load(p, map_location="cpu", weights_only=False)
    blocks = {}
    for b in raw["blocks"]:
        blocks[b["block"]] = {k: (v.numpy() if hasattr(v, "numpy") else v)
                              for k, v in b.items()}
    return dict(step=raw["step"], relevant_dim=raw["relevant_dim"],
                repeats=raw["repeats"], topk=raw["topk"], modes=raw["modes"],
                blocks=blocks)


@st.cache_data(show_spinner="loading bottleneck-Walsh artifact...", max_entries=8)
def load_bottleneck_walsh(root: str, run: str, step: int) -> dict | None:
    """Optional bottleneck code analysis (analysis.neuron_walsh.bottleneck_walsh)."""
    p = (Path(root) / run / "analysis" / "neuron_walsh" / "bottleneck_walsh"
         / f"step_{step:08d}.pt")
    if not p.exists():
        return None
    raw = torch.load(p, map_location="cpu", weights_only=False)
    bns = {}
    for b in raw["bottlenecks"]:
        bns[b["block"]] = {k: (v.numpy().astype(np.float64) if hasattr(v, "numpy") else v)
                           for k, v in b.items()}
    return dict(step=raw["step"], num_samples=raw["num_samples"], modes=raw["modes"],
                bottlenecks=bns)


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
                             ["current checkpoint", "final checkpoint",
                              "fixed neuron index"], index=0,
                             help="which checkpoint's coefficients define the grouping and "
                                  "sorting; 'current' re-groups at every step")
min_power = st.sidebar.number_input("Noise floor: min neuron tracked power",
                                    value=1e-8, format="%.1e")
log_color = st.sidebar.checkbox("log10 color scale for heatmaps", value=False,
                                key="log_color")
log_floor = st.sidebar.number_input("log-scale floor (values below clip here)",
                                    value=1e-8, format="%.1e", key="log_floor",
                                    disabled=not log_color)


def heat_values(Z: np.ndarray, label: str) -> tuple[np.ndarray, str]:
    """Apply the sidebar log scale to a heatmap matrix. Non-negative representations
    are shown as log10(max(Z, floor)); the signed coefficient keeps a linear scale."""
    if not log_color or label == "a":
        return Z, label
    return np.log10(np.maximum(Z, log_floor)), f"log10 {label}"

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


def mode_noise_threshold(ref_var: np.ndarray, abs_floor: float, mult: float) -> np.ndarray:
    """Per-neuron power floor for 'produced' modes: max(abs_floor, mult * var_i / n),
    i.e. a multiple of the sampling variance of a_iS estimated from n samples."""
    return np.maximum(abs_floor, mult * ref_var / max(1, data["num_samples"]))


def neuron_order(kind: str, ref_C: np.ndarray, ref_var: np.ndarray,
                 hd_floor: float = 1e-6, hd_mult: float = 10.0) -> np.ndarray:
    """Overview ordering. Groups follow hierarchy order (d2_0..d2_7, d4_*, ...);
    inside a group neurons sort by the power of their group mode."""
    rP = ref_C ** 2
    if kind == "original neuron index":
        return np.arange(N)
    if kind == "dominant mode":
        g = rP.argmax(0)
    elif kind == "highest-degree mode produced":
        degs = np.array([m["degree"] for m in mode_sel])
        g = M.highest_degree_mode(ref_C, degs, mode_noise_threshold(ref_var, hd_floor, hd_mult))
    else:
        raise ValueError(kind)
    return np.lexsort((-rP[g, np.arange(N)], g))


tabs = st.tabs(["Overview", "Input Walsh", "Bottleneck Walsh", "Mode view",
                "Neuron view", "Layer population", "Mode interactions",
                "Training dynamics", "Hierarchy"])

# ── Overview: neuron x mode heatmap ──────────────────────────────────────────
with tabs[0]:
    c1, c2, c3 = st.columns([2, 2, 3])
    n_sort = c1.selectbox("Neuron sorting", [
        "original neuron index", "dominant mode", "highest-degree mode produced"],
        index=2, key="ov_sort")
    m_sort = c2.selectbox("Mode sorting",
                          ["hierarchy", "degree", "total mode power", "N_eff"])
    drop_lowvar = c3.checkbox("remove low-variance neurons", value=False,
                              key="drop_lowvar")
    var_exp = c3.slider("variance threshold: 10^", -6.0, 0.0, -4.0, step=0.1,
                        key="var_exp", disabled=not drop_lowvar)
    hd_floor, hd_mult = 1e-6, 10.0
    if n_sort == "highest-degree mode produced":
        h1, h2 = st.columns(2)
        hd_floor = h1.number_input("produced-mode floor: min a² (absolute)", value=1e-6,
                                   format="%.1e", key="hd_floor")
        hd_mult = h2.number_input("produced-mode floor: × sampling noise σ²/n", value=10.0,
                                  min_value=0.0, step=1.0, key="hd_mult")
        st.caption("A neuron is assigned the highest-degree tracked mode whose power "
                   "a² ≥ max(floor, mult·σ²_i/n) — ties by power; neurons with nothing "
                   "above the floor fall back to their dominant mode.")

    # ordering reference: which checkpoint supplies the sort keys.
    # "fixed neuron index" + a sort uses FINAL-checkpoint keys so the order
    # stays stable while scrubbing the step slider.
    if order_ref == "current checkpoint":
        ref, ref_var = C, var
    else:  # "final checkpoint", or "fixed neuron index" with a sort chosen
        fin = load_step(str(RUNS_ROOT), run, steps[-1])["locations"][loc_name]
        ref, ref_var = fin["C"][mode_idx], fin["var"]
    nord = neuron_order(n_sort, ref, ref_var, hd_floor, hd_mult)
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
    Z, zlabel = heat_values(Z, zlabel)
    if log_color and zlabel == "a":
        st.caption("log color scale is not applied to the signed coefficient; pick power or |a|.")
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

# ── Input Walsh: what each neuron type reads ─────────────────────────────────
with tabs[1]:
    iw_all = load_input_walsh(str(RUNS_ROOT), run, step)
    if iw_all is None:
        st.info("No input-Walsh artifact for this step. Compute it with\n\n"
                f"`python -m analysis.neuron_walsh.input_walsh --run {run}`\n\n"
                "(or pass `--input-walsh` to analysis.neuron_walsh.compute).")
    else:
        iw = iw_all["blocks"][loc["block"]]
        k_rel = iw_all["relevant_dim"]
        # bitmask -> display name: tree modes by tree name, singletons chi_j,
        # anything else chi_{j1,j2,...} (0-based relevant coordinates)
        tree_name = {sum(1 << j for j in m["support"]): m["name"]
                     for m in iw_all["modes"] if m["degree"] > 1}

        def mode_label(s: int) -> str:
            if s == 0:
                return "1"
            if s in tree_name:
                return tree_name[s]
            bits = [j for j in range(k_rel) if (s >> j) & 1]
            return ("χ_" + str(bits[0])) if len(bits) == 1 else \
                "χ_{" + ",".join(map(str, bits)) + "}"

        # group = OUTPUT mode (same C / degree filter as the sidebar), by either rule
        rule = st.radio("Grouping rule", ["highest-degree mode produced", "dominant mode"],
                        horizontal=True, key="iw_rule")
        if rule == "dominant mode":
            grp = dom
        else:
            r1, r2 = st.columns(2)
            iw_floor = r1.number_input("produced-mode floor: min a² (absolute)", value=1e-6,
                                       format="%.1e", key="iw_hd_floor")
            iw_mult = r2.number_input("produced-mode floor: × sampling noise σ²/n", value=10.0,
                                      min_value=0.0, step=1.0, key="iw_hd_mult")
            grp = M.highest_degree_mode(C, np.array([m["degree"] for m in mode_sel]),
                                        mode_noise_threshold(var, iw_floor, iw_mult))
        counts = np.bincount(grp, minlength=len(mode_names))
        group_labels = [f"{n}  ({counts[i]} neurons)" for i, n in enumerate(mode_names)]
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        g_label = c1.selectbox("Neuron group: output mode", group_labels,
                               index=len(group_labels) - 1, key="iw_group")
        gi = group_labels.index(g_label)
        max_terms = c2.slider("max terms", 1, 16, 6, key="iw_terms")
        rel_thr = c3.number_input("relative threshold (× largest |c|)", 0.0, 1.0,
                                  0.05, step=0.01, format="%.2f", key="iw_rel")
        noise_mult = c4.number_input("noise multiplier (× robust σ)", 0.0, 100.0,
                                     5.0, step=1.0, key="iw_noise")
        c5, c6, c7 = st.columns(3)
        show_const = c5.checkbox("include constant term", value=True, key="iw_const")
        hide_low = c6.checkbox("hide neurons below the sidebar noise floor",
                               value=True, key="iw_hide_low")
        max_rows = c7.number_input("max rows", 10, 5000, 300, step=10, key="iw_rows")

        members = np.where(grp == gi)[0]
        if hide_low:
            members = members[~low_power[members]]
        a_out = C[gi, members]
        members = members[np.argsort(-np.abs(a_out))]
        a_out = C[gi, members]

        rows = []
        for ni, a in zip(members[:int(max_rows)], a_out[:int(max_rows)]):
            idx, coef = iw["top_idx"][ni], iw["top_coef"][ni]
            if not show_const:
                keep0 = idx != 0
                idx, coef = idx[keep0], coef[keep0]
            if len(coef) == 0:
                rows.append(dict(neuron=int(ni), a_out=float(a), input_power=float(iw["power"][ni]),
                                 shown_frac=0.0, n_above=0, input=""))
                continue
            thr = max(rel_thr * np.abs(coef).max(), noise_mult * iw["noise_sigma"][ni])
            above = np.abs(coef) >= thr
            n_above = int(above.sum())
            sel = np.where(above)[0][:max_terms]
            # display order: support size ascending (constant first), then by the
            # support itself (chi_0 < chi_1, d2_3 < d4_0 < d4_1, ...)
            sel = sorted(sel, key=lambda j: (bin(int(idx[j])).count("1"),
                                             [b for b in range(k_rel) if (int(idx[j]) >> b) & 1]))
            terms = [f"{coef[j]:+.3f}·{mode_label(int(idx[j]))}" for j in sel]
            shown = float((coef[sel] ** 2).sum() / (iw["power"][ni] + 1e-30))
            rows.append(dict(
                neuron=int(ni), a_out=float(a), input_power=float(iw["power"][ni]),
                shown_frac=shown,
                n_above=(f"≥{iw_all['topk']}" if n_above >= iw_all["topk"] else n_above),
                input=" ".join(terms).replace("+", "+ ").replace("-", "- ").lstrip("+ "),
            ))
        st.caption(
            f"{loc_name} @ step {step}: {len(members)} neurons whose output mode ({rule}) is "
            f"**{mode_names[gi]}**, sorted by |a_out| (signed coefficient of that mode in the "
            f"post-nonlinearity output). `input` is the exact Walsh decomposition of the "
            f"pre-activation V·h over the {k_rel} relevant bits (top-{iw_all['topk']} stored), "
            f"keeping terms with |c| ≥ max(rel·max|c|, mult·σ_noise), listed by support size "
            f"(constant first) then by support index; `shown_frac` is the share "
            f"of E[z²] those terms carry, `n_above` how many of the stored terms pass the "
            f"threshold. Names: tree modes by name, singletons χ_j, other supports χ_{{j,…}} "
            f"(0-based relevant coordinates); `1` is the constant term. Irrelevant bits "
            f"averaged over {iw_all['repeats']} draws.")
        st.dataframe(rows, use_container_width=True, height=min(900, 60 + 35 * max(1, len(rows))),
                     column_config={
                         "a_out": st.column_config.NumberColumn(format="%.4f"),
                         "input_power": st.column_config.NumberColumn(format="%.3e"),
                         "shown_frac": st.column_config.NumberColumn(format="%.3f"),
                         "input": st.column_config.TextColumn(width="large"),
                     })
        with st.expander("plain-text list (full decompositions)", expanded=False):
            st.code("\n".join(
                f"n{r['neuron']:5d}  a_out={r['a_out']:+.4f}  E[z²]={r['input_power']:.2e}  "
                f"shown={r['shown_frac']:.2f}  z = {r['input']}" for r in rows) or "(empty)",
                language=None)
        with st.expander("group summary: input power by support size"):
            dp = iw["degree_power"][members] / (iw["power"][members, None] + 1e-30)
            fdeg = go.Figure(go.Bar(x=[str(j) for j in range(dp.shape[1])], y=dp.mean(0)))
            fdeg.update_layout(height=260, xaxis_title="|S| (support size)",
                               yaxis_title="mean share of E[z²]",
                               margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fdeg, use_container_width=True)

# ── Bottleneck Walsh: code neurons and what passes through ───────────────────
with tabs[2]:
    bw_all = load_bottleneck_walsh(str(RUNS_ROOT), run, step)
    if bw_all is None:
        st.info("No bottleneck-Walsh artifact for this step (only bottleneck models have "
                "one). Compute it with\n\n"
                f"`python -m analysis.neuron_walsh.bottleneck_walsh --run {run}`")
    else:
        bw_blocks = sorted(bw_all["bottlenecks"])
        bw_modes_all = bw_all["modes"]
        c1, c2, c3 = st.columns([1, 2, 2])
        bw_block = c1.selectbox("Bottleneck (after block)", bw_blocks,
                                index=bw_blocks.index(loc["block"]) if loc["block"] in bw_blocks else 0,
                                key="bw_block")
        stage_opts = ["pre-TopK (every code neuron)", "post-TopK (what passes through)"]
        if "C_out" in bw_all["bottlenecks"][bw_blocks[0]]:
            stage_opts.append("post-decoder (stream to the next block)")
        stage = c2.radio("Stage", stage_opts, horizontal=True, key="bw_stage")
        bw_degrees = c3.multiselect("Mode degrees", sorted({m["degree"] for m in bw_modes_all}),
                                    default=sorted({m["degree"] for m in bw_modes_all}),
                                    key="bw_degrees")
        bw = bw_all["bottlenecks"][bw_block]
        bw_sel = [m for m in bw_modes_all if m["degree"] in bw_degrees]
        if not bw_sel:
            st.warning("select at least one degree")
            st.stop()
        bw_idx = np.array([m["index"] for m in bw_sel])
        bw_names = [m["name"] for m in bw_sel]
        bw_degs = np.array([m["degree"] for m in bw_sel])
        pre = stage.startswith("pre")
        decoder_stage = stage.startswith("post-decoder")
        if decoder_stage:
            Cb, varb = bw["C_out"][bw_idx], bw["var_out"]              # [M, N] residual coordinates
            freq = bw["var_out"]                                       # "activity" = variance
        else:
            Cb = (bw["C_pre"] if pre else bw["C_post"])[bw_idx]      # [M, d]
            varb = bw["var_pre"] if pre else bw["var_post"]
            freq = bw["select_freq"]
        d_b = Cb.shape[1]
        Pb = Cb ** 2
        totb = Pb.sum(0)

        c4, c5, c6, c7 = st.columns([2, 1, 1, 1])
        bw_sort = c4.selectbox("Neuron sorting", [
            "highest-degree mode produced", "dominant mode", "selection frequency",
            "original neuron index"], key="bw_sort")
        keep_n = c5.number_input("keep most active (0 = all)" if not decoder_stage
                                 else "keep highest-variance coordinates (0 = all)",
                                 0, d_b, 0 if decoder_stage else 64, step=8, key="bw_keep")
        bw_floor = c6.number_input("produced-mode floor: min a²", value=1e-6, format="%.1e",
                                   key="bw_floor")
        bw_mult = c7.number_input("× sampling noise σ²/n", value=10.0, min_value=0.0,
                                  step=1.0, key="bw_mult")
        keep = np.argsort(-freq)[:int(keep_n)] if keep_n > 0 else np.arange(d_b)
        keep = np.sort(keep)
        if bw_sort == "original neuron index":
            nordb = keep
        elif bw_sort == "selection frequency":
            nordb = keep[np.argsort(-freq[keep])]
        else:
            if bw_sort == "dominant mode":
                g = Pb.argmax(0)
            else:
                thr = np.maximum(bw_floor, bw_mult * varb / max(1, bw_all["num_samples"]))
                g = M.highest_degree_mode(Cb, bw_degs, thr)
            nordb = keep[np.lexsort((-Pb[g[keep], keep], g[keep]))]
        gdom = Pb.argmax(0)

        if rep == "Power |a|^2":
            Zb, zl = Pb, "power"
        elif rep == "Raw signed coefficient":
            Zb, zl = Cb, "a"
        elif rep == "Absolute coefficient":
            Zb, zl = np.abs(Cb), "|a|"
        else:
            Zb, zl = Pb / (totb[None, :] + 1e-30), "p(S|j)"
        cdb = np.stack([nordb.astype(float), freq[nordb], totb[nordb],
                        varb[nordb]], axis=-1)[None, :, :].repeat(len(bw_sel), 0)
        Zshow, zl = heat_values(Zb[:, nordb], zl)
        fig = go.Figure(go.Heatmap(
            z=Zshow, y=bw_names, x=np.arange(len(nordb)), customdata=cdb,
            hovertemplate=(("residual coordinate" if decoder_stage else "code neuron") +
                           " %{customdata[0]:.0f}<br>mode %{y}<br>" + zl + " %{z:.3e}<br>" +
                           ("" if decoder_stage else "selected %{customdata[1]:.3f} of inputs · ") +
                           "tracked power %{customdata[2]:.3e} · var %{customdata[3]:.3e}"),
            colorscale="Viridis", colorbar=dict(title=zl)))
        fig.update_layout(height=max(420, 90 + 22 * len(bw_sel)),
                          margin=dict(l=10, r=10, t=25, b=10),
                          xaxis_title=("residual coordinate" if decoder_stage else "code neuron")
                                      + " (sorted position — hover for id)",
                          yaxis_title="mode")
        st.plotly_chart(fig, use_container_width=True)
        n_active = int((bw["select_freq"] > 0).sum())
        st.caption(
            f"bottleneck after block {bw_block} (d={bw['dim']}, K={bw['k']}, "
            f"{'Abs' if bw['abs_topk'] else 'signed '}TopK) @ step {step}: "
            f"{len(nordb)}/{d_b} {'residual coordinates' if decoder_stage else 'code neurons'} shown "
            f"({n_active} code neurons are ever selected, "
            f"{int((bw['select_freq'] > 0.01).sum())} on >1% of inputs). Coefficients of the "
            f"{'decoder output W_dec·code, i.e. the stream the next block receives' if decoder_stage else ('encoder output before TopK' if pre else 'sparse code after TopK')} on "
            f"{len(bw_sel)} tracked modes, uniform data. Sorting: as in Overview, over the "
            f"stage shown; 'keep most active' ranks by selection frequency first.")

        # what passes through: per-mode power of pre / post codes and of the decoder output
        with st.expander("modes passed through this bottleneck (layer-level Walsh power)",
                         expanded=True):
            pre_pow = (bw["C_pre"][bw_idx] ** 2).sum(1)
            post_pow = (bw["C_post"][bw_idx] ** 2).sum(1)
            out_pow = bw["output_power"][bw_idx]
            fp = go.Figure()
            fp.add_bar(x=bw_names, y=pre_pow, name="Σ_j a² pre-TopK")
            fp.add_bar(x=bw_names, y=post_pow, name="Σ_j a² post-TopK")
            fp.add_bar(x=bw_names, y=out_pow, name="decoder output H(S)")
            fp.update_layout(barmode="group", height=320, yaxis_type="log",
                             margin=dict(l=10, r=10, t=20, b=10), yaxis_title="power")
            st.plotly_chart(fp, use_container_width=True)
            st.caption("decoder output H(S) = ‖E[W_dec·code · χ_S]‖² is what the next block "
                       "receives from this bottleneck; pre/post sums are over code neurons.")

        with st.expander("kept code neurons (table)"):
            table = [dict(code_neuron=int(j), select_freq=float(freq[j]),
                          dominant_mode=bw_names[gdom[j]], dominant_power=float(Pb[gdom[j], j]),
                          tracked_power=float(totb[j]), variance=float(varb[j]))
                     for j in nordb]
            st.dataframe(table, use_container_width=True,
                         column_config={
                             "select_freq": st.column_config.NumberColumn(format="%.3f"),
                             "dominant_power": st.column_config.NumberColumn(format="%.3e"),
                             "tracked_power": st.column_config.NumberColumn(format="%.3e"),
                             "variance": st.column_config.NumberColumn(format="%.3e")})

# ── Mode view ────────────────────────────────────────────────────────────────
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
with tabs[6]:
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
with tabs[7]:
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
with tabs[8]:
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
