"""Snapshot of what TensorBoard and Streamlit currently show on this box."""
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[2])
order_file = root / ".streamlit_order"
order = [l.strip() for l in order_file.read_text().splitlines() if l.strip()] if order_file.exists() else []
runs = {}
for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
    nw = d / "analysis" / "neuron_walsh"
    runs[d.name] = dict(
        tb_logs=(d / "tb_logs").is_dir() and any((d / "tb_logs").iterdir()),
        neuron_walsh=nw.is_dir() and any(nw.glob("step_*.pt")),
        input_walsh=(nw / "input_walsh").is_dir(),
        bottleneck_walsh=(nw / "bottleneck_walsh").is_dir(),
        hide_from_streamlit=(d / ".hide_from_streamlit").exists(),
        has_final=(d / "checkpoints" / "final.pt").exists(),
        results_pdf=(d / "results.pdf").exists(),
    )
state = dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=os.uname().nodename,
             remote_root="gdrive:parity/runs_germany", tensorboard=dict(logdir="runs", port=8081),
             streamlit=dict(port=8501, runs_root="runs", order=order), runs=runs)
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(state, indent=1))
tb = [n for n, r in runs.items() if r["tb_logs"]]; sl = [n for n, r in runs.items() if r["neuron_walsh"] and not r["hide_from_streamlit"]]
print(f"runs: {len(runs)} | TB-visible: {len(tb)} | Streamlit-visible: {len(sl)} | hidden markers: {sum(r['hide_from_streamlit'] for r in runs.values())} | order entries: {len(order)}")
print("Streamlit list:", [n for n in order if n in sl] + sorted(n for n in sl if n not in order))
