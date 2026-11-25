import json, os, numpy as np
from polymarket_gnn.evaluation.metrics import cm_to_metrics
def load(label):
    f = f"val_history_{label}.json"
    if not os.path.exists(f): return None
    v = json.load(open(f))
    if not v: return None
    out = {}
    for k, field in (("all", "cm"), ("dedup", "cm_dedup")):
        out[k] = cm_to_metrics(sum(np.array(d[field]) for d in v if field in d))
    return out
rows = [("reference (alpha 2,1,2)", "{H}_lb_ablation_s0"), ("alpha auto (aw)", "{H}_lb_aw_ablation_s0"),
        ("undersampling 1.5 (us)", "{H}_lb_us_ablation_s0"), ("regression (reg)", "{H}_lb_reg_ablation_s0")]
for H in ("1h", "6h", "12h"):
    bl = json.load(open(f"baseline_results_{H}.json"))["baselines"]["Mean Reversion H (causal)"]
    print(f"\n=== {H} ===  (Mean Reversion H: all bal_acc={bl['all']['balanced_accuracy']:.4f} macroF1={bl['all']['macro_f1']:.4f} acc={bl['all']['accuracy']:.4f} | dedup bal_acc={bl['dedup']['balanced_accuracy']:.4f})")
    for name, lab in rows:
        m = load(lab.format(H=H))
        if m is None: print(f"  {name:<26} not available"); continue
        a, d = m["all"], m["dedup"]; pc = a["per_class"]
        print(f"  {name:<26} all: bal_acc={a['balanced_accuracy']:.4f} macroF1={a['macro_f1']:.4f} acc={a['accuracy']:.4f} "
              f"F1 D/C/U={pc['DOWN']['f1']:.3f}/{pc['CONSTANT']['f1']:.3f}/{pc['UP']['f1']:.3f} "
              f"recall D/C/U={pc['DOWN']['recall']:.3f}/{pc['CONSTANT']['recall']:.3f}/{pc['UP']['recall']:.3f} | dedup: bal_acc={d['balanced_accuracy']:.4f} acc={d['accuracy']:.4f}")
    c = f"calibration_{H}_lb_reg_ablation_s0.json"
    if os.path.exists(c):
        c = json.load(open(c))
        vs = sorted(c.get("val_sweep", []), key=lambda r: -r["balanced_accuracy"])
        print(f"  regression calibration: tau={c['tau']:.5f} (train bal_acc at tau={c.get('train_balanced_accuracy_at_tau', float('nan')):.4f}, n={c.get('n_calibration_samples')}); "
              f"diagnostic only — best tau on validation would be {vs[0]['tau']:.5f} with bal_acc={vs[0]['balanced_accuracy']:.4f}" if vs else "")
