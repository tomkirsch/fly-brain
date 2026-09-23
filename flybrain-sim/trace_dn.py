"""
Trace upstream AND downstream partners for one or more neuron types.

Usage:
  python trace_dn.py DNp02 DNp11 [--top 20] [--doomfly ../doomfly]

Reports:
  1. How many neurons of that type are in the graph
  2. Top upstream sources (who sends TO this type)
  3. Top downstream targets (what this type sends TO)

Useful for inferring function of under-characterised DNs from their connectivity.
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd


TYPE_ALIASES = {
    "DNa02":  ["DNa02", "dna02", "aDN2"],
    "DNp01":  ["DNp01"],
    "DNp02":  ["DNp02"],
    "DNp04":  ["DNp04"],
    "DNp11":  ["DNp11"],
    "LC4":    ["LC4", "lc4"],
    "LPLC2":  ["LPLC2", "lplc2"],
    "DNg29":  ["DNg29", "dng29"],
}


def resolve_doomfly(path: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [Path(path).expanduser()]
    if not candidates[0].is_absolute():
        candidates += [here / path, here.parent / path, here.parent / "doomfly"]
    for c in candidates:
        r = c.resolve()
        if (r / "doom").is_dir():
            return r
    print("ERROR: DOOMFLY not found; tried:")
    for c in candidates:
        print(f"  {c.resolve()}")
    sys.exit(1)


def load_graph(dp: Path):
    gp = dp / "outputs/doom/malecns_v1/graph.npz"
    if not gp.exists():
        print(f"ERROR: {gp} not found"); sys.exit(1)
    d = np.load(gp, allow_pickle=False)
    return d, d["ids"].astype(np.int64), d["ptr"].astype(np.int64), \
           d["post"].astype(np.int32), d["weight"].astype(np.float32)


def load_ann(dp: Path) -> pd.DataFrame:
    base = dp / "connectome_data/malecns_v1"
    for n in ["annotations.feather", "neuron_annotations.feather"]:
        p = base / n
        if p.exists():
            return pd.read_feather(p)
    for m in sorted(base.glob("*.feather")):
        return pd.read_feather(m)
    for pat in ["*neuron*.csv", "*.csv"]:
        ms = list(base.glob(pat))
        if ms:
            return pd.read_csv(ms[0], low_memory=False)
    raise FileNotFoundError(f"No annotation under {base}")


def get_cols(df):
    id_col   = next((c for c in df.columns if "body" in c.lower() or c.lower() == "id"), df.columns[0])
    side_col = next((c for c in df.columns if "side" in c.lower() or "soma" in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ("type","cell_type","celltype")), None)
    return id_col, type_col, side_col


def resolve_type(df, type_col, name):
    for alias in TYPE_ALIASES.get(name, [name]):
        mask = df[type_col].astype(str).str.strip() == alias
        if mask.any():
            return mask
    return pd.Series([False]*len(df))


def find_internal_indices(df, id_col, type_col, brain_ids, type_name):
    mask = resolve_type(df, type_col, type_name)
    if not mask.any():
        return np.array([], dtype=np.int64), 0
    bio_ids = df.loc[mask, id_col].astype(np.int64).values
    idx = np.searchsorted(brain_ids, bio_ids)
    valid = (idx < len(brain_ids)) & (brain_ids[np.minimum(idx, len(brain_ids)-1)] == bio_ids)
    return idx[valid].astype(np.int64), len(bio_ids)


def build_reverse_index(ptr, post, weight, n):
    """Build a reverse adjacency: for each node j, list (src, weight) tuples."""
    rev = [[] for _ in range(n)]
    for src in range(n):
        for e in range(ptr[src], ptr[src+1]):
            j = int(post[e])
            rev[j].append((src, float(weight[e])))
    return rev


def label(brain_ids, idx, id_to_row, id_col, type_col, side_col):
    bio_id = int(brain_ids[idx])
    row = id_to_row.get(bio_id)
    t = str(row[type_col]).strip() if row is not None and type_col else "?"
    s = str(row[side_col]).strip() if row is not None and side_col else "?"
    return t, s, bio_id


def trace(type_name, df, id_col, type_col, side_col, brain_ids, ptr, post, weight, rev, top_n):
    print(f"\n{'='*70}")
    print(f"  TYPE: {type_name}")
    print(f"{'='*70}")

    internal_idx, n_bio = find_internal_indices(df, id_col, type_col, brain_ids, type_name)
    print(f"  {n_bio} bio IDs, {len(internal_idx)} in graph")

    id_to_row = {int(r[id_col]): r for _, r in df.iterrows()}
    tgt_set = set(internal_idx)

    if not len(internal_idx):
        print("  (not found in graph)")
        return

    # ── UPSTREAM: who sends to this type ─────────────────────────────────────
    up_w = {}
    up_n = {}
    for tgt in internal_idx:
        for src, w in rev[tgt]:
            up_w[src] = up_w.get(src, 0.0) + w
            up_n[src] = up_n.get(src, 0) + 1

    print(f"\n  UPSTREAM (top {top_n} by synaptic weight):")
    print(f"  {'Rank':<4} {'Type':<28} {'Side':<5} {'Syn':<6} {'Weight':<10} {'BioID'}")
    print(f"  {'-'*65}")
    for rank, (src_idx, total_w) in enumerate(sorted(up_w.items(), key=lambda x: -x[1])[:top_n], 1):
        t, s, bio = label(brain_ids, src_idx, id_to_row, id_col, type_col, side_col)
        print(f"  {rank:<4} {t:<28} {s:<5} {up_n[src_idx]:<6} {total_w:<10.1f} {bio}")
    print(f"  ({len(up_w)} total upstream partners)")

    # ── DOWNSTREAM: what this type sends to ──────────────────────────────────
    dn_w = {}
    dn_n = {}
    for src in internal_idx:
        for e in range(ptr[src], ptr[src+1]):
            j = int(post[e])
            w = float(weight[e])
            dn_w[j] = dn_w.get(j, 0.0) + w
            dn_n[j] = dn_n.get(j, 0) + 1

    print(f"\n  DOWNSTREAM (top {top_n} by synaptic weight):")
    print(f"  {'Rank':<4} {'Type':<28} {'Side':<5} {'Syn':<6} {'Weight':<10} {'BioID'}")
    print(f"  {'-'*65}")
    for rank, (tgt_idx, total_w) in enumerate(sorted(dn_w.items(), key=lambda x: -x[1])[:top_n], 1):
        t, s, bio = label(brain_ids, tgt_idx, id_to_row, id_col, type_col, side_col)
        print(f"  {rank:<4} {t:<28} {s:<5} {dn_n[tgt_idx]:<6} {total_w:<10.1f} {bio}")
    print(f"  ({len(dn_w)} total downstream partners)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("types", nargs="+", help="Neuron types to trace (e.g. DNp02 DNp11)")
    ap.add_argument("--top",     type=int, default=20)
    ap.add_argument("--doomfly", default="../doomfly")
    args = ap.parse_args()

    dp = resolve_doomfly(args.doomfly)
    data, brain_ids, ptr, post, weight = load_graph(dp)
    df = load_ann(dp)
    id_col, type_col, side_col = get_cols(df)

    print(f"Graph: {len(brain_ids)} neurons, {len(post)} synapses")
    print(f"Building reverse index (this takes ~30s)...")
    rev = build_reverse_index(ptr, post, weight, len(brain_ids))
    print(f"Done.\n")

    for t in args.types:
        trace(t, df, id_col, type_col, side_col, brain_ids, ptr, post, weight, rev, args.top)


if __name__ == "__main__":
    main()
