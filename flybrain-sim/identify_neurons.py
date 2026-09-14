"""
One-time script: identify T4/T5/LC4/LPLC2/DN neuron internal indices from
the MaleCNS annotation data, mapped against DOOMFLY's processed graph.

Run after DOOMFLY setup:
  python identify_neurons.py --doomfly ../doomfly

Outputs: neuron_groups.json  (biological_id → internal index mappings)

Extra modes:
  --list-types [PATTERN]     Print all unique type names (optionally filtered).
                             Use this to discover annotation type names.
  --trace-from TYPE [--top N]  Traverse outgoing synapses from all neurons of
                             TYPE (both sides) and print the top N downstream
                             partners by total synaptic weight.  Good for
                             tracing e.g. LPLC2 → escape command DNs.
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_TYPES = {
    "t4a_left":   ("T4a", "L"),
    "t4a_right":  ("T4a", "R"),
    "t4b_left":   ("T4b", "L"),
    "t4b_right":  ("T4b", "R"),
    "t4c_left":   ("T4c", "L"),
    "t4c_right":  ("T4c", "R"),
    "t4d_left":   ("T4d", "L"),
    "t4d_right":  ("T4d", "R"),
    "t5a_left":   ("T5a", "L"),
    "t5a_right":  ("T5a", "R"),
    "t5b_left":   ("T5b", "L"),
    "t5b_right":  ("T5b", "R"),
    "t5c_left":   ("T5c", "L"),
    "t5c_right":  ("T5c", "R"),
    "t5d_left":   ("T5d", "L"),
    "t5d_right":  ("T5d", "R"),
    "lc4_left":   ("LC4",   "L"),
    "lc4_right":  ("LC4",   "R"),
    "lplc2_left": ("LPLC2", "L"),
    "lplc2_right":("LPLC2", "R"),
    # Descending neurons — motor output
    "dna02_left":  ("DNa02", "L"),
    "dna02_right": ("DNa02", "R"),
    "dng100_left": ("DNg100", "L"),
    "dng100_right":("DNg100", "R"),
    "dng13_left":  ("DNg13", "L"),
    "dng13_right": ("DNg13", "R"),
    # Giant Fiber escape circuit — GF absent from MaleCNS annotation;
    # DNp01 and DNp103 are the top DN targets of LPLC2 by synaptic weight
    # (trace-from LPLC2: DNp01 L=727/R=611, DNp103 R=788/L=605)
    "gf_left":    ("GF",     "L"),   # kept for alias search; likely absent
    "gf_right":   ("GF",     "R"),
    "dnp01_left":  ("DNp01",  "L"),
    "dnp01_right": ("DNp01",  "R"),
    "dnp103_left": ("DNp103", "L"),
    "dnp103_right":("DNp103", "R"),
    # Mechanosensory groups are populated by the prefix scan below.  The
    # MaleCNS annotation vocabulary has changed between exports, so these are
    # deliberately not hard-coded to one JON/chordotonal subtype.
    "mech_left": ("__DISCOVER__", "L"),
    "mech_right": ("__DISCOVER__", "R"),
}

MECH_TYPE_RE = r"(?i)(jon|johnston|chordot|campaniform|mechanosens|proprio)"

# Some type names may differ slightly in the annotations — aliases tried if primary fails.
TYPE_ALIASES = {
    "DNa02":  ["DNa02", "dna02", "aDN2"],
    "DNg100": ["DNg100", "DNg100a", "DNg100b"],
    "DNg13":  ["DNg13", "gDN13"],
    "GF":     ["GF", "Giant_Fiber", "giantfiber", "giant fiber", "GiantFiber"],
    "LPLC2":  ["LPLC2", "lplc2"],
    "LC4":    ["LC4", "lc4"],
}


def resolve_doomfly(path: str) -> Path:
    """Resolve DOOMFLY from cwd, flybrain-sim, or fly-brain repo root."""
    requested = Path(path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        here = Path(__file__).resolve().parent
        candidates.extend([here / requested, here.parent / requested, here.parent / "doomfly"])
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "doom").is_dir():
            return resolved
    print("ERROR: DOOMFLY not found; checked:")
    for candidate in candidates:
        print(f"  {candidate.resolve()}")
    print("Put DOOMFLY at ../doomfly or pass --doomfly /path/to/doomfly")
    sys.exit(1)


def load_annotations(doomfly_path: Path):
    """Find and load the MaleCNS neuron annotation file (feather or CSV)."""
    base = doomfly_path / "connectome_data/malecns_v1"
    for name in ["annotations.feather", "neuron_annotations.feather"]:
        p = base / name
        if p.exists():
            return pd.read_feather(p)
    for m in sorted(base.glob("*.feather")):
        return pd.read_feather(m)
    for pattern in ["neuprint_Neuprint_Meta_*.csv", "*neuron*.csv", "*.csv"]:
        matches = list(base.glob(pattern))
        for m in matches:
            if "Meta" in m.name or "neuron" in m.name.lower():
                return pd.read_csv(m, low_memory=False)
        if matches:
            return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(
        f"No annotation file found under {base}. "
        "Run DOOMFLY's data download step first."
    )


def resolve_type(df, type_col, type_name: str):
    candidates = TYPE_ALIASES.get(type_name, [type_name])
    for t in candidates:
        mask = df[type_col].astype(str).str.strip() == t
        if mask.any():
            return mask
    return pd.Series([False] * len(df))


def get_col_names(df):
    id_col   = next((c for c in df.columns if "body" in c.lower() or "id" in c.lower()), df.columns[0])
    side_col = next((c for c in df.columns if "side" in c.lower() or "soma" in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ("type", "cell_type", "celltype")), None)
    return id_col, type_col, side_col


def load_graph(doomfly_path: Path):
    graph_path = doomfly_path / "outputs/doom/malecns_v1/graph.npz"
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found.")
        print("Run DOOMFLY's full setup first.")
        sys.exit(1)
    data = np.load(graph_path, allow_pickle=False)
    return data, data["ids"].astype(np.int64)


# ── list-types mode ─────────────────────────────────────────────────────────

def cmd_list_types(args):
    doomfly_path = resolve_doomfly(args.doomfly)
    df = load_annotations(doomfly_path)
    _, type_col, _ = get_col_names(df)
    if type_col is None:
        print("ERROR: no type column found. Columns:", list(df.columns))
        sys.exit(1)
    types = sorted(df[type_col].dropna().astype(str).unique())
    pattern = args.list_types.lower() if args.list_types and args.list_types != "ALL" else None
    filtered = [t for t in types if pattern is None or pattern in t.lower()]
    print(f"{len(filtered)} type names" + (f" matching '{pattern}'" if pattern else "") + ":")
    for t in filtered:
        n = (df[type_col].astype(str) == t).sum()
        print(f"  {t:40s}  ({n} neurons)")


# ── trace-from mode ──────────────────────────────────────────────────────────

def cmd_trace_from(args):
    doomfly_path = resolve_doomfly(args.doomfly)
    top_n = args.top

    data, brain_ids = load_graph(doomfly_path)
    ptr    = data["ptr"].astype(np.int64)
    post   = data["post"].astype(np.int32)
    weight = data["weight"].astype(np.float32)

    df = load_annotations(doomfly_path)
    id_col, type_col, side_col = get_col_names(df)
    if type_col is None:
        print("ERROR: no type column found.")
        sys.exit(1)

    # Find source neurons (both sides)
    source_type = args.trace_from
    src_mask = resolve_type(df, type_col, source_type)
    if not src_mask.any():
        print(f"ERROR: no neurons found for type '{source_type}'.")
        print("Run --list-types to browse available names.")
        sys.exit(1)

    src_bio_ids = df.loc[src_mask, id_col].astype(np.int64).values
    src_idx = np.searchsorted(brain_ids, src_bio_ids)
    valid = (src_idx < len(brain_ids)) & (brain_ids[np.minimum(src_idx, len(brain_ids)-1)] == src_bio_ids)
    src_internal = src_idx[valid]
    print(f"\nSource: {source_type}  ({len(src_bio_ids)} bio IDs, {len(src_internal)} in graph)\n")

    # Accumulate outgoing synapse weight per target
    tgt_weight = {}
    tgt_count  = {}
    for src in src_internal:
        for e in range(ptr[src], ptr[src + 1]):
            j = int(post[e])
            w = float(weight[e])
            tgt_weight[j] = tgt_weight.get(j, 0.0) + w
            tgt_count[j]  = tgt_count.get(j, 0) + 1

    if not tgt_weight:
        print("No outgoing synapses found.")
        return

    # Build a bio-id → annotation row lookup
    id_to_row = {int(row[id_col]): row for _, row in df.iterrows()}

    sorted_tgts = sorted(tgt_weight.items(), key=lambda x: -x[1])[:top_n]

    print(f"{'Rank':<5} {'Type':<30} {'Side':<6} {'Synapses':<10} {'Total weight':<14} {'Bio ID'}")
    print("-" * 80)
    for rank, (tgt_idx, total_w) in enumerate(sorted_tgts, 1):
        bio_id = int(brain_ids[tgt_idx])
        row = id_to_row.get(bio_id)
        tgt_type = str(row[type_col]).strip() if row is not None and type_col else "?"
        tgt_side = str(row[side_col]).strip() if row is not None and side_col else "?"
        n_syn = tgt_count[tgt_idx]
        print(f"{rank:<5} {tgt_type:<30} {tgt_side:<6} {n_syn:<10} {total_w:<14.1f} {bio_id}")

    print(f"\n(Showing top {len(sorted_tgts)} of {len(tgt_weight)} downstream partners)")
    print("\nTo add a type to neuron_groups.json, add it to TARGET_TYPES and re-run without flags.")


# ── main identify mode ───────────────────────────────────────────────────────

def cmd_identify(args):
    doomfly_path = resolve_doomfly(args.doomfly)

    data, brain_ids = load_graph(doomfly_path)
    print(f"Loading graph from {doomfly_path / 'outputs/doom/malecns_v1/graph.npz'}...")
    print(f"  {len(brain_ids)} neurons in graph")

    print("Loading annotations...")
    df = load_annotations(doomfly_path)
    print(f"  {len(df)} rows in annotation CSV")

    id_col, type_col, side_col = get_col_names(df)
    if type_col is None:
        print("ERROR: could not find type column in annotation CSV.")
        print("Available columns:", list(df.columns))
        sys.exit(1)
    print(f"  id_col={id_col}, type_col={type_col}, side_col={side_col}")

    groups = {}
    missing = []

    for group_key, (type_name, side) in TARGET_TYPES.items():
        if type_name == "__DISCOVER__":
            # Keep this broad at discovery time; the resulting JSON is the
            # auditable list of annotation types that actually existed in the
            # downloaded release.
            type_mask = df[type_col].astype(str).str.contains(MECH_TYPE_RE, regex=True, na=False)
        else:
            type_mask = resolve_type(df, type_col, type_name)
        if side_col:
            side_mask = df[side_col].astype(str).str.upper().str.startswith(side)
            mask = type_mask & side_mask
        else:
            mask = type_mask

        bio_ids = df.loc[mask, id_col].astype(np.int64).values
        if len(bio_ids) == 0:
            missing.append(group_key)
            groups[group_key] = []
            continue

        idx = np.searchsorted(brain_ids, bio_ids)
        valid = (idx < len(brain_ids)) & (brain_ids[np.minimum(idx, len(brain_ids)-1)] == bio_ids)
        internal = idx[valid].tolist()
        groups[group_key] = internal
        print(f"  {group_key}: {len(bio_ids)} bio IDs → {len(internal)} in graph")

    if missing:
        print(f"\nWARNING: no neurons found for: {missing}")
        print("Run --list-types to browse available type names.")

    out_path = Path(__file__).parent / "neuron_groups.json"
    out_path.write_text(json.dumps(groups, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"Total groups: {len(groups)}, empty: {len(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify neuron groups and trace circuits in MaleCNS.")
    parser.add_argument("--doomfly", default="../doomfly",
                        help="Path to DOOMFLY repo root")
    parser.add_argument("--list-types", nargs="?", const="ALL", metavar="PATTERN",
                        help="List annotation type names (optionally filtered by substring)")
    parser.add_argument("--trace-from", metavar="TYPE",
                        help="Trace outgoing synapses from this neuron type")
    parser.add_argument("--top", type=int, default=30,
                        help="How many downstream partners to show (default: 30)")
    args = parser.parse_args()

    if args.list_types is not None:
        cmd_list_types(args)
    elif args.trace_from:
        cmd_trace_from(args)
    else:
        cmd_identify(args)


if __name__ == "__main__":
    main()
