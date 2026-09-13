"""
One-time script: identify T4/T5/LC4/LPLC2/DN neuron internal indices from
the MaleCNS annotation data, mapped against DOOMFLY's processed graph.

Run after DOOMFLY setup:
  python identify_neurons.py --doomfly ../doomfly

Outputs: neuron_groups.json  (biological_id → internal index mappings)
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
    # Descending neurons for motor output
    "dna02_left":  ("DNa02", "L"),
    "dna02_right": ("DNa02", "R"),
    "dng100_left": ("DNg100", "L"),
    "dng100_right":("DNg100", "R"),
    "dng13_left":  ("DNg13", "L"),
    "dng13_right": ("DNg13", "R"),
}

# Some type names may differ slightly in the annotations — these aliases are tried if the primary fails.
TYPE_ALIASES = {
    "DNa02": ["DNa02", "dna02", "aDN2"],
    "DNg100": ["DNg100", "DNg100a", "DNg100b"],
    "DNg13": ["DNg13", "gDN13"],
}


def load_annotations(doomfly_path: Path):
    """Find and load the MaleCNS neuron annotation CSV from the connectome_data dir."""
    for pattern in [
        "connectome_data/malecns_v1/neuprint_Neuprint_Meta_*.csv",
        "connectome_data/malecns_v1/*neuron*.csv",
        "connectome_data/malecns_v1/*.csv",
    ]:
        matches = list(doomfly_path.glob(pattern))
        # prefer the one with 'Meta' or 'neurons' in name
        for m in matches:
            if "Meta" in m.name or "neuron" in m.name.lower():
                return pd.read_csv(m, low_memory=False)
        if matches:
            return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(
        f"No annotation CSV found under {doomfly_path}/connectome_data/malecns_v1/. "
        "Run DOOMFLY's data download step first."
    )


def resolve_type(df, type_name: str):
    candidates = TYPE_ALIASES.get(type_name, [type_name])
    for t in candidates:
        mask = df["type"].astype(str).str.strip() == t
        if mask.any():
            return mask
    return pd.Series([False] * len(df))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doomfly", default="../doomfly",
                        help="Path to DOOMFLY repo root")
    args = parser.parse_args()

    doomfly_path = Path(args.doomfly).resolve()
    graph_path = doomfly_path / "outputs/doom/malecns_v1/graph.npz"

    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found.")
        print("Run DOOMFLY's full setup first (connectome import + prepare + build_kernel).")
        sys.exit(1)

    print(f"Loading graph from {graph_path}...")
    data = np.load(graph_path, allow_pickle=False)
    brain_ids = data["ids"].astype(np.int64)
    print(f"  {len(brain_ids)} neurons in graph")

    print("Loading annotations...")
    df = load_annotations(doomfly_path)
    print(f"  {len(df)} rows in annotation CSV")

    # Normalize column names
    id_col = next((c for c in df.columns if "body" in c.lower() or "id" in c.lower()), df.columns[0])
    side_col = next((c for c in df.columns if "side" in c.lower() or "soma" in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ("type", "cell_type", "celltype")), None)

    if type_col is None:
        print("ERROR: could not find type column in annotation CSV.")
        print("Available columns:", list(df.columns))
        sys.exit(1)

    print(f"  id_col={id_col}, type_col={type_col}, side_col={side_col}")

    groups = {}
    missing = []

    for group_key, (type_name, side) in TARGET_TYPES.items():
        type_mask = resolve_type(df, type_name)

        if side_col:
            side_mask = df[side_col].astype(str).str.upper().str.startswith(side)
            mask = type_mask & side_mask
        else:
            # No side info; use all, warn
            mask = type_mask

        bio_ids = df.loc[mask, id_col].astype(np.int64).values
        if len(bio_ids) == 0:
            missing.append(group_key)
            groups[group_key] = []
            continue

        # Map biological IDs → internal graph indices
        idx = np.searchsorted(brain_ids, bio_ids)
        valid = (idx < len(brain_ids)) & (brain_ids[np.minimum(idx, len(brain_ids)-1)] == bio_ids)
        internal = idx[valid].tolist()
        groups[group_key] = internal
        print(f"  {group_key}: {len(bio_ids)} bio IDs → {len(internal)} in graph")

    if missing:
        print(f"\nWARNING: no neurons found for: {missing}")
        print("Type names in annotations may differ. Check with:")
        print("  python -c \"import pandas as pd; df=pd.read_csv('<csv>'); print(df['type'].unique()[:50])\"")

    out_path = Path(__file__).parent / "neuron_groups.json"
    out_path.write_text(json.dumps(groups, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"Total groups: {len(groups)}, empty: {len(missing)}")


if __name__ == "__main__":
    main()
