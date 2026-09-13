"""Download MaleCNS connectome data into doomfly/connectome_data/malecns_v1/."""
from pathlib import Path
import hashlib, json, urllib.request, sys

doomfly = Path(__file__).parent.parent / "doomfly"
if not doomfly.exists():
    sys.exit(f"ERROR: doomfly not found at {doomfly}\nRun from inside fly-brain/ after cloning DOOMFLY.")

name = "malecns_v1"
registry = json.loads((doomfly / "doom/datasets.json").read_text())["datasets"][name]
locked = json.loads((doomfly / f"data-provenance/{name}/source.lock.json").read_text())
root = doomfly / "connectome_data" / name
root.mkdir(parents=True, exist_ok=True)

for filename, url in registry["files"].items():
    target = root / filename
    if target.exists():
        print(f"  already downloaded: {filename}")
    else:
        print(f"  downloading {filename} ...")
        partial = target.with_suffix(".download")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    print(f"  verifying {filename} ...")
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    if digest != locked[filename]["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {filename}")
    print(f"  OK")

(root / "source.lock.json").write_text(json.dumps(locked, indent=2) + "\n")
print("\nAll files verified. Run: python -m doom.connectome malecns_v1")
