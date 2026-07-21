"""Build the explicit asset ledger + SHA256 manifest for the whole server workspace,
plus env/pip/nvidia snapshots. Output: experiments/final_manifest/ (manifest.json +
snapshots). No writes outside experiments/final_manifest. Then this manifest is the
basis for the timestamped backup and dual-end SHA verification."""
import glob
import hashlib
import json
import os
import subprocess

R = "/root/autodl-tmp/OmniSelect"
OUT = os.path.join(R, "experiments/final_manifest")
os.makedirs(OUT, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


groups = {
    "outputs_json": sorted(glob.glob(f"{R}/outputs/**/*.json", recursive=True)),
    "experiments_logs": sorted(glob.glob(f"{R}/experiments/*.log")),
    "experiments_json": sorted(glob.glob(f"{R}/experiments/**/*.json", recursive=True)),
    "code_scripts": sorted(glob.glob(f"{R}/scripts/*.py")),
    "code_src": sorted(glob.glob(f"{R}/src/**/*.py", recursive=True)),
    "code_baselines": sorted(glob.glob(f"{R}/baselines/**/*.py", recursive=True)),
    "code_root": [f"{R}/runall.sh"] + sorted(glob.glob(f"{R}/requirements*.txt")),
    "root_batch": sorted(glob.glob("/root/batch*.sh") + glob.glob("/root/*controller_reval*.py")
                         + glob.glob("/root/*.log")),
    "data_pool": [f"{R}/data/processed/qpool_train.jsonl", f"{R}/data/processed/qpool_heldout.jsonl",
                  f"{R}/data/processed/pool_manifest.json"],
    "data_small": (sorted(glob.glob(f"{R}/data/daisy/*.dat*")) + sorted(glob.glob(f"{R}/data/tep/*.dat"))
                   + sorted(glob.glob(f"{R}/data/cifar_n/*.pt"))),
    "markers": sorted(glob.glob("/root/*_VALIDATED_OK") + glob.glob("/root/*_OK")
                      + glob.glob("/root/REPLAY_COMPLETE*")),
}

manifest = {"root": R, "groups": {}}
total = 0
for g, files in groups.items():
    entries = {}
    for f in files:
        if os.path.isfile(f):
            entries[f] = {"sha256": sha256(f), "bytes": os.path.getsize(f)}
    manifest["groups"][g] = entries
    total += len(entries)
    print(f"  {g}: {len(entries)} files")
manifest["total_files"] = total

# snapshots
snap = os.path.join(OUT, "snapshots")
os.makedirs(snap, exist_ok=True)
subprocess.run(f"{R}/.venv/bin/pip freeze > {snap}/pip_freeze.txt", shell=True)
subprocess.run(f"nvidia-smi > {snap}/nvidia_smi.txt 2>&1", shell=True)
with open(f"{snap}/env.txt", "w") as fh:
    for k, v in sorted(os.environ.items()):
        if not any(t in k.upper() for t in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            fh.write(f"{k}={v}\n")
subprocess.run(f"cd {R} && git rev-parse HEAD > {snap}/git_head.txt 2>&1", shell=True)

mpath = os.path.join(OUT, "manifest.json")
tmp = mpath + ".tmp"
json.dump(manifest, open(tmp, "w"), indent=1, sort_keys=True)
os.replace(tmp, mpath)
print(f"\nmanifest -> {mpath} total_files={total}")
print(f"manifest_sha256={sha256(mpath)}")
