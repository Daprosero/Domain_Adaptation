"""Remote entry point of the v58 timing probe (`tools/timing_v58.py`), for a cloud session.

    run(device="cuda", groups=..., strategies=..., phase2_epochs={"CCA-LSC": 4})

Runs the same short probe as the local one (2 epochs, warm-up excluded; CCA-LSC 4) on
the processed data mounted as an input dataset. The processed tree is found through
`FORGE_INPUT_DATASETS` (the directory holding `manifest.csv` under the mounted
dataset), else the repository's own `MIL-CREDA-EEG/Data/processed`. Writes into the
session's working directory, which is what a fetch returns: `timing_probe/probe.json`
(the measurements) and, when run by the remote runner (`FORGE_RUN_MODE` set),
`shard.json`, the evidence stamp of commit, device and GPU name.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

from MIL_CREDA_EEG import bag_data

PROBE_FILE = bag_data.REPOSITORY / "tools" / "timing_v58.py"
DATASET_REF = "daprosero/balladeer-existing-local"


def processed_root(datasets_env: str | None = None) -> Path:
    """The processed tree: the mounted input dataset's directory holding `manifest.csv`,
    else the repository's own processed directory."""
    text = os.environ.get("FORGE_INPUT_DATASETS") if datasets_env is None else datasets_env
    mounts = json.loads(text) if text else {}
    for ref, mount in mounts.items():
        if mount is None:
            continue
        base = Path(mount)
        hits = sorted(base.glob("manifest.csv")) or sorted(base.glob("*/manifest.csv")) \
            or sorted(base.glob("*/*/manifest.csv"))
        if hits:
            return hits[0].parent
    return bag_data.REPOSITORY / bag_data.PROCESSED_RELATIVE


def _probe_module():
    spec = importlib.util.spec_from_file_location("timing_v58", PROBE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device_evidence(device_name: str) -> dict:
    import torch
    out = {"requested": device_name, "torch": torch.__version__,
           "cudaAvailable": bool(torch.cuda.is_available()),
           "threads": torch.get_num_threads(), "cpuCount": os.cpu_count(),
           "platform": platform.platform(), "processor": platform.processor()}
    device = torch.device(device_name)
    out["kind"] = device.type
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device cuda requested but torch.cuda.is_available() is False")
        props = torch.cuda.get_device_properties(device)
        out.update({"name": props.name, "capability": f"sm_{props.major}{props.minor}",
                    "totalMemoryMB": props.total_memory / 2**20,
                    "cudaVersion": torch.version.cuda})
    else:
        try:
            model = next((line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), None)
        except OSError:
            model = None
        out["name"] = model or platform.processor() or "cpu"
    return out


def run(device: str = "cpu", transfer: str = "L1->L11", epochs: int = 2,
        bags_per_step: int = 8, groups: list | None = None,
        strategies: list | None = None, phase2_epochs: dict | None = None,
        out_dir: str = "timing_probe", smoke: bool = True) -> dict:
    """Run the probe on `device` and write its record into `out_dir` (cwd-relative)."""
    started = time.time()
    evidence = device_evidence(device)
    print(f"[timing] device {json.dumps(evidence)}", flush=True)
    root = processed_root()
    print(f"[timing] processed root {root}", flush=True)
    probe = _probe_module()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        result = probe.run(transfer, device, epochs, bags_per_step, Path(scratch),
                           groups, strategies, phase2_epochs, processed_root=root)
    result["deviceEvidence"] = evidence
    result["processedRoot"] = str(root)
    result["commit"] = os.environ.get("FORGE_CLONE_COMMIT")
    (target / "probe.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    errors = {f"{k}.{n}": v["error"] for k in ("phase1", "phase2", "sourcePass")
              for n, v in result[k].items() if v.get("error")}
    if os.environ.get("FORGE_RUN_MODE"):
        stamp = {"evidence": {"commit": result["commit"], "device": evidence,
                              "probe": str(target / "probe.json"),
                              "wallSeconds": time.time() - started,
                              "errors": errors}}
        if not errors:
            stamp["evidence"]["complete"] = True
        Path("shard.json").write_text(json.dumps(stamp, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"device": evidence.get("name"), "wallSeconds": result["wallSeconds"],
                      "errors": errors}), flush=True)
    return {"wallSeconds": result["wallSeconds"], "errors": errors}


if __name__ == "__main__":
    kwargs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    run(**kwargs)
