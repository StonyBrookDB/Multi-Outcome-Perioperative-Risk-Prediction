"""
Download the three encoders on the LOGIN node and pin their commit shas.

    python fetch_models.py            # download + report the shas
    python fetch_models.py --write    # ... and write them into config_encoder.py

Compute nodes generally have no route to huggingface.co, so the weights have to
be in the shared HF_HOME before anything is queued; env.sh then sets
HF_HUB_OFFLINE=1 so a job can never silently start a download and hang.

The commit sha matters. HuggingFace authors update weights in place under the
same name, so `abhinand/MedEmbed-large-v0.1` alone does not identify what was
actually run. embed.py refuses to start against an unpinned encoder.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_encoder  # noqa: E402


def fetch(slug, spec):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    print(f"\n=== {slug}  ({spec['hf_id']}) ===")
    path = snapshot_download(
        spec["hf_id"],
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
        ignore_patterns=["*.onnx", "*.onnx_data", "openvino*", "*.msgpack", "*.h5"],
    )
    sha = os.path.basename(os.path.realpath(path))
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=sha)
    model = AutoModel.from_pretrained(spec["hf_id"], revision=sha)
    dim = model.config.hidden_size
    pos = getattr(model.config, "max_position_embeddings", None)
    print(f"  revision  {sha}")
    print(f"  dim       {dim}   (config_encoder says {spec['dim']})")
    print(f"  max pos   {pos}   (config_encoder MAX_LEN {config_encoder.MAX_LEN})")
    print(f"  pooling   {spec['pooling']}")
    if dim != spec["dim"]:
        raise SystemExit(f"{slug}: hidden_size {dim} != config_encoder dim {spec['dim']}")
    if pos is not None and config_encoder.MAX_LEN > pos:
        raise SystemExit(
            f"{slug}: MAX_LEN {config_encoder.MAX_LEN} exceeds the model's {pos} positions")
    del model, tok
    return sha


def write_revisions(revs):
    """Patch the `revision=None` lines in config_encoder.py, in ENCODERS order."""
    p = os.path.join(HERE, "config_encoder.py")
    src = open(p).read()
    for slug, sha in revs.items():
        pat = re.compile(
            r'("' + re.escape(slug) + r'": dict\(\s*\n\s*hf_id="[^"]+",\s*\n\s*)'
            r'revision=(?:None|"[0-9a-f]+")')
        src, n = pat.subn(lambda m: f'{m.group(1)}revision="{sha}"', src, count=1)
        if n != 1:
            raise SystemExit(f"could not patch revision for {slug}")
    open(p, "w").write(src)
    print(f"\nwritten into {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="patch the resolved shas into config_encoder.ENCODERS")
    a = ap.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise SystemExit("HF_HUB_OFFLINE=1 -- unset it; this script must reach the hub")
    print(f"HF_HOME = {os.environ.get('HF_HOME', '<unset, using ~/.cache>')}")
    revs = {slug: fetch(slug, spec) for slug, spec in config_encoder.ENCODERS.items()}
    if a.write:
        write_revisions(revs)
    else:
        print("\nre-run with --write to pin these into config_encoder.py")


if __name__ == "__main__":
    main()
