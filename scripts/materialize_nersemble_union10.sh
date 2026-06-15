#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
DATA_ROOT="$ROOT/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed"
OUT_ROOT="$DATA_ROOT/union10_bundles"
INVENTORY_TSV="$DATA_ROOT/nersemble_subjects.tsv"
MAP_TSV="$DATA_ROOT/nersemble_avatar_map.tsv"

mkdir -p "$OUT_ROOT"

python - "$DATA_ROOT" "$OUT_ROOT" "$INVENTORY_TSV" "$MAP_TSV" <<'PY'
from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

data_root = Path(__import__("sys").argv[1]).resolve()
out_root = Path(__import__("sys").argv[2]).resolve()
inventory_tsv = Path(__import__("sys").argv[3]).resolve()
map_tsv = Path(__import__("sys").argv[4]).resolve()

archives = {
    "074": data_root / "release" / "074.zip",
    "165": data_root / "release" / "more" / "165.zip",
    "218": data_root / "release" / "218.zip",
}

subject_rows: list[tuple[str, str]] = []
map_rows: list[tuple[str, str, int]] = []

for offset, sid in enumerate(sorted(archives.keys()), start=2001):
    archive = archives[sid]
    if not archive.exists():
        raise SystemExit(f"archive not found: {archive}")

    bundle_root = out_root / sid
    bundle_root.mkdir(parents=True, exist_ok=True)
    print(f"[materialize] subject={sid} archive={archive}")

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        union_root = None
        for name in names:
            m = re.search(rf"(.*?/UNION10_{sid}_[^/]+)/canonical_flame_param\.npz$", name)
            if m:
                union_root = m.group(1)
                break
        if union_root is None:
            raise SystemExit(f"UNION10 root not found for {sid}")

        union_name = Path(union_root).name
        export_root = union_root.rsplit("/", 1)[0]
        export_prefix = export_root + "/"

        # Extract UNION10 metadata dir itself into a temp area first using unzip, then strip the long prefix.
        stage_root = bundle_root / ".stage"
        if stage_root.exists():
            shutil.rmtree(stage_root)
        stage_root.mkdir(parents=True, exist_ok=True)
        print(f"[materialize] extract union10 metadata: {union_name}")
        subprocess.run(
            ["unzip", "-q", "-o", str(archive), union_root + "/*", "-d", str(stage_root)],
            check=True,
        )

        union_dir = bundle_root / union_name
        staged_union = stage_root / export_root / union_name
        union_dir.parent.mkdir(parents=True, exist_ok=True)
        if union_dir.exists():
            shutil.rmtree(union_dir)
        shutil.move(str(staged_union), str(union_dir))
        train_json = json.loads((union_dir / "transforms_train.json").read_text())
        val_json = json.loads((union_dir / "transforms_val.json").read_text())
        test_json = json.loads((union_dir / "transforms_test.json").read_text())

        referenced_dirs: set[str] = set()
        for payload in (train_json, val_json, test_json):
            for frame in payload["frames"]:
                parts = Path(frame["file_path"]).parts
                if len(parts) >= 2:
                    referenced_dirs.add(parts[1])

        for dirname in sorted(referenced_dirs):
            print(f"[materialize] extract referenced dir: {dirname}")
            prefix = export_root + f"/{dirname}/"
            if not any(n.startswith(prefix) for n in names):
                raise SystemExit(f"referenced dir not found in archive: {dirname} ({sid})")

            subprocess.run(
                ["unzip", "-q", "-o", str(archive), prefix + "*", "-d", str(stage_root)],
                check=True,
            )
            staged_dir = stage_root / export_root / dirname
            target_dir = bundle_root / dirname
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(staged_dir), str(target_dir))

        shutil.rmtree(stage_root)
        print(f"[materialize] subject done: {sid}")

    source_dir = str((bundle_root / Path(union_root).name).resolve())
    subject_id = f"nersemble_{sid}_union10"
    subject_rows.append((subject_id, source_dir))
    map_rows.append((sid, source_dir, offset))

inventory_tsv.parent.mkdir(parents=True, exist_ok=True)
with open(inventory_tsv, "w", encoding="utf-8") as f:
    f.write("subject_id\tsource_dir\n")
    for subject_id, source_dir in subject_rows:
        f.write(f"{subject_id}\t{source_dir}\n")

with open(map_tsv, "w", encoding="utf-8") as f:
    f.write("subject\tsource_dir\tsuggested_avatar_id\n")
    for sid, source_dir, avatar_id in map_rows:
        f.write(f"{sid}\t{source_dir}\t{avatar_id}\n")

print(f"wrote {inventory_tsv}")
print(f"wrote {map_tsv}")
for sid, source_dir, avatar_id in map_rows:
    print(f"{sid}\t{source_dir}\t{avatar_id}")
PY
