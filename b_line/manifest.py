"""Data manifest and train/val/test split (60/20/20 by speaker/source)."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List


def create_manifest(entries: List[dict]) -> List[dict]:
    """Normalize a list of {id, path, speaker, source, duration_sec, noisy_of}."""
    out = []
    for e in entries:
        out.append({
            "id": e["id"],
            "path": e["path"],
            "speaker": e.get("speaker", "unknown"),
            "source": e.get("source", "unknown"),
            "duration_sec": float(e.get("duration_sec", 0.0)),
            "noisy_of": e.get("noisy_of"),   # id of the clean clip this is derived from
        })
    return out


def split_manifest(entries: List[dict], train=0.6, val=0.2, test=0.2,
                   group_by: str = "speaker", seed: int = 0) -> Dict[str, List[dict]]:
    """Split by group so noisy versions stay with their clean counterpart."""
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        groups[str(e.get(group_by, "unknown"))].append(e)
    keys = list(groups.keys())
    rng.shuffle(keys)

    train_s, val_s, test_s = [], [], []
    total = sum(e["duration_sec"] for e in entries) or float(len(entries))
    acc_train = acc_val = 0.0
    for k in keys:
        g = groups[k]
        gdur = sum(e["duration_sec"] for e in g) or 1.0
        if acc_train / total < train:
            train_s.extend(g)
            acc_train += gdur
        elif (acc_train + acc_val) / total < train + val:
            val_s.extend(g)
            acc_val += gdur
        else:
            test_s.extend(g)
    return {"train": train_s, "val": val_s, "test": test_s}


def save_manifest(splits: Dict[str, List[dict]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
