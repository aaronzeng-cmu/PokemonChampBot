"""Memory-mapped BC dataset I/O (avoids loading multi-GB .pt into RAM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

DOUBLES_ARRAYS = (
    "token_ids",
    "action_slot0",
    "action_slot1",
    "mask_slot0",
    "mask_slot1",
)


def mmap_dataset_dir(pt_path: Path) -> Path:
    return pt_path.with_name(f"{pt_path.stem}_mmap")


def _chunk_path(mmap_dir: Path, chunk_ref: str) -> Path:
    """Resolve a manifest chunk path (e.g. chunks/chunk_0000)."""
    return mmap_dir / chunk_ref


def _chunk_ref(root_dir: Path, chunk_dir: Path) -> str:
    return chunk_dir.relative_to(root_dir).as_posix()


def save_mmap_dataset(dataset: dict, out_dir: Path) -> Path:
    """Write numpy arrays to out_dir for mmap training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in DOUBLES_ARRAYS:
        arr = np.asarray(dataset[key])
        np.save(out_dir / f"{key}.npy", arr)
    meta = {
        "format": "doubles",
        "samples": int(np.asarray(dataset["token_ids"]).shape[0]),
        "arrays": list(DOUBLES_ARRAYS),
        "chunks": [out_dir.name],
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def save_mmap_chunk(dataset: dict, chunk_dir: Path, *, chunk_name: str) -> int:
    """Save one parse chunk; returns sample count."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(dataset["token_ids"]).shape[0])
    if n == 0:
        return 0
    for key in DOUBLES_ARRAYS:
        np.save(chunk_dir / f"{key}.npy", np.asarray(dataset[key]))
    meta_rows = dataset.get("meta")
    if meta_rows is not None:
        (chunk_dir / "meta.json").write_text(
            json.dumps(meta_rows, ensure_ascii=False), encoding="utf-8"
        )
    (chunk_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "doubles",
                "samples": n,
                "name": chunk_name,
                "has_meta": meta_rows is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return n


def write_mmap_manifest(root_dir: Path, chunk_dirs: list[Path]) -> None:
    total = 0
    chunks: list[dict] = []
    for chunk_dir in chunk_dirs:
        meta = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
        n = int(meta["samples"])
        total += n
        chunks.append({"dir": _chunk_ref(root_dir, chunk_dir), "samples": n})
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "doubles",
                "samples": total,
                "arrays": list(DOUBLES_ARRAYS),
                "chunks": chunks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def has_mmap_dataset(out_dir: Path) -> bool:
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return False
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    chunks = meta.get("chunks")
    if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
        return all(
            (_chunk_path(out_dir, c["dir"]) / f"{key}.npy").is_file()
            for c in chunks
            for key in DOUBLES_ARRAYS
        )
    return all((out_dir / f"{key}.npy").is_file() for key in DOUBLES_ARRAYS)


def export_pt_to_mmap(pt_path: Path, out_dir: Path | None = None) -> Path:
    """One-time convert bc_dataset.pt → mmap .npy files."""
    out_dir = out_dir or mmap_dataset_dir(pt_path)
    if has_mmap_dataset(out_dir):
        pt_mtime = pt_path.stat().st_mtime if pt_path.is_file() else 0
        if out_dir.stat().st_mtime >= pt_mtime:
            return out_dir
    print(f"Loading {pt_path} (mmap if supported)...", flush=True)
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(f"Writing mmap arrays to {out_dir}...", flush=True)
    save_mmap_dataset(data, out_dir)
    del data
    return out_dir


class MmapDoublesDataset(Dataset):
    def __init__(self, mmap_dir: Path, indices: np.ndarray) -> None:
        self._indices = np.asarray(indices, dtype=np.int64)
        self._tokens = np.load(mmap_dir / "token_ids.npy", mmap_mode="r")
        self._y0 = np.load(mmap_dir / "action_slot0.npy", mmap_mode="r")
        self._y1 = np.load(mmap_dir / "action_slot1.npy", mmap_mode="r")
        self._m0 = np.load(mmap_dir / "mask_slot0.npy", mmap_mode="r")
        self._m1 = np.load(mmap_dir / "mask_slot1.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self._indices[i])
        return (
            torch.as_tensor(self._tokens[j], dtype=torch.long),
            torch.as_tensor(self._y0[j], dtype=torch.long),
            torch.as_tensor(self._y1[j], dtype=torch.long),
            torch.as_tensor(self._m0[j], dtype=torch.bool),
            torch.as_tensor(self._m1[j], dtype=torch.bool),
        )


class _ChunkIndexedDataset(Dataset):
    """Map global indices into one mmap chunk."""

    def __init__(self, chunk_dir: Path, global_indices: np.ndarray) -> None:
        self._global = np.asarray(global_indices, dtype=np.int64)
        self._tokens = np.load(chunk_dir / "token_ids.npy", mmap_mode="r")
        self._y0 = np.load(chunk_dir / "action_slot0.npy", mmap_mode="r")
        self._y1 = np.load(chunk_dir / "action_slot1.npy", mmap_mode="r")
        self._m0 = np.load(chunk_dir / "mask_slot0.npy", mmap_mode="r")
        self._m1 = np.load(chunk_dir / "mask_slot1.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self._global.shape[0])

    def __getitem__(self, i: int):
        j = int(self._global[i])
        return (
            torch.as_tensor(self._tokens[j], dtype=torch.long),
            torch.as_tensor(self._y0[j], dtype=torch.long),
            torch.as_tensor(self._y1[j], dtype=torch.long),
            torch.as_tensor(self._m0[j], dtype=torch.bool),
            torch.as_tensor(self._m1[j], dtype=torch.bool),
        )


def load_doubles_sample_count(mmap_dir: Path) -> int:
    manifest = json.loads((mmap_dir / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["samples"])


def build_chunked_datasets(
    mmap_dir: Path, train_idx: np.ndarray, val_idx: np.ndarray
) -> tuple[Dataset, Dataset]:
    meta = json.loads((mmap_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks = meta.get("chunks") or []
    if not chunks or not isinstance(chunks[0], dict):
        return MmapDoublesDataset(mmap_dir, train_idx), MmapDoublesDataset(mmap_dir, val_idx)

    def _split(indices: np.ndarray) -> Dataset:
        parts: list[Dataset] = []
        offset = 0
        for chunk in chunks:
            n = int(chunk["samples"])
            chunk_dir = _chunk_path(mmap_dir, chunk["dir"])
            mask = (indices >= offset) & (indices < offset + n)
            local = indices[mask] - offset
            if local.size:
                parts.append(_ChunkIndexedDataset(chunk_dir, local))
            offset += n
        if not parts:
            raise ValueError("empty index split for chunked mmap dataset")
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    return _split(train_idx), _split(val_idx)


class _MmapChunkStore:
    def __init__(self, chunk_dir: Path, offset: int, n: int) -> None:
        self.chunk_dir = chunk_dir
        self.offset = offset
        self.n = n
        self._tokens: np.ndarray | None = None
        self._y0: np.ndarray | None = None
        self._y1: np.ndarray | None = None
        self._meta: list[dict] | None = None

    def _ensure_arrays(self) -> None:
        if self._tokens is None:
            self._tokens = np.load(self.chunk_dir / "token_ids.npy", mmap_mode="r")
            self._y0 = np.load(self.chunk_dir / "action_slot0.npy", mmap_mode="r")
            self._y1 = np.load(self.chunk_dir / "action_slot1.npy", mmap_mode="r")

    def _ensure_meta(self) -> None:
        if self._meta is None:
            meta_path = self.chunk_dir / "meta.json"
            if meta_path.is_file():
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                self._meta = [{} for _ in range(self.n)]

    def contains(self, global_idx: int) -> bool:
        return self.offset <= global_idx < self.offset + self.n

    def local_index(self, global_idx: int) -> int:
        return global_idx - self.offset

    def tokens_row(self, global_idx: int) -> torch.Tensor:
        self._ensure_arrays()
        j = self.local_index(global_idx)
        return torch.as_tensor(self._tokens[j], dtype=torch.long)  # type: ignore[index]

    def actions(self, global_idx: int) -> tuple[int, int]:
        self._ensure_arrays()
        j = self.local_index(global_idx)
        return int(self._y0[j]), int(self._y1[j])  # type: ignore[index]

    def meta_row(self, global_idx: int) -> dict:
        self._ensure_meta()
        j = self.local_index(global_idx)
        return self._meta[j]  # type: ignore[index]


class DoublesMmapStore:
    """Low-RAM random access to chunked doubles BC data (+ per-chunk meta.json)."""

    def __init__(self, mmap_dir: Path) -> None:
        if not has_mmap_dataset(mmap_dir):
            raise FileNotFoundError(f"No mmap dataset at {mmap_dir}")
        root = json.loads((mmap_dir / "manifest.json").read_text(encoding="utf-8"))
        chunks = root.get("chunks") or []
        if not chunks or not isinstance(chunks[0], dict):
            raise ValueError(f"Expected chunked mmap dataset at {mmap_dir}")
        self.mmap_dir = mmap_dir
        self._chunks: list[_MmapChunkStore] = []
        offset = 0
        for chunk in chunks:
            n = int(chunk["samples"])
            self._chunks.append(
                _MmapChunkStore(_chunk_path(mmap_dir, chunk["dir"]), offset, n)
            )
            offset += n
        self.n_samples = offset

    def __len__(self) -> int:
        return self.n_samples

    def has_meta(self) -> bool:
        return (self._chunks[0].chunk_dir / "meta.json").is_file()

    def _chunk_for(self, global_idx: int) -> _MmapChunkStore:
        for chunk in self._chunks:
            if chunk.contains(global_idx):
                return chunk
        raise IndexError(global_idx)

    def get(self, global_idx: int) -> dict:
        chunk = self._chunk_for(global_idx)
        y0, y1 = chunk.actions(global_idx)
        return {
            "tokens": chunk.tokens_row(global_idx),
            "action_slot0": y0,
            "action_slot1": y1,
            "meta": chunk.meta_row(global_idx),
        }

    def get_batch_tensors(
        self, indices: list[int] | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        rows = [self.get(int(i)) for i in indices]
        tokens = torch.stack([r["tokens"] for r in rows])
        y0 = torch.tensor([r["action_slot0"] for r in rows], dtype=torch.long)
        y1 = torch.tensor([r["action_slot1"] for r in rows], dtype=torch.long)
        meta = [r["meta"] for r in rows]
        return tokens, y0, y1, meta


def open_doubles_mmap_store(dataset_path: Path) -> DoublesMmapStore:
    return DoublesMmapStore(mmap_dataset_dir(dataset_path))


def load_doubles_bc_data(dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Load doubles BC arrays; prefers mmap chunks when available."""
    mmap_dir = mmap_dataset_dir(dataset_path)
    if has_mmap_dataset(mmap_dir):
        store = DoublesMmapStore(mmap_dir)
        idx = list(range(len(store)))
        return store.get_batch_tensors(idx)
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    tokens = torch.as_tensor(data["token_ids"], dtype=torch.long)
    y0 = torch.as_tensor(data["action_slot0"], dtype=torch.long)
    y1 = torch.as_tensor(data["action_slot1"], dtype=torch.long)
    meta: list[dict] = data["meta"]
    return tokens, y0, y1, meta
