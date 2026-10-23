"""Immutable raw snapshots and append-only research runs; local + Google Drive backends."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from .config import FIELDS


def checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parquet_bytes(df: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    df.to_parquet(stream, index=True, engine="pyarrow")
    return stream.getvalue()


class LocalStore:
    def __init__(self, raw_root: str | Path, result_root: str | Path):
        self.raw_root, self.result_root = Path(raw_root), Path(result_root)

    def _read(self, root, path):
        return (root / path).read_bytes() if (root / path).is_file() else None

    def _write_new(self, root, path, data):
        loc = root / path
        loc.parent.mkdir(parents=True, exist_ok=True)
        if loc.exists():
            if checksum(loc.read_bytes()) != checksum(data):
                raise FileExistsError(f"Immutable artifact conflict: {path}")
            return
        loc.write_bytes(data)

    def read_raw(self, path):
        return self._read(self.raw_root, path)

    def write_raw(self, path, data):
        self._write_new(self.raw_root, path, data)

    def write_result(self, path, data):
        self._write_new(self.result_root, path, data)


class GoogleDriveStore:
    """Google Drive API by verified folder IDs; identical interface to LocalStore.

    Requires interactive Colab `google.colab.auth.authenticate_user()` first.
    No service-account keys or OAuth tokens are written to the repository.
    """
    def __init__(self, raw_folder_id: str, results_folder_id: str):
        import google.auth
        from googleapiclient.discovery import build
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.file",
                                               "https://www.googleapis.com/auth/drive"])
        self.api = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.raw_root, self.result_root = raw_folder_id, results_folder_id

    def _find(self, parent_id, name):
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        q = f"'{parent_id}' in parents and name = '{escaped}' and trashed = false"
        result = self.api.files().list(q=q, fields="nextPageToken,files(id,name,mimeType,md5Checksum)",
                                       page_size=100, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        if result.get("nextPageToken"):
            raise RuntimeError("ambiguous large folder listing; refusing unsafe lookup")
        matches = result.get("files", [])
        if len(matches) > 1:
            raise RuntimeError("duplicate Drive names under same parent; refusing ambiguous lookup")
        return matches[0] if matches else None

    def _folder(self, root, segments, create):
        fid = root
        for name in segments:
            found = self._find(fid, name)
            if found is None:
                if not create:
                    return None
                found = self.api.files().create(body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [fid]},
                                                fields="id,name,mimeType", supportsAllDrives=True).execute()
            if found["mimeType"] != "application/vnd.google-apps.folder":
                raise ValueError("Expected Google Drive folder")
            fid = found["id"]
        return fid

    def _read(self, root, path):
        parts = path.split("/")
        folder = self._folder(root, parts[:-1], create=False)
        if not folder:
            return None
        found = self._find(folder, parts[-1])
        if found is None:
            return None
        from googleapiclient.http import MediaIoBaseDownload
        buf = io.BytesIO()
        request = self.api.files().get_media(fileId=found["id"])
        reader = MediaIoBaseDownload(buf, request)
        finished = False
        while not finished:
            _, finished = reader.next_chunk()
        data = buf.getvalue()
        if found.get("md5Checksum") and hashlib.md5(data).hexdigest() != found["md5Checksum"]:
            raise ValueError("Drive integrity check failed")
        return data

    def _write_new(self, root, path, data):
        parts = path.split("/")
        folder = self._folder(root, parts[:-1], create=True)
        existing = self._find(folder, parts[-1])
        if existing:
            if checksum(self._read(root, path)) != checksum(data):
                raise FileExistsError(f"Immutable artifact conflict: {path}")
            return
        from googleapiclient.http import MediaIoBaseUpload
        mime = "application/octet-stream" if path.endswith(".parquet") else ("application/json" if path.endswith(".json") else "text/plain")
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=True)
        self.api.files().create(body={"name": parts[-1], "parents": [folder]}, media_body=media,
                                fields="id", supportsAllDrives=True).execute()
        if checksum(self._read(root, path)) != checksum(data):
            raise ValueError("Drive upload readback checksum failed")

    def read_raw(self, path):
        return self._read(self.raw_root, path)

    def write_raw(self, path, data):
        self._write_new(self.raw_root, path, data)

    def write_result(self, path, data):
        self._write_new(self.result_root, path, data)


def snapshot_path(session: str, start: str, end: str, revision: str | None = None):
    if session not in ("full_session", "afterhours"):
        raise ValueError("invalid session")
    base = f"{session}/{start}_{end}"
    return base if revision is None else base + "__revision_" + revision


def read_snapshot(store, path: str):
    metadata = store.read_raw(path + "/manifest.json")
    if metadata is None:
        return None
    manifest = json.loads(metadata)
    if manifest.get("schema_version") != "1.0" or set(manifest.get("files", {})) != set(FIELDS):
        raise ValueError("Invalid cached raw manifest")
    frames = {}
    for field, digest in manifest["files"].items():
        data = store.read_raw(f"{path}/{field}.parquet")
        if data is None or checksum(data) != digest:
            raise ValueError("Cached raw missing/corrupted: " + field)
        frames[field] = pd.read_parquet(io.BytesIO(data))
    return frames


def save_snapshot(store, path: str, frames, *, dataset: str, start: str, end: str):
    if store.read_raw(path + "/manifest.json") is not None:
        raise FileExistsError("Snapshot already committed; use cache or new revision")
    file_hashes = {}
    for field in FIELDS:
        data = parquet_bytes(frames[field])
        store.write_raw(f"{path}/{field}.parquet", data)
        file_hashes[field] = checksum(data)
    manifest = {"schema_version": "1.0", "dataset": dataset, "start": start, "end": end,
                "files": file_hashes, "source": "FinLab data.get", "unit_count": "contracts", "unit_value": "thousand_NTD",
                "saved_at_utc": datetime.now(timezone.utc).isoformat()}
    # Publish manifest LAST; partial uploads are not accepted as valid snapshots.
    raw = json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    store.write_raw(path + "/manifest.json", raw)
    return manifest
