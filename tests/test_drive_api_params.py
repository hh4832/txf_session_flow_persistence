"""Regression tests for Google Drive v3 keyword case and lookup safety.

Google's discovery client rejects snake_case `page_size`; it requires `pageSize`.
These tests run without Google OAuth credentials or network access.
"""
import pytest
from txf_session_flow_persistence.storage import GoogleDriveStore


class StrictFiles:
    def __init__(self, matches=(), next_token=None):
        self.matches = list(matches)
        self.next_token = next_token
        self.calls = []

    def list(self, **kwargs):
        allowed = {
            "q", "fields", "pageSize", "supportsAllDrives",
            "includeItemsFromAllDrives",
        }
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError("Got an unexpected keyword argument " + sorted(unexpected)[0])
        assert kwargs["pageSize"] == 100
        assert kwargs["supportsAllDrives"] is True
        assert kwargs["includeItemsFromAllDrives"] is True
        self.calls.append(kwargs)
        payload = {"files": self.matches}
        if self.next_token is not None:
            payload["nextPageToken"] = self.next_token

        class Response:
            def execute(self_inner):
                return payload

        return Response()


class FakeAPI:
    def __init__(self, files):
        self._files = files

    def files(self):
        return self._files


def drive_store(files):
    store = GoogleDriveStore.__new__(GoogleDriveStore)
    store.api = FakeAPI(files)
    store.raw_root = "verified-raw-folder-id"
    store.result_root = "verified-results-folder-id"
    return store


def test_missing_snapshot_lookup_uses_drive_v3_pageSize():
    files = StrictFiles()
    store = drive_store(files)
    assert store.read_raw("full_session/2023-10-16_2026-10-22/manifest.json") is None
    assert len(files.calls) == 1
    assert "'verified-raw-folder-id' in parents" in files.calls[0]["q"]


def test_existing_folder_and_file_lookups():
    files = StrictFiles([{"id": "folder-1", "name": "afterhours",
                          "mimeType": "application/vnd.google-apps.folder"}])
    store = drive_store(files)
    assert store._find("parent-1", "afterhours")["id"] == "folder-1"
    assert store._folder("parent-1", ["afterhours"], create=False) == "folder-1"


def test_drive_lookup_fails_closed_on_duplicates_or_pagination():
    entry = {"id": "x", "name": "manifest.json", "mimeType": "application/json"}
    with pytest.raises(RuntimeError, match="duplicate"):
        drive_store(StrictFiles([entry, {**entry, "id": "y"}]))._find("p", "manifest.json")
    with pytest.raises(RuntimeError, match="ambiguous large folder"):
        drive_store(StrictFiles([entry], next_token="next"))._find("p", "manifest.json")


def test_drive_query_escapes_quotes():
    files = StrictFiles()
    drive_store(files)._find("root", "test's data")
    assert "test\\'s data" in files.calls[0]["q"]
