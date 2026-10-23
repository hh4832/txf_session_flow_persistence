"""Regression for the editable src-layout import failure in a live Colab kernel."""
import json
import os
import subprocess
import sys
from pathlib import Path


def test_colab_bootstrap_installs_into_kernel_and_checks_resolved_source():
    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/run_txf_session_flow_persistence.ipynb").read_text(encoding="utf-8"))
    bootstrap = "".join(notebook["cells"][1]["source"])
    assert "sys.executable" in bootstrap
    assert 'sys.path.insert(0, src_path)' in bootstrap
    assert 'importlib.invalidate_caches()' in bootstrap
    assert 'importlib.import_module("txf_session_flow_persistence.config")' in bootstrap
    assert 'Path(config_module.__file__).resolve() != expected_config' in bootstrap
    assert '["git", "-C", str(repo_dir), "merge", "--ff-only", "FETCH_HEAD"]' in bootstrap
    assert 'if not repo_dir.exists()' not in bootstrap


def test_namespace_package_shadow_is_repaired_by_in_process_src_path(tmp_path):
    """Before the path fix the repo name shadows the package; afterwards config loads."""
    checkout = tmp_path / "txf_session_flow_persistence"
    pkg = checkout / "src" / "txf_session_flow_persistence"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "config.py").write_text("VALUE = 21\n", encoding="utf-8")
    code = """
import importlib, sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo.parent))
import txf_session_flow_persistence
assert txf_session_flow_persistence.__file__ is None  # namespace package
try:
    import txf_session_flow_persistence.config
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Expected namespace-package shadow')
source = str((repo / 'src').resolve())
sys.path.insert(0, source)
for name in list(sys.modules):
    if name == 'txf_session_flow_persistence' or name.startswith('txf_session_flow_persistence.'):
        del sys.modules[name]
importlib.invalidate_caches()
cfg = importlib.import_module('txf_session_flow_persistence.config')
assert cfg.VALUE == 21
assert Path(cfg.__file__).resolve() == (repo / 'src/txf_session_flow_persistence/config.py').resolve()
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-S", "-c", code, str(checkout)], check=True, cwd=tmp_path, env=env)
