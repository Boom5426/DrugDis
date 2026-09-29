"""Relative data paths in the substrate config resolve against DRUGDIS_DATA."""

import json
import os

import pytest

import paths


def test_config_paths_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUGDIS_DATA", str(tmp_path))
    cfg = paths.load_config(paths.CONFIG)
    for key in paths.CONFIG_PATH_KEYS:
        assert os.path.isabs(cfg[key])
        assert cfg[key].startswith(str(tmp_path.resolve()))
    raw = json.load(open(paths.CONFIG))
    assert {k: v for k, v in raw.items() if k not in paths.CONFIG_PATH_KEYS} == \
        {k: v for k, v in cfg.items() if k not in paths.CONFIG_PATH_KEYS}


def test_absolute_paths_are_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUGDIS_DATA", str(tmp_path))
    cfg = paths.resolve_config({"master_table_path": "/elsewhere/master.parquet"})
    assert cfg["master_table_path"] == "/elsewhere/master.parquet"


def test_missing_data_root_is_an_error(monkeypatch):
    monkeypatch.delenv("DRUGDIS_DATA", raising=False)
    with pytest.raises(RuntimeError):
        paths.data_dir()


def test_frozen_manifests_are_shipped():
    man = json.load(open(paths.MANIFESTS / "MANIFEST.json"))
    for key in man["manifest_sha256"]:
        assert (paths.MANIFESTS / f"{key}.json").exists()
