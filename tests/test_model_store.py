import pytest

from autosubtitler import model_store
from autosubtitler.model_store import MoveError, move_models


@pytest.fixture
def models(tmp_path):
    src = tmp_path / "models"
    (src / "whisper" / "m").mkdir(parents=True)
    (src / "whisper" / "m" / "model.bin").write_bytes(b"w" * 1000)
    (src / "nllb").mkdir()
    (src / "nllb" / "weights").write_bytes(b"n" * 500)
    return src


def assert_moved(src, dst):
    assert (dst / "whisper" / "m" / "model.bin").read_bytes() == b"w" * 1000
    assert (dst / "nllb" / "weights").read_bytes() == b"n" * 500
    assert not (src / "whisper").exists() and not (src / "nllb").exists()


def test_move_same_drive(models, tmp_path):
    dst = tmp_path / "new" / "place"
    move_models(models, dst)
    assert_moved(models, dst)
    assert not models.exists()  # emptied source folder is removed


def test_move_across_drives_copies_with_progress(models, tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "_same_device", lambda a, b: False)
    seen = []
    dst = tmp_path / "other"
    move_models(models, dst, seen.append)
    assert_moved(models, dst)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_unrelated_files_are_left_alone(models, tmp_path):
    (models / "notes.txt").write_text("keep me")
    move_models(models, tmp_path / "dst")
    assert (models / "notes.txt").read_text() == "keep me"


def test_failed_copy_rolls_back(models, tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "_same_device", lambda a, b: False)

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(model_store.shutil, "copystat", boom)
    dst = tmp_path / "dst"
    with pytest.raises(MoveError):
        move_models(models, dst)
    assert (models / "whisper" / "m" / "model.bin").exists()
    assert not (dst / "whisper").exists()


@pytest.mark.parametrize("make_dst, message", [
    (lambda src, tmp: src, "already in that folder"),
    (lambda src, tmp: src / "sub", "inside"),
])
def test_refused_destinations(models, tmp_path, make_dst, message):
    with pytest.raises(MoveError, match=message):
        move_models(models, make_dst(models, tmp_path))


def test_refuses_to_overwrite_existing_models(models, tmp_path):
    (tmp_path / "dst" / "nllb").mkdir(parents=True)
    with pytest.raises(MoveError, match="already exists"):
        move_models(models, tmp_path / "dst")
    assert (models / "nllb" / "weights").exists()


def test_nothing_to_move(tmp_path):
    with pytest.raises(MoveError, match="No downloaded models"):
        move_models(tmp_path / "empty", tmp_path / "dst")
