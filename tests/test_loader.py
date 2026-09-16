from pathlib import Path

import pytest

from fraud_detection.data.loader import DataLoader, DatasetConfig


def loader(tmp_path: Path) -> DataLoader:
    return DataLoader(DatasetConfig(raw_dir=tmp_path, file_name="data.csv"))


def test_sample_size_none_loads_all_rows_and_zero_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("x,fraud\n1,0\n2,1\n", encoding="utf-8")

    assert len(loader(tmp_path).load(sample_size=None)) == 2
    with pytest.raises(ValueError, match="positive integer"):
        loader(tmp_path).load(sample_size=0)


@pytest.mark.parametrize("sample_size", [-1, True, False, 1.5])
def test_invalid_sample_sizes_fail(tmp_path: Path, sample_size) -> None:
    (tmp_path / "data.csv").write_text("x,fraud\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        loader(tmp_path).load(sample_size=sample_size)


def test_empty_csv_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        loader(tmp_path).load()
