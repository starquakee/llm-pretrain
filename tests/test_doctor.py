from __future__ import annotations

import json

import pytest

from llm_pretrain import doctor


def test_doctor_returns_structured_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_platform_check",
        lambda: doctor.DoctorCheck("platform", True, "test Linux"),
    )
    monkeypatch.setattr(
        doctor,
        "_torch_checks",
        lambda _minimum: (doctor.DoctorCheck("torch", True, "test torch"),),
    )
    monkeypatch.setattr(doctor, "_memory_available_bytes", lambda: doctor.GIB)

    report = doctor.run_doctor(
        tmp_path,
        min_free_disk_gb=0,
        min_free_ram_gb=0,
        min_free_vram_gb=0,
        io_test_bytes=1024,
    )

    assert report.ready
    payload = json.loads(report.to_json())
    assert payload["ready"] is True
    assert {item["name"] for item in payload["checks"]} >= {
        "platform",
        "python",
        "torch",
        "disk",
        "data_io",
        "ram",
    }
    io_check = next(item for item in payload["checks"] if item["name"] == "data_io")
    assert io_check["details"]["write_mib_s"] > 0
    assert list(tmp_path.iterdir()) == []


def test_python_check_requires_python_312(monkeypatch) -> None:
    monkeypatch.setattr(doctor.sys, "version_info", (3, 11, 9))
    check = doctor._python_check()
    assert not check.ok
    assert "Python 3.12" in check.message


def test_failed_required_check_blocks_training() -> None:
    report = doctor.DoctorReport(
        (
            doctor.DoctorCheck("cuda", False, "not found"),
            doctor.DoctorCheck("optional", False, "ignored", required=False),
        )
    )

    assert not report.ready
    with pytest.raises(doctor.DoctorError, match="cuda: not found"):
        doctor.require_training_ready(report)


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        doctor.run_doctor(min_free_disk_gb=-1)
