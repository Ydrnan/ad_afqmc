import os
from pathlib import Path

import pytest

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked as slow integration coverage",
    )


def _is_slow_test(item: pytest.Item) -> bool:
    path = Path(str(item.fspath)).name
    test_name = getattr(item, "originalname", item.name.split("[", 1)[0])

    slow_files = {
        "test_rhf_fp.py",
        "test_uhf_fp.py",
        "test_cisd_fp.py",
        "test_ucisd_fp.py",
        "test_pt2ccsd.py",
        "test_io.py",
        "test_cis.py",
        "test_eom_cisd.py",
        "test_eom_t_cisd.py",
        "test_ucisdt_molecular.py",
    }
    if path in slow_files:
        return True

    if path == "test_lno.py":
        return test_name == "test_calc_rhf"

    if path == "test_ghf.py":
        return test_name == "test_calc_ghf_hamiltonian"

    if path == "test_gcisd.py":
        return test_name == "test_calc_ghf_hamiltonian"

    if path == "test_ucisd.py":
        return test_name == "test_calc_rhf_hamiltonian"

    if path == "test_rhf.py" and test_name == "test_calc_rhf_hamiltonian":
        callspec = getattr(item, "callspec", None)
        return callspec is not None and callspec.params.get("walker_kind") != "restricted"

    if path == "test_uhf.py" and test_name == "test_calc_rhf_hamiltonian":
        callspec = getattr(item, "callspec", None)
        return callspec is not None and callspec.params.get("walker_kind") != "restricted"

    return False


def _is_integration_test(item: pytest.Item) -> bool:
    path = Path(str(item.fspath)).name
    return path != "test_cis.py" and _is_slow_test(item)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    run_slow = config.getoption("--run-slow")

    for item in items:
        if not _is_slow_test(item):
            continue

        item.add_marker(pytest.mark.slow)
        if _is_integration_test(item):
            item.add_marker(pytest.mark.integration)

        if not run_slow:
            item.add_marker(skip_slow)
