"""The suite is offline: `--disable-socket` (pytest-socket, in pyproject.toml's addopts)
makes a test that opens a network socket or looks up a host name in the pytest process
fail, instead of quietly reaching the network. Subprocesses are not covered: the
scripts/vast/ runs rely on tests/vast_fakes.py instead, checked here too.
"""

import socket
import subprocess

import pytest
from pytest_socket import SocketBlockedError

from fragile_compassion.betley.fetch import fetch_text
from tests.vast_fakes import NETWORKED, Fakes


def test_a_test_cannot_open_a_network_socket():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_the_betley_fetcher_cannot_download_from_a_test():
    # conftest points FC_CACHE_DIR at an empty temp dir, so this is a cache miss.
    with pytest.raises(SocketBlockedError):
        fetch_text("first_plot")


def test_every_networked_command_is_a_fake_in_a_vast_script_run(tmp_path):
    fakes = Fakes(tmp_path)
    names = [*NETWORKED, "sleep", "python3"]
    out = subprocess.run(
        ["/bin/bash", "-c", 'for c in "$@"; do type -t "$c"; done; ssh vast-em true', "-", *names],
        env=fakes.env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.stdout.split() == ["function"] * len(names)
    assert out.returncode == 255  # ssh can't reach anything
    assert fakes.calls() == ["ssh vast-em true"]
