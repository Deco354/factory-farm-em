"""The suite is offline: `--disable-socket` (pytest-socket, in pyproject.toml's addopts)
makes any test that opens a network socket or looks up a host name fail, instead of
quietly reaching the network.
"""

import socket

import pytest
from pytest_socket import SocketBlockedError

from fragile_compassion.betley.fetch import fetch_text


def test_a_test_cannot_open_a_network_socket():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_the_betley_fetcher_cannot_download_from_a_test():
    # conftest points FC_CACHE_DIR at an empty temp dir, so this is a cache miss.
    with pytest.raises(SocketBlockedError):
        fetch_text("first_plot")
