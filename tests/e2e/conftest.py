"""Run the actual CLI in an isolated process with temporary data and caches."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from tests.dummy_data import write_dataset


@pytest.fixture(scope="session")
def dashboard_url(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dashboard")
    config = write_dataset(directory)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, MPLCONFIGDIR=str(directory / "matplotlib"))
    env["_MNE_FAKE_HOME_DIR"] = str(directory)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    with (directory / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "ctapdash", "--config", str(config),
             "--no-window", "--no-browser", "--port", str(port)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(url + "/setup", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                log.seek(0)
                pytest.fail("Dashboard failed to start:\n" + log.read())
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture(scope="module")
def e2e_browser(browser_name, pytestconfig):
    # The plugin's session-scoped sync Playwright loop stays active during the
    # unit tests, preventing their asyncio.run() calls. Close it after this module.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = getattr(playwright, browser_name).launch(
            headless=not pytestconfig.getoption("--headed")
        )
        yield browser
        browser.close()


@pytest.fixture
def page(e2e_browser):
    context = e2e_browser.new_context()
    page = context.new_page()
    yield page
    context.close()
