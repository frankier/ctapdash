"""Browser smoke coverage of every page and each rendered content family."""
import pytest
from playwright.sync_api import expect


PARTICIPANT = "1001P_dummy_intake"
QUERY = f"?source=dummy&participant={PARTICIPANT}"


def visit(page, dashboard_url, path):
    response = page.goto(dashboard_url + path)
    assert response.status == 200


def test_home_dataset_and_participant_selection(page, dashboard_url):
    visit(page, dashboard_url, "/")
    page.get_by_label("Dataset", exact=True).select_option("dummy")
    expect(page.get_by_role("heading", name="Dataset overview")).to_be_visible()
    expect(page.get_by_role("table")).to_contain_text("3_fine_clean")
    page.get_by_role("link", name="Participants", exact=True).click()
    expect(page.get_by_role("heading", name="Participants")).to_be_visible()
    page.get_by_label("Participant", exact=True).select_option(PARTICIPANT)
    expect(page.get_by_role("columnheader", name="Observations")).to_be_visible()
    expect(page.locator("#channel-statistics-table table")).to_be_visible(timeout=30000)
    page.locator("#statistics-step").select_option("2")
    expect(page.locator("#channel-statistics-table").get_by_role("columnheader", name="Step", exact=True)).to_have_count(0)
    expect(page.locator("#channel-statistics-table td[aria-label]").first).to_be_visible()


def test_setup(page, dashboard_url):
    visit(page, dashboard_url, "/setup")
    expect(page.get_by_role("heading", name="Data sources")).to_be_visible()
    expect(page.get_by_role("table")).to_contain_text("dummy")


def test_statistics_fragment(page, dashboard_url):
    visit(page, dashboard_url, "/participant/statistics" + QUERY)
    expect(page.get_by_role("table")).to_be_visible()
    expect(page.get_by_role("rowheader", name="Mean", exact=True)).to_have_count(3)


@pytest.mark.parametrize("participant,step", [(PARTICIPANT, "1"), ("1002P_dummy_intake", "3")])
def test_classic_eeg(page, dashboard_url, participant, step):
    visit(page, dashboard_url, f"/participant/overview?source=dummy&participant={participant}")
    page.get_by_role("link", name="Classic EEG viewer").click()
    expect(page.locator("#step-select")).to_be_visible()
    page.locator("#step-select").select_option(step)
    expect(page.locator("canvas").first).to_be_visible(timeout=30000)
    expect(page.locator("#yaxis-select")).to_be_visible()


def test_venndiff_eeg(page, dashboard_url):
    visit(page, dashboard_url, "/participant/overview" + QUERY)
    page.get_by_role("link", name="Venndiff EEG viewer").click()
    expect(page.locator(".bokeh-host canvas").first).to_be_visible(timeout=60000)
    expect(page.get_by_text("Step A (red)", exact=True)).to_be_visible()
    expect(page.get_by_text("Step B (blue)", exact=True)).to_be_visible()


@pytest.mark.parametrize("peek,count", [("CTAP_peek_data", 2), ("CTAP_blink2event", 1)])
def test_quality_control_images(page, dashboard_url, peek, count):
    visit(page, dashboard_url, "/participant/overview" + QUERY)
    page.get_by_role("link", name="Quality control (peeks)").click()
    page.locator("#peek-select").select_option(peek)
    images = page.locator("#peek-content img")
    expect(images).to_have_count(count)
    for image in images.all():
        expect(image).to_be_visible()
        expect(image).to_have_js_property("complete", True)
        assert image.evaluate("img => img.naturalWidth") > 0


def test_logs(page, dashboard_url):
    visit(page, dashboard_url, "/participant/overview" + QUERY)
    page.get_by_role("link", name="Logs", exact=True).click()
    page.locator("#log-select").select_option(f"logs/CTAP_load_data/{PARTICIPANT}.log")
    expect(page.locator("#log-content pre")).to_contain_text("Processing completed.")
