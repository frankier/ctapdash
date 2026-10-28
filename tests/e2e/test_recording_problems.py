"""One damaged input must not take down the dataset or its healthy steps."""

import pytest
from playwright.sync_api import expect


@pytest.mark.parametrize("dashboard_url", ["damaged"], indirect=True)
def test_recording_problems(page, dashboard_url):
    response = page.goto(dashboard_url + "/overview?source=dummy")
    assert response.status == 200
    expect(page.get_by_role("rowheader", name="mixed Has problems")).to_be_visible()
    expect(page.get_by_role("rowheader", name="broken Has problems")).to_be_visible()
    expect(page.get_by_text("1_load/mixed.set", exact=False)).to_be_visible()
    expect(page.get_by_text("(incomplete)", exact=False)).to_have_count(2)
    expect(page.get_by_text("._mixed", exact=False)).to_have_count(0)

    query = "?source=dummy&participant=mixed"
    response = page.goto(dashboard_url + "/participant/overview" + query)
    assert response.status == 200
    expect(
        page.get_by_text("Has problems: some processing steps are unavailable.")
    ).to_be_visible()
    expect(page.locator("#participant-select")).not_to_contain_text("._mixed")
    expect(page.locator("#channel-statistics-table table")).to_be_visible()
    expect(page.locator("#channel-statistics-table")).to_contain_text(
        "results are incomplete"
    )
    page.locator("#statistics-step").select_option("1")
    expect(page.locator("#channel-statistics-table")).to_contain_text(
        "No usable recordings"
    )
    page.locator("#statistics-step").select_option("2")
    expect(page.locator("#channel-statistics-table table")).to_be_visible()
    expect(page.locator("#channel-statistics-table")).not_to_contain_text(
        "results are incomplete"
    )

    response = page.goto(dashboard_url + "/participant/venn-time-series" + query)
    assert response.status == 200
    expect(page.locator(".bokeh-host canvas").first).to_be_visible(timeout=60000)
    page.wait_for_function("""() => Bokeh.documents.some(doc =>
        [...doc.all_models].some(model => model.name === 'step-a'
            && JSON.stringify(model.options) === '["2"]'))""")

    response = page.goto(dashboard_url + "/participant/steps" + query + "&step=1")
    assert response.status == 200
    expect(page.get_by_role("status").filter(has_text="mixed.set")).to_be_visible()
    expect(page.locator("canvas")).to_have_count(0)
    page.locator("#step-select").select_option("2")
    expect(page.locator("canvas").first).to_be_visible(timeout=30000)

    response = page.goto(
        dashboard_url + "/participant/overview?source=dummy&participant=broken"
    )
    assert response.status == 200
    expect(
        page.get_by_text("Has problems: some processing steps are unavailable.")
    ).to_be_visible()
    expect(page.locator("#channel-statistics-table")).to_contain_text(
        "No usable recordings"
    )

    # A participant with healthy data still gets their complete statistics.
    response = page.goto(
        dashboard_url
        + "/participant/overview?source=dummy&participant=1001P_dummy_intake"
    )
    assert response.status == 200
    expect(page.locator("#channel-statistics-table table")).to_be_visible()
    expect(page.locator("#viewer")).not_to_contain_text("Has problems")
