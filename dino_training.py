from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto("https://chromedino.com/")
    page.wait_for_selector("canvas")

    with open("dino_ml.js", "r") as f:
        page.evaluate(f.read())

    # start training
    page.evaluate("setupTrainingEnv(1000)")

    # optional: wait / inspect 
    input("Press Enter to export Q-table...")

    episode = 0
    totalEpisode = 10000
    step = 100

    while episode <= totalEpisode:
        page.evaluate(
            f"window.browserControls.runGame({step})"
        )

        episode += step

        with page.expect_download() as download_info:
            page.evaluate(
                f'window.browserControls.exportQTableCSV("1qTable_{episode}.csv")'
            )

        download = download_info.value
        download.save_as(f"1qTable_{episode}.csv")
    
    input("Press Enter to close...")
    browser.close()