from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto("https://chromedino.com/")
    page.wait_for_selector("canvas")

    with open("dino_ml.js", "r") as f:
        page.evaluate(f.read())

    # start training
    page.evaluate("startDinoTraining(1000)")

    # optional: wait / inspect 

    input("Press Enter to export Q-table...")

    with page.expect_download() as download_info:
        page.evaluate("window.__agent.exportQTableCSV()")

    download = download_info.value

    download.save_as("q_table.csv")
    
    input("Press Enter to close...")
    browser.close()