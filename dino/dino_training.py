from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto("https://chromedino.com/")
    page.wait_for_selector("canvas")

    # Inject D3
    page.add_script_tag(
        url="https://d3js.org/d3.v7.min.js"
    )
    
    
    input("Press Enter to start training...")
    
    # Inject RLVisualizer.js
    page.add_script_tag(
        path="./dino/rl_visualizer.js"
    )
    
    page.add_script_tag(
        path="./dino/charts.js"
    )
    
    page.add_script_tag(
        path="./dino/browser_controls.js"
    )
    
    
    page.add_script_tag(
        path="./dino/dino_ml.js"
    )
    

    input("Press Enter to start training...")

    # start training
    page.evaluate("setupTrainingEnv(1000)")

    # optional: wait / inspect 
    input("Press Enter to run training episodes...")

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