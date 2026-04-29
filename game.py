from PIL import Image
from matplotlib.pyplot import gray
from playwright.sync_api import sync_playwright, Playwright


threshold = 120

def handle_dialog(dialog):
    print(f"Dialog message: {dialog.message}")
    dialog.accept()
    
  
def to_grayscale(data):
    out = []
    color_set = set()
    
    for i in range(0, len(data), 4):
        r, g, b, a = data[i:i+4]
        gray = 0.299*r + 0.587*g + 0.114*b
        color_set.add(gray)
        
        clouds = [64, 51, 102]
        
        if a  in clouds:
            # transparent pixel
            out.extend([255, 255, 255, 255])
        else:
            if gray < threshold:
                # obstacle
                out.extend([0, 0, 0, 255])
            else:
                # background
                out.extend([255, 255, 255, 255])
    
    for color in color_set:
        print(f"Unique color: {color}")
    return out

def save_image(width, height, data, path):
    img = Image.frombytes("RGBA", (width, height), bytes(data))
    img.save(path)

def run(playwright: Playwright):
    chromium = playwright.chromium # or "firefox" or "webkit".
    browser = chromium.launch(headless=False, slow_mo=50)
    context = browser.new_context()
    context.set_offline(True)
    page = context.new_page()    
    try:
        page.goto("https://google.com/")
    except Exception as e:
        print(f"Error occurred: {e}")
    page.keyboard.press("Space")
    page.wait_for_timeout(5000)
    raw  = page.evaluate("""
    () => {
        const canvas = document.querySelector('canvas');
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        const data = ctx.getImageData(0, 0, width, height).data;
        return { width, height, data: Array.from(data) };
    }
    """)
    
    gray = to_grayscale(raw["data"])

    save_image(raw["width"], raw["height"], gray, "gray.png")

    print("Grayscale pixels:", len(gray))
    page.wait_for_timeout(200000)
    browser.close()
    
        
with sync_playwright() as playwright:
    run(playwright)