import time
from playwright.sync_api import sync_playwright, Playwright
from PIL import Image
import numpy as np
from scipy.ndimage import generate_binary_structure, label
import numpy as np

from environments.dino import DinoGame, DinoWorld

def save_image(width, height, data, path):
    img = Image.frombytes("RGBA", (width, height), bytes(data))
    img.save(path)


# using the canvas API to get the frame data as RGBA values
def get_frame(page):
    raw = page.evaluate("""
    () => {
        const canvas = document.querySelector('canvas');
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        const data = ctx.getImageData(0, 0, width, height).data;
        return { width, height, data: Array.from(data) };
        }
    """)
    return raw

# using the game's internal state to get the speed and position of the nearest obstacle
def get_game_state_info(page):
    return page.evaluate("""
    () => {
        
        if (!window.Runner || !window.Runner.instance_) return null;

        const r = window.Runner.instance_;
        const obs = r.horizon.obstacles;
        return {
            speed: r.currentSpeed,
            jumping: r.tRex.jumping,
            x: obs.length ? obs[0].xPos : null,
            y: obs.length ? obs[0].yPos : null,
            width: obs.length ? obs[0].width : null
        };
    }
    """)


def get_frame_info(raw):
    raw = np.array(raw["data"], dtype=np.uint8).reshape((raw["height"], raw["width"], 4))
    
    gray = (
        0.299 * raw[:, :, 0] +
        0.587 * raw[:, :, 1] +
        0.114 * raw[:, :, 2]
    )

    gray = gray.astype(int)
    mask = (gray == 255).astype(np.uint8)

    structure = generate_binary_structure(2,2)  # 8-connectivity
    labeled, n = label(mask, structure=structure)

    # 8-connectivity
    objects = []

    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < 100:
            continue
        
        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        
        objects.append({
            "size": len(xs),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        })

    # Get second nrearest object on bases of x_min, as first is the player itself
    if len(objects) > 1:
        nearest = sorted(objects, key=lambda o: o["x_min"])[1]
    else:        
        nearest = None
      
    return objects, nearest

def run(playwright: Playwright):
    
    browser = playwright.chromium.launch(headless=False, slow_mo=0)
    context = browser.new_context()
    context.set_offline(False)
    page = context.new_page()

    try:
        page.goto("https://chromedino.com/")
    except Exception as e:
        print(f"Error: {e}")

    page.wait_for_timeout(2000)

    page.keyboard.press("Space")
    
    time_between_frames = 0.01 # 10ms
    previous_position = 0
    
    set2 = set()
    frame = 0
    while True:
        try:
            
            
            frame += 1
            state = get_game_state_info(page)

            set2.add(round(state["speed"]))
            
            if frame % 100 == 0:
                for s in sorted(set2):
                    print(f"Speed: {s}")
                frame = 0
                
            
            page.wait_for_timeout(int(time_between_frames * 1000))
            
            #_ , nearest = get_frame_info(raw)
                       
            #print(f"Frame {frame} - Nearest object: {nearest['x_min'] if nearest else 'None'}")

            # Calculate speed based on nearest object position change
            #if nearest and previous_position is not None:
            #    current_position = nearest["x_min"]
            #    print(f"Frame {frame} - Current position: {current_position}, Last position: {previous_position}")
            #    speed = (current_position - previous_position) / time_between_frames
            #    print(f"Frame {frame} - Speed: {speed:.2f} pixels/sec - Nearest object at x_min: {current_position}")
            #    previous_position = current_position
            
            #if nearest:
            # Print speed and start position for the nearest object
            #    last_frame_position = nearest["x_min"]
        
        except Exception as e:
            print(f"Frame error: {e}")
            break

    browser.close()


def transform_image_to_mask(raw):
    data = np.array(raw)  # shape: (H, W, 4)
    gray = (
        0.299 * data[:, :, 0] +
        0.587 * data[:, :, 1] +
        0.114 * data[:, :, 2]
    )

    gray = gray.astype(int)
    mask = (gray == 255).astype(np.uint8)
    return mask

def get_frame_features(mask):

    structure = generate_binary_structure(2,2)  # 8-connectivity
    labeled, n = label(mask, structure=structure)

      # 8-connectivity
    objects = []

    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < 100:
            continue
        
        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        
        objects.append({
            "size": len(xs),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        })

    # Get second nrearest object on bases of x_min, as first is the player itself
    if len(objects) > 1:
        nearest = sorted(objects, key=lambda o: o["x_min"])[1]
    else:        
        nearest = None
      
    return objects, nearest

#with sync_playwright() as playwright:
#    run(playwright)


with DinoGame() as game:
    while True:
        state = game.get_state()
        #if state is not None:
        #    if state["distance"] is not None and state["distance"] < 2:
        #        game.perform_action("jump")
        #print(state)
        time.sleep(1)        
    