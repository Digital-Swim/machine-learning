from environments.world import World
from playwright.sync_api import sync_playwright, Playwright

class DinoGame:
    
    def __init__(self, dino_world_url="https://chromedino.com/"):
        self.dino_world_url = dino_world_url
        self.playwright = None
        self.browser = None
        self.page = None
        
   
    def reset(self):
        if self.page is not None:
            self.page.keyboard.press("Space")
            self.page.wait_for_timeout(1000)  # Wait for the game to start
            return self.get_env_state()
        
    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page()
        self.page.goto(self.dino_world_url)
        self.page.wait_for_selector("canvas")
        return self
        
    def close(self):
        if self.page:
            self.page.context.browser.close()
            
        if self.playwright:
            self.playwright.stop()
    
    
    # Returns state with speed, distance to nearest obstacle, and whether it's a terminal state (collision) and reward
    def get_env_state(self):
        state = self.page.evaluate("""
        () => {
            if (!window.Runner || !window.Runner.instance_) return null;
            const r = window.Runner.instance_;
            const obs = r.horizon.obstacles;
            return {
                speed: r.currentSpeed,
                jumping: r.tRex.jumping,
                x: obs.length > 0 ? obs[0].xPos : null,
                y: obs.length > 0 ? obs[0].yPos : null,
                width: obs.length > 0 ? obs[0].width : null,
                playerY: r.tRex.yPos,
                playerX: r.tRex.xPos,
                obstacles: obs.map(o => ({x: o.xPos, y: o.yPos, width: o.width, height: o.height})),
                terminal:  r.crashed  
            };  
        }
        """)

        if state is not None:
            if state["terminal"]:
                reward = -100
            else:
                reward = 0.1
            
            # Return state, reward, and terminal flag
            return (round(state['speed']), round(state['x'] // 50) if state['x'] is not None else None), reward, state["terminal"]
        
        return (0,0,0), 0.1, False

    
    def perform_action(self, action):
        if action == "jump":
            self.page.keyboard.press("ArrowUp")
        elif action == "duck":
            self.page.keyboard.down("ArrowDown")
            self.page.wait_for_timeout(500)  # Duck for 500ms
            self.page.keyboard.up("ArrowDown")
        elif action == "do_nothing":
            pass  # No action taken

                                 
                                 
class DinoWorld(World):
    
    def __init__(self, dino_world_url="https://chromedino.com/"):
        self.dino_world_url = dino_world_url
        self.game = None
    
    def reset(self):
        if self.game is None:
            self.game = DinoGame(self.dino_world_url).start()
            return self.game.get_env_state()
        else:
            return self.game.reset()
            
    def step(self, state, action):
        self.game.perform_action(action=action)
        self.game.page.wait_for_timeout(0.01)
        return self.game.get_env_state()

        
    def render(self, q_table=None, state=None, show_heatmap=False):
        # Rendering is handled by the game itself
        pass

    def get_actions(self):
        return ["jump", "duck", "do_nothing"]

    def is_terminal(self, state):
        # Terminal state is when the dinosaur collides with an obstacle
        return state['terminal']  # We will determine this from the game's internal state
    
        for episode in range(episodes):
            state, _, _ = self.game.get_env_state()
            total_reward = 0
            step_count = 0

            while not self.is_terminal(state):
                action = agent.choose_action(state)
                self.game.perform_action(action)
                next_state, _, _ = self.game.get_env_state()

                reward = -1 if self.is_terminal(next_state) else 1
                agent.learn(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                step_count += 1

                if visualize:
                    time.sleep(delay)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {step_count}")