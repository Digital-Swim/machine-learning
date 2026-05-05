from environments.world import World
from playwright.sync_api import sync_playwright, Playwright
import time
import matplotlib.pyplot as plt

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
         # inject RL loop
        self.page.evaluate("""
        () => {
            if (window.__rlLoop) return;
            window.__rlLoop = true;

            const r = window.Runner.instance_;

            const computeState = () => {
                const player = r.tRex;
                const obstacles = r.horizon.obstacles;

                const velocity = Math.round(r.currentSpeed * 10);

                let distance = null;
                let timeToCollision = null;

                if (obstacles.length > 0) {
                    const o = obstacles[0];

                    if (o.xPos + o.width >= player.xPos) {
                        distance = o.xPos - (player.xPos + player.config.WIDTH);

                        if (velocity > 0) {
                            timeToCollision = distance / velocity;
                        }
                    }
                }

                window.__rlState = {
                    velocity,
                    distance,
                    timeToCollision,
                    crashed: r.crashed,
                    isJumping: player.jumping ? 1 : 0
                };

                requestAnimationFrame(computeState);
            };

            computeState();
        }
        """)
        
        return self
        
    def close(self):
        if self.page:
            self.page.context.browser.close()
            
        if self.playwright:
            self.playwright.stop()
    
    def draw_overlay(self):
        self.page.evaluate("""
        () => {
            if (!window.Runner || !window.Runner.instance_) return;

            if (window.__debugOverlayInjected) return;
            window.__debugOverlayInjected = true;

            const r = window.Runner.instance_;
            const gameCanvas = document.querySelector('canvas.runner-canvas');
            if (!gameCanvas) return;

            const canvas = document.createElement('canvas');
            canvas.id = 'debug-overlay';
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            canvas.style.position = 'fixed';
            canvas.style.left = '0';
            canvas.style.top = '0';
            canvas.style.pointerEvents = 'none';
            canvas.style.zIndex = '9999';
            document.body.appendChild(canvas);

            const ctx = canvas.getContext('2d');

            function draw() {
                debugger;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                const rect = gameCanvas.getBoundingClientRect();
                const obs = r.horizon.obstacles;
                const player = r.tRex;
                scaleX = rect.width / gameCanvas.width;
                scaleY = rect.height / gameCanvas.height;
                
               const px = rect.left + (player.xPos + 1) * scaleX;
                const py = rect.top + (player.yPos + 1) * scaleY;

                const pw = (player.config.WIDTH - 2) * scaleX;
                const ph = (player.config.HEIGHT - 2) * scaleY;

                ctx.strokeStyle = 'green';
                ctx.strokeRect(px, py, pw, ph);
                
                // Obstacles (FIXED)
                ctx.strokeStyle = 'red';
                obs.forEach(o => {
                const ox = rect.left + (o.xPos + 1) * scaleX;
                const oy = rect.top + (o.yPos + 1) * scaleY;

                const ow = (o.typeConfig.width * o.size - 2) * scaleX;
                const oh = (o.typeConfig.height - 2) * scaleY;

                ctx.strokeStyle = 'red';
                ctx.strokeRect(ox, oy, ow, oh);
            });
                        

                // Debug text
                ctx.fillStyle = 'green';
                ctx.font = '12px monospace';
                ctx.fillText(`Speed: ${r.currentSpeed}`, 10, 20);
                ctx.fillText(`Obstacles: ${obs.length}`, 10, 40);
                ctx.fillText(`Jumping: ${player.jumping}`, 10, 60);
                ctx.fillText(`Crashed: ${r.crashed}`, 10, 80);

                // State info for debugging
                v = Math.round(r.currentSpeed * 10); // 10 pixels per second per speed unit, rounded to nearest integer
                distance = obs.length > 0 ? obs[0].xPos - (player.xPos + player.config.WIDTH) : null;
                distanceBucket = distance !== null ? Math.floor(distance / 10) * 10 : null; // Bucket distance into 10 pixel intervals
                timeToCollision = distance !== null && v > 0 ? Math.floor(distance / v * 10) : null;  // Convert to milliseconds
                
                
                // limit distance bucket to 100 for better state representation
                if (distanceBucket !== null) {
                    distanceBucket = Math.min(distanceBucket, 100);
                }
                ctx.fillText(`velocity: ${v}`, 10, 100);
                ctx.fillText(`distance to next obstacle: ${distance !== null ? distance : 'N/A'}`, 10, 120);
                ctx.fillText(`distance bucket: ${distanceBucket !== null ? distanceBucket : 'N/A'}`, 10, 140);

                // Check if we passed an obstacle
                if (obs.length > 0 && obs[0].xPos + obs[0].width < player.xPos) {
                    ctx.fillText(`Passed an obstacle!`, 10, 160);
                }
                else {
                    ctx.fillText(`Not passed an obstacle yet.`, 10, 160);
                }
                ctx.fillText(`time to collision: ${timeToCollision !== null ? timeToCollision : 'N/A'}`, 10, 180);
                
                
                
                requestAnimationFrame(draw);
            }

            draw();
        }
        """)
        
    def get_env_state1(self):
        stateInfo = self.page.evaluate("""
        () => {
            if (!window.Runner || !window.Runner.instance_) return null;
            const r = window.Runner.instance_;
            const obs = r.horizon.obstacles;
            return {
                speed: r.currentSpeed,
                jumping: r.tRex.jumping,
                xPos: obs.length > 0 ? obs[0].xPos : null,
                yPos: obs.length > 0 ? obs[0].yPos : null,
                width: obs.length > 0 ? obs[0].width : null,
                playerY: r.tRex.yPos,
                playerX: r.tRex.xPos,
                playerWidth: r.tRex.width,
                playerHeight: r.tRex.height,
                obstacles: obs.map(o => ({xPos: o.xPos, yPos: o.yPos, width: o.width, height: o.height})),
                terminal:  r.crashed  
            };  
        }
        """)
                
        if stateInfo is not None:
            # Get time to collision for the obstacle in front of the player, if any ignoring those that have already been passed
            obstacle = None
            for obs in stateInfo['obstacles']:
                if obs['xPos'] > stateInfo['playerX'] + 10:  # Only consider obstacles that are ahead of the player
                    obstacle = obs
                    break
            time_to_collision = (obstacle['xPos'] - (stateInfo['playerX'] + 10 )) / stateInfo['speed'] if obstacle is not None and stateInfo['speed'] > 0 else float('inf')

            # bucket time to collision into 3 second intervals for better state representation and round to nearest bucket, ignore buckets above 5
            if time_to_collision == float('inf'):
                time_to_collision = 15  # Cap the time to collision at 15 seconds for state representation
            time_to_collision = round(time_to_collision / 3) * 3
            time_to_collision = min(time_to_collision, 15)
            
            # check if we passed an obstacle and give a reward for that, otherwise give a small positive reward for surviving each step
            reward = 0.1  # Default small positive reward for surviving each step
            if obstacle is not None and obstacle['xPos'] < stateInfo['playerX'] - 10:  # Passed an obstacle
                reward = 10
            # check for terminal state and give a negative reward for that
            if stateInfo["terminal"]:
                reward = -100
            
            return (round(stateInfo['speed']), time_to_collision), reward, stateInfo["terminal"]
        
        return None, 0, False  # Default state, reward, and terminal flag if game state is not available
            
    # Returns state with speed, distance to nearest obstacle, and whether it's a terminal state (collision) and reward

    def get_env_state(self, previous_reward=None):
        stateInfo = self.page.evaluate("""
        () => {
            if (!window.Runner || !window.Runner.instance_) return null;
            const r = window.Runner.instance_;
            const obs = r.horizon.obstacles;
            return {
                speed: r.currentSpeed,
                jumping: r.tRex.jumping,
                xPos: obs.length > 0 ? obs[0].xPos : null,
                yPos: obs.length > 0 ? obs[0].yPos : null,
                width: obs.length > 0 ? obs[0].width : null,
                playerY: r.tRex.yPos,
                playerX: r.tRex.xPos,
                playerWidth: r.tRex.config.WIDTH,
                obstacles: obs.map(o => ({xPos: o.xPos, yPos: o.yPos, width: o.width, height: o.height})),
                terminal:  r.crashed  
            };  
        }
        """)
        
        done = stateInfo["terminal"]
        isJumping = stateInfo["jumping"]
        
        if done:
            return None, -100, done 
        
        #if previous_reward == 10:
        #    for obs in stateInfo['obstacles']:
        #        if obs['xPos'] > stateInfo['playerX'] + stateInfo["playerWidth"]:  # Only consider obstacles that are ahead of the player
        #            obstacle = obs
        #            break
        #else:
        
        obstacle = stateInfo["obstacles"][0] if len(stateInfo["obstacles"]) > 0 else None
        
        reward = 0.1

        if obstacle is not None:
            if obstacle["xPos"] + obstacle["width"] + 5 < stateInfo["playerX"]:
                reward = 10
            elif isJumping and obstacle["xPos"] > 150:
                reward = -0.1
            elif not isJumping and obstacle["xPos"] < 10:
                reward = -0.1
                        
        # state details 
        v = round(stateInfo["speed"], 1)
        pos = round(obstacle["xPos"] / 10) if obstacle is not None else None
        w = obstacle["width"] if obstacle is not None else None
        
        return (v, pos, w, isJumping), reward, done 

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
            
    def step(self, state, action, reward=None):
        self.game.perform_action(action=action)
        #self.game.page.wait_for_timeout(0.01)
        time.sleep(0.001)
        return self.game.get_env_state(previous_reward=reward)


    def train(self, agent, episodes=1000, visualize=False, delay=0.1, show_heatmap=False):
        episode_rewards = []
        
        if visualize:
            plt.ion()
            
        for _ in range(episodes):
            state = self.reset()
            print("reseting")
            done = False
            total_reward = 0
            preveReward = 0
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = self.step(state, action, reward=preveReward)
                total_reward += reward
                
                if reward == 10 or reward == -100:
                    print(state, action, reward, next_state)
                
                agent.learn(state, action, reward, next_state)
                state = next_state
                preveReward = reward

                if visualize:
                    # LEFT: grid
                    #self.render(q_table=agent.q_table, state=state, show_heatmap=show_heatmap)

                    # RIGHT: reward graph
                    self.plot_rewards(episode_rewards)

                    plt.pause(delay)
    
            time.sleep(2)
            episode_rewards.append(total_reward)
        
        if visualize:
            plt.ioff()
            plt.show()

    def render(self, q_table=None, state=None, show_heatmap=False):
        # Rendering is handled by the game itself
        pass

    def get_actions(self):
        return ["jump", "do_nothing"]


    
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