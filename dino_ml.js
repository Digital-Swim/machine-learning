(() => {

class QLearningAgent {
    constructor(actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.actions = actions;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.qTable = new Map();
    }

    key(state, action) {
        return JSON.stringify({ state, action });
    }

    getQ(state, action) {
        return this.qTable.get(this.key(state, action)) ?? 0;
    }

    chooseAction(state) {
        if (!state) return this.actions[0];

        if (Math.random() < this.epsilon) {
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        }

        const values = this.actions.map(a => this.getQ(state, a));
        const max = Math.max(...values);

        const best = this.actions.filter(a => this.getQ(state, a) === max);
        return best[Math.floor(Math.random() * best.length)];
    }

    learn(state, action, reward, nextState) {
        if (!state || !nextState) return;

        const maxNext = Math.max(
            ...this.actions.map(a => this.getQ(nextState, a))
        );

        const current = this.getQ(state, action);

        const updated =
            current + this.alpha * (reward + this.gamma * maxNext - current);

        this.qTable.set(this.key(state, action), updated);
    }

    exportQTableJSON() {
        const data = [];

        for (let [key, value] of this.qTable.entries()) {
            const parsed = JSON.parse(key);
            data.push({
                state: parsed.state,
                action: parsed.action,
                value
            });
        }

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "q_table.json";
        a.click();

        URL.revokeObjectURL(url);
    }

    exportQTableCSV() {
        const rows = [
            ["velocity", "distance", "width", "jumpVelocity", "action", "q_value"]
        ];

        for (let [key, value] of this.qTable.entries()) {
            const { state, action } = JSON.parse(key);
            const [v, d, w, j] = state;

            rows.push([v, d, w, j, action, value]);
        }

        const csv = rows.map(r => r.join(",")).join("\n");

        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "q_table.csv";
        a.click();

        URL.revokeObjectURL(url);
    }
}

class DinoEnv {
    constructor() {
        this.lastPassedObsId = null;
    }

    getState() {
        const r = window.Runner?.instance_;
        if (!r) return null;

        const player = r.tRex;
        const obs = r.horizon.obstacles;
        const done = r.crashed;

        if (done) {
            return { state: null, reward: -100, done: true };
        }

        let obstacle = obs?.[0] ?? null;
        const inAir = obstacle  ? ( Runner.defaultDimensions.HEIGHT - Runner.config.BOTTOM_PAD -(obstacle.yPos + obstacle.typeConfig.height)) : null;
    
        let reward = 0.02;

        if (obstacle) {
            if (obstacle.xPos + obstacle.width + 5 < player.xPos) {
                reward = 5;
            } else if ((player.jumping || player.ducking ) && obstacle.xPos > 150) {
                reward = -0.2;
            } 
        }
        else
            {
                // Unneccessory jumping or ducking 
                if (player.jumping || player.ducking)
                    {
                        reward = -0.1;
                    } 
        }

        // Time to collision
        const timeToCollision = obstacle ? Math.min(Math.round(obstacle.xPos / r.currentSpeed), 30) : null;
        const obsGapBucket = obstacle?.gap == null ? null : Math.min(6, Math.floor(obstacle.gap / 50));
        const velocity = Math.round(r.currentSpeed * 10) / 10;
        const pos = obstacle ? Math.min(obstacle.xPos, 150) : null;
        const obsWidth = obstacle ? obstacle.width : null;

        const playerAction = player.ducking ? 2 :
            ( player.jumpVelocity < 0 ? -1 :
              player.jumpVelocity > 0 ? 1 : 0 );
                
        if(obstacle){

            console.log(
            'time to collsion:', obstacle.xPos / r.currentSpeed,
            'velocity:', r.currentSpeed ,
            'gapBucket:', obstacle.gap ,
            'obs width:', obstacle.width,
            'obs in air:', inAir,
            'reward:', reward
            ); 
        }

        if (reward == 10 || reward == -100){
            console.log(
            'time to collsion:', timeToCollision,
            'gapBucket:', obsGapBucket,
            'width:', obsWidth,
            'playerAction:', playerAction,
            'heightAboveTheGround:', inAir,
            'reward:', reward
            );        
        }

        return {
            state: [timeToCollision, obsGapBucket, obsWidth, playerAction, inAir],
            reward,
            done
        };
    }

    step(action) {
        const r = window.Runner.instance_;
        if(!r.tRex.jumping && !r.tRex.ducking){
            if (action === "jump") {
                r.tRex.startJump(r.currentSpeed);
            }
            if (action === "duck") {
                r.tRex.setDuck(true);
                setTimeout(() => r.tRex.setDuck(false), 20);
            }
        }
        return this.getState();
    }

    reset() {
        const r = window.Runner.instance_;
        r.restart();
        this.lastPassedObsId = null;
        return this.getState();
    }
}

async function train(episodes = 300, delay = 20) {

    let canvas = document.getElementById("rewardChart");

    if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.id = "rewardChart";
        canvas.width = 600;
        canvas.height = 400;

        canvas.style.border = "1px solid #ccc";
        canvas.style.display = "block";
        canvas.style.marginTop = "10px";
        const header = document.getElementById('main-frame-error');
        header.appendChild(canvas);
    }
    
    const ctx = canvas.getContext("2d");
    
    // ─────────────────────────────
    // overlay info element (top-left)
    // ─────────────────────────────
    let info = document.getElementById("rewardInfo");

    if (!info) {
        info = document.createElement("div");
        info.id = "rewardInfo";

        info.style.position = "absolute";
        info.style.left = canvas.offsetLeft + "px";
        info.style.top = canvas.offsetTop + "px";
        info.style.padding = "6px 10px";
        info.style.font = "12px monospace";
        info.style.background = "rgba(0,0,0,0.6)";
        info.style.color = "#0f0";
        info.style.pointerEvents = "none";
        info.style.borderRadius = "4px";

        document.body.appendChild(info);
    }

   // mouse tracking
    let mouse = { x: 0, y: 0 };

    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
    });
    
    const env = new DinoEnv();
    const agent = new QLearningAgent(["jump", "do_nothing", "duck"]);
    const rewards = [];


    for (let ep = 0; ep < episodes; ep++) {
        let result = env.reset();

        if (!result) continue;

        let state = result.state;
        let done = false;
        let totalReward = 0;

        while (!done) {

            // if (!state || state.includes(null)) {
            //     await new Promise(r => setTimeout(r, delay));
            //     continue;
            // }

            const action = agent.chooseAction(state);

            const stepResult = env.step(action);

            const nextState = stepResult?.state;
            const reward = stepResult?.reward ?? 0;
            done = stepResult?.done ?? true;
            
            agent.learn(state, action, reward, nextState);

            state = nextState;
            totalReward += reward;

            await new Promise(r => setTimeout(r, delay));
        }

        rewards.push(totalReward);
        drawChart(ctx, canvas, rewards, ep, mouse);
        console.log(`Episode ${ep} → reward: ${totalReward}`);
        agent.epsilon = Math.max(0.01, agent.epsilon * 0.995);
    }

    window.__agent = agent;

    return "training_done";
}

function drawChart(ctx, canvas, data, episode, mouse) {

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (data.length < 2) return;

    const maxReward = Math.max(...data);
    const minReward = Math.min(...data);

    const padding = 40;

    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    const xStep = chartWidth / (data.length - 1);

    const normalizeY = (value) =>
        height - padding -
        ((value - minReward) / (maxReward - minReward || 1)) * chartHeight;

    // ─────────────────────────────
    // axes
    // ─────────────────────────────
    ctx.strokeStyle = "#333";
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // ─────────────────────────────
    // line
    // ─────────────────────────────
    ctx.strokeStyle = "#00aaff";
    ctx.beginPath();
    ctx.moveTo(padding, normalizeY(data[0]));

    for (let i = 1; i < data.length; i++) {
        const x = padding + i * xStep;
        const y = normalizeY(data[i]);
        ctx.lineTo(x, y);
    }
    ctx.stroke();

    // ─────────────────────────────
    // points
    // ─────────────────────────────
    ctx.fillStyle = "#ff5500";
    for (let i = 0; i < data.length; i++) {
        const x = padding + i * xStep;
        const y = normalizeY(data[i]);

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
    }

    // ─────────────────────────────
    // top-left labels (inside canvas)
    // ─────────────────────────────
    const latest = data[data.length - 1];

    ctx.fillStyle = "#000";
    ctx.font = "12px monospace";

    ctx.fillText(`Canvas: ${width}x${height}`, padding, 15);
    ctx.fillText(`Episode: ${episode}`, padding, 30);
    ctx.fillText(`Latest Reward: ${latest.toFixed(2)}`, padding, 45);
    ctx.fillText(`Max: ${maxReward.toFixed(2)} Min: ${minReward.toFixed(2)}`, padding, 60);

    // ─────────────────────────────
    // mouse X/Y readout (data space)
    // ─────────────────────────────
    if (mouse) {
        const i = Math.floor((mouse.x - padding) / xStep);

        if (i >= 0 && i < data.length) {
            const xVal = i;
            const yVal = data[i];

            ctx.fillStyle = "#ff00ff";
            ctx.fillText(`X: ${xVal} Y: ${yVal.toFixed(2)}`, mouse.x + 10, mouse.y);
        }
    }
}
// expose globally
window.startDinoTraining = train;

})();