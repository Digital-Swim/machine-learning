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
            return { state: null, reward: -20, done: true };
        }

        let obstacle = obs?.[0] ?? null;
        const obsAboveGround = obstacle  ? ( Runner.defaultDimensions.HEIGHT - Runner.config.BOTTOM_PAD -(obstacle.yPos + obstacle.typeConfig.height)) : null;
        const timeToCollision = obstacle ? this.bucket(obstacle.xPos / r.currentSpeed, 1, 30) : 999;

        let reward = 0.02;

        if (obstacle) {
            if (obstacle.xPos + obstacle.width + 5 < player.xPos) {
                reward = 5;
            } else if ((player.jumping || player.ducking ) && timeToCollision >= 30) {
                reward = -0.2;
            } else if(player.ducking && timeToCollision < 2){
                reward = -0.2;
            }
        }
        else {
            // Unneccessory jumping or ducking 
            if (player.jumping || player.ducking)
                {
                    reward = -0.1;
                } 
        }

        
        const speed = this.bucket(r.currentSpeed * 10, 5); // Math.round(r.currentSpeed * 10);
        const distance = obstacle ?  this.bucket(obstacle.xPos, 10, 15) : null;
        const obsGap = obstacle ?  this.bucket(obstacle.gap, 50, 6) : null;   // Math.min(6, Math.floor(obstacle.gap / 50));
        const obsWidth = obstacle ? this.bucket(obstacle.width, 10) : null;
        const isJumping = player.jumping;
        const isDucking = player.ducking;
        
        

        return {
            state: [speed, timeToCollision, obsGap, obsWidth, isJumping, isDucking, obsAboveGround],
            reward,
            done
        };
    }

    bucket(value, step, maxBucket = Infinity) {
        return Math.min(
            maxBucket,
            Math.floor((value) / step)
        );
    }

step(action) {
    const KEY = {
        jump: 32,
        duck: 40
    };

    function fire(type, keyCode) {
        const e = new KeyboardEvent(type, {
            bubbles: true,
            cancelable: true
            });

            Object.defineProperty(e, 'keyCode', { value: keyCode });
            Object.defineProperty(e, 'which', { value: keyCode });

            document.dispatchEvent(e);
        }

        console.log(action)

        switch (action) {
            case "jump":
                fire("keydown", KEY.jump);
                fire("keyup", KEY.jump);
                break;

            case "duck_on":
                fire("keydown", KEY.duck);
                break;
            
            case "duck_off":
                fire("keyup", KEY.duck);
                break;
            
                case "do_nothing":
            default:
                break;
        }

        return this.getState();
    }

    reset() {
        const r = window.Runner.instance_;
        r.restart();
        return this.getState();
    }
}

async function train(episodes = 300, delay = 20) {

    create_chart();

    const env = new DinoEnv();
    const agent = new QLearningAgent(["jump", "do_nothing", "duck_on", "duck_off"]);
    const rewards = [];
    let frameSkip = 10;

    for (let ep = 0; ep < episodes; ep++) {
        
        let result = env.reset();
        if (!result) continue;

        let state = result.state;
        let done = false;
        let totalReward = 0;

        while (!done) {

            const action = agent.chooseAction(state);
            
            for (let i = 0; i < frameSkip; i++) {
                await new Promise(requestAnimationFrame);
            }
            
            const stepResult = await env.step(action);

            const nextState = stepResult.state;
            const reward = stepResult.reward;
            done = stepResult.done ;
            
            console.log(state, action, reward, nextState)
            agent.learn(state, action, reward, nextState);

            state = nextState;
            totalReward += reward;

        }

        rewards.push(totalReward);
        drawChart(rewards, ep);
        //agent.epsilon = Math.max(0.01, agent.epsilon * 0.995);
    }

    window.__agent = agent;

    return "training_done";
}

function create_chart(){

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
}

function drawChart(data, episode) {

    const canvas = document.getElementById("rewardChart");
    const ctx = canvas.getContext("2d");
   
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


}
// expose globally
window.startDinoTraining = train;

})();