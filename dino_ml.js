(() => {

class QLearningAgent {
    constructor(actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1, qTable = null) {
        this.actions = actions;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.qTable = qTable ?? new Map();
        this.training = true;
    }

    setQTable(qTable){
        this.qTable = qTable; 
    }

    key(state, action) {
        return JSON.stringify({ state, action });
    }

    getQ(state, action) {
        return this.qTable.get(this.key(state, action)) ?? 0;
    }

    chooseAction(state) {
        if (!state) return this.actions[0];

        // If training then explore
        if (Math.random() < this.epsilon && this.training) {
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        }

        // Exploit
        const values = this.actions.map(a => this.getQ(state, a));
        const max = Math.max(...values);

        const best = this.actions.filter(a => this.getQ(state, a) === max);
        return best[Math.floor(Math.random() * best.length)];
    }

    learn(state, action, reward, nextState) {

        // Not training then do not learn
        if(!this.training) return;

        if (!state || !nextState) return;

        const maxNext = Math.max(
            ...this.actions.map(a => this.getQ(nextState, a))
        );

        const current = this.getQ(state, action);

        const updated =
            current + this.alpha * (reward + this.gamma * maxNext - current);

        this.qTable.set(this.key(state, action), updated);
    }

   
}

class DinoEnv {

    constructor(frameSkip = 10) {
        this.frameSkip = frameSkip;
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

    async run_episode(agent){

        let result = this.reset();
        if (!result) return 0;

        let state = result.state;
        let done = false;
        let totalReward = 0;

        while (!done) {

            const action = agent.chooseAction(state);
            
            for (let i = 0; i < this.frameSkip; i++) {
                await new Promise(requestAnimationFrame);
            }
            
            const stepResult = await this.step(action);

            const nextState = stepResult.state;
            const reward = stepResult.reward;
            done = stepResult.done ;

            agent.learn(state, action, reward, nextState);

            state = nextState;
            totalReward += reward;

        }

        return totalReward;
    }
    
    bucket(value, step, maxBucket = Infinity) {
        return Math.min(
            maxBucket,
            Math.floor((value) / step)
        );
    }
}

class RewardChart {
    constructor(options = {}) {
        this.rootId = options.rootId || "main-frame-error";
        this.canvasId = options.canvasId || "rewardChart";
        this.infoId = options.infoId || "rewardInfo";

        this.canvas = null;
        this.ctx = null;
        this.infoElement = null;
    }

    // ─────────────────────────────────────
    // HTML
    // ─────────────────────────────────────

    createHTML() {
        this.createCanvas();
        this.createInfoElement();
    }

    createCanvas() {
        let canvas = document.getElementById(this.canvasId);

        if (!canvas) {
            canvas = document.createElement("canvas");

            canvas.id = this.canvasId;
            canvas.width = 700;
            canvas.height = 400;

            Object.assign(canvas.style, {
                border: "1px solid #ccc",
                marginTop: "16px",
                display: "block"
            });

            const root = document.getElementById(this.rootId);

            root?.appendChild(canvas);
        }

        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");

        return canvas;
    }

    createInfoElement() {
        if (!this.canvas) {
            return null;
        }

        let info = document.getElementById(this.infoId);

        if (!info) {
            info = document.createElement("div");

            info.id = this.infoId;

            Object.assign(info.style, {
                position: "absolute",
                left: `${this.canvas.offsetLeft}px`,
                top: `${this.canvas.offsetTop}px`,
                padding: "6px 10px",
                font: "12px monospace",
                background: "rgba(0,0,0,0.7)",
                color: "#0f0",
                borderRadius: "4px",
                pointerEvents: "none"
            });

            document.body.appendChild(info);
        }

        this.infoElement = info;

        return info;
    }

    // ─────────────────────────────────────
    // DRAW
    // ─────────────────────────────────────

    draw(data = [], episode = 0) {
        if (!this.canvas || !this.ctx || data.length < 2) {
            return;
        }

        const ctx = this.ctx;
        const { width, height } = this.canvas;

        ctx.clearRect(0, 0, width, height);

        const padding = 40;

        const maxReward = Math.max(...data);
        const minReward = Math.min(...data);

        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;

        const xStep = chartWidth / (data.length - 1);

        const normalizeY = value => {
            return (
                height -
                padding -
                ((value - minReward) / (maxReward - minReward || 1)) *
                    chartHeight
            );
        };

        this.drawAxes(ctx, width, height, padding);
        this.drawLine(ctx, data, padding, xStep, normalizeY);
        this.drawPoints(ctx, data, padding, xStep, normalizeY);

        this.drawLabels({
            ctx,
            padding,
            width,
            height,
            episode,
            latest: data[data.length - 1],
            maxReward,
            minReward
        });
    }

    drawAxes(ctx, width, height, padding) {
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
    }

    drawLine(ctx, data, padding, xStep, normalizeY) {
        ctx.strokeStyle = "#0ea5e9";
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(padding, normalizeY(data[0]));

        data.slice(1).forEach((value, index) => {
            const x = padding + (index + 1) * xStep;
            const y = normalizeY(value);

            ctx.lineTo(x, y);
        });

        ctx.stroke();
    }

    drawPoints(ctx, data, padding, xStep, normalizeY) {
        ctx.fillStyle = "#f97316";

        data.forEach((value, index) => {
            const x = padding + index * xStep;
            const y = normalizeY(value);

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawLabels({
        ctx,
        padding,
        width,
        height,
        episode,
        latest,
        maxReward,
        minReward
    }) {
        ctx.fillStyle = "#111";
        ctx.font = "12px monospace";

        const labels = [
            `Canvas: ${width}x${height}`,
            `Episode: ${episode}`,
            `Latest Reward: ${latest.toFixed(2)}`,
            `Max: ${maxReward.toFixed(2)}  Min: ${minReward.toFixed(2)}`
        ];

        labels.forEach((label, index) => {
            ctx.fillText(label, padding, 18 + index * 16);
        });
    }
}

class BrowserControls {

    constructor(options = {}) {
        this.enabled = false;
        this.running = false;

        this.env = options.env;
        this.agent = options.agent;
        this.episodes = options.episodes || 100;

        this.rootId = options.rootId || "main-frame-error";

        this.chart = new RewardChart({
            rootId: this.rootId,
            canvasId: options.canvasId
        });
    }

    // ─────────────────────────────────────
    // STATE
    // ─────────────────────────────────────

    setEnabled(value) {
        this.enabled = Boolean(value);

        this.updateToggleButton();
    }

    isEnabled() {
        return this.enabled;
    }

    updateToggleButton() {
        if (!this.toggleElement) {
            return;
        }

        this.toggleElement.dataset.enabled = String(this.enabled);
        this.toggleElement.textContent = this.enabled ? "ON" : "OFF";
        this.toggleElement.style.background = this.enabled
            ? "#16a34a"
            : "#dc2626";
    }

    // ─────────────────────────────────────
    // HTML
    // ─────────────────────────────────────

    createHTML() {
        this.chart.createHTML();
        this.createControls();
    }

    createControls() {
        const header = document.querySelector("header");

        if (!header) {
            return;
        }

        const panel = this.createPanel();

        panel.appendChild(
            this.createControlRow(
                "Training",
                this.createToggleButton()
            )
        );

        panel.appendChild(
            this.createControlRow(
                "Load QTable",
                this.createLoadButton()
            )
        );

        panel.appendChild(
            this.createControlRow(
                "Start",
                this.createStartButton()
            )
        );

        header.appendChild(panel);
    }

    createPanel() {
        const panel = document.createElement("div");

        Object.assign(panel.style, {
            display: "flex",
            flexDirection: "column",
            gap: "10px",
            marginTop: "12px",
            width: "320px",
            fontFamily: "sans-serif"
        });

        return panel;
    }

    createControlRow(label, element) {
        const row = document.createElement("div");

        Object.assign(row.style, {
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "16px"
        });

        const text = document.createElement("span");

        text.textContent = label;

        Object.assign(text.style, {
            fontSize: "14px",
            fontWeight: "600",
            minWidth: "120px"
        });

        row.appendChild(text);
        row.appendChild(element);

        return row;
    }

    createButton(text, background) {
        const button = document.createElement("button");

        button.textContent = text;

        Object.assign(button.style, {
            background,
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            padding: "8px 16px",
            cursor: "pointer",
            minWidth: "90px",
            fontWeight: "600"
        });

        return button;
    }

    createToggleButton() {
        const button = this.createButton("OFF", "#dc2626");

        button.dataset.enabled = "false";

        button.addEventListener("click", () => {
            this.setEnabled(!this.enabled);
        });

        this.toggleElement = button;

        return button;
    }

    createLoadButton() {
        const button = this.createButton("Load", "#2563eb");

        button.addEventListener("click", () => {
            this.openCSVFilePicker();
        });

        return button;
    }

    createStartButton() {
        const button = this.createButton("Start", "#16a34a");

        button.addEventListener("click", async () => {
            if (typeof this.runGame === "function") {
                await this.runGame();
            }
        });

        return button;
    }

    // ─────────────────────────────────────
    // CHART
    // ─────────────────────────────────────

    drawChart(data = [], episode = 0) {
        this.chart.draw(data, episode);
    }

    // ─────────────────────────────────────
    // CSV EXPORT
    // ─────────────────────────────────────

    exportQTableCSV() {
        
        debugger;

        const rows = [
            [
                "Speed",
                "Time To Collision",
                "Obs Gap",
                "Obs Width",
                "Is Jumping",
                "Is Ducking",
                "Obs Above Ground",
                "Action",
                "Q Value"
            ]
        ];

        for (const [key, value] of this.agent.qTable.entries()) {
            const { state, action } = JSON.parse(key);

            rows.push([
                ...state,
                action,
                value
            ]);
        }

        const csv = rows.map(row => row.join(",")).join("\n");

        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");

        a.href = url;
        a.download = "q_table.csv";
        a.click();

        URL.revokeObjectURL(url);
    }

    // ─────────────────────────────────────
    // CSV IMPORT
    // ─────────────────────────────────────

    openCSVFilePicker() {
        const input = document.createElement("input");

        input.type = "file";
        input.accept = ".csv";

        input.addEventListener("change", async event => {
            const file = event.target.files?.[0];
            if (!file) {
                return;
            }
            this.agent.setQTable(await this.loadQTableFromFile(file));
        });

        input.click();
    }

    async loadQTableFromFile(file) {
        const text = await file.text();

        return this.loadQTableFromCSV(text);
    }

    loadQTableFromCSV(csvText) {
        const lines = csvText.trim().split("\n");

        const qTable = new Map();

        for (let i = 1; i < lines.length; i++) {
            const [
                s,
                t,
                g,
                w,
                j,
                d,
                a,
                action,
                qValue
            ] = lines[i].split(",");

            const state = [
                Number(s),
                Number(t),
                Number(g),
                Number(w),
                j === "true",
                d === "true",
                a === "true"
            ];

            const key = JSON.stringify({
                state,
                action
            });

            qTable.set(key, Number(qValue));
        }

        console.log(`QTable loaded: ${qTable.size}`);

        return qTable;
    }

    async runGame(){
        
        if (this.running) return;
        
        const rewards = [];
        this.running = true;

        for (let ep = 0; ep < this.episodes; ep++) {
            const totalReward = await this.env.run_episode(this.agent);
            rewards.push(totalReward);
            this.drawChart(rewards, ep);
        }

        this.running = false;

    }
}

let browser = null;
async function start(episodes = 300) {

    const env = new DinoEnv(10);
    const agent = new QLearningAgent(["jump", "do_nothing", "duck_on", "duck_off"], 0.1, 0.1, 0.1);
    browser = new BrowserControls({env, agent, episodes});
    browser.createHTML();
    window.browserControls = browser;
}

// expose globally
window.startDinoTraining = start;

})();