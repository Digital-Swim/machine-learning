(() => {

class QLearningAgent {
    constructor(actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1, qTable = null) {
        this.actions = actions;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.qTable = qTable ?? new Map();
        this.training = false;
    }

    setQTable(qTable){
        this.qTable = qTable; 
    }

    key(state, action) {
        const s = this.normalizeState(state);
        return `${s.join(",")}|${action}`;
    }

    decodeKey(key) {
        const [stateStr, action] = key.split("|");
        const state = stateStr.split(",").map(Number);
        return { state, action };
    }

    normalizeState(state) {
        return state.map(v => {
            if (v === null || v === undefined) return 0;
            if (typeof v === "boolean") return v ? 1 : 0;
            return v;
        });
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
        //r.currentSpeed = 10;
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

            console.log(state, action, reward, nextState);
            agent.learn(state, action, reward, nextState);

            state = nextState;
            totalReward += reward;

        }

        this.updateEpsilon(agent);

        console.log(agent.epsilon);

        return totalReward;
    }
    
    updateEpsilon(agent) {
        const min = 0.1;
        agent.epsilon = Math.max(min, agent.epsilon * 0.997);
    }

    bucket(value, step, maxBucket = Infinity) {
        return Math.min(
            maxBucket,
            Math.floor((value) / step)
        );
    }
}


class BrowserControls {

    constructor(options = {}) {
        this.enabled = false;
        this.running = false;

        this.env = options.env;
        this.agent = options.agent;
        this.episodes = options.episodes || 100;

        this.dashboard = new LiveDashboard({
            rootId:  "main-frame-error",
            threshold: 0.001
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

    updateCharts(reward, newQTable) {
        this.dashboard.update({
             reward,
            newQ:this.agent.qTable
        })
    }

    // ─────────────────────────────────────
    // CSV EXPORT
    // ─────────────────────────────────────

    exportQTableCSV(filename) {

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

            const { state, action} = this.agent.decodeKey(key);

            rows.push([
                ...state,
                action,
                value
            ]);
        }

        const csv = rows
            .map(row => row.join(","))
            .join("\n");

        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = filename ?? "q_table.csv";
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
            await this.loadQTableFromFile(file)
        });

        input.click();
    }

    async loadQTableFromFile(file) {
        const text = await file.text();

        const lines = text.trim().split("\n");

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
                Number(j),
                Number(d),
                Number(a)
            ];

            const key = this.agent.key(state, action)

            qTable.set(key, Number(qValue));
        }

        console.log(`QTable loaded: ${qTable.size}`);

        this.agent.qTable = qTable;

    }



    async runGame(episodes){
        
        if (this.running) return;
        
        this.running = true;

        for (let ep = 0; ep < episodes ?? this.episodes; ep++) {
            
            const totalReward = await this.env.run_episode(this.agent);
            const status = this.dashboard.update({
                    reward:totalReward,
                    newQ:this.agent.qTable
                });

                console.log(status);

            if (status.converged) {
                console.log("Training stabilized (ΔQ < threshold)");
            }
        }

        this.running = false;

    }
}

class BaseChart {
    constructor(options = {}) {
        this.rootId = options.rootId || "main-frame-error";
        this.canvasId = options.canvasId;
        this.width = options.width || 700;
        this.height = options.height || 400;

        this.canvas = null;
        this.ctx = null;
    }

    createCanvas() {
        let canvas = document.getElementById(this.canvasId);

        if (!canvas) {
            canvas = document.createElement("canvas");
            canvas.id = this.canvasId;
            canvas.width = this.width;
            canvas.height = this.height;

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

    clear() {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.width, this.height);
    }

    drawAxes(padding = 40) {
        const ctx = this.ctx;

        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, this.height - padding);
        ctx.lineTo(this.width - padding, this.height - padding);
        ctx.stroke();
    }

    normalizeY(value, min, max, chartHeight, padding) {
        return (
            this.height -
            padding -
            ((value - min) / (max - min || 1)) * chartHeight
        );
    }
}

class RewardChart extends BaseChart {
    constructor(options = {}) {
        super({ ...options, canvasId: options.canvasId || "rewardChart" });

        this.createCanvas();
    }

    draw(data = [], episode = 0) {
        if (!this.ctx || data.length < 2) return;

        const ctx = this.ctx;
        const padding = 40;

        const maxReward = Math.max(...data);
        const minReward = Math.min(...data);

        const chartWidth = this.width - padding * 2;
        const chartHeight = this.height - padding * 2;

        const xStep = chartWidth / (data.length - 1);

        this.clear();
        this.drawAxes(padding);

        const normalizeY = (v) =>
            this.normalizeY(v, minReward, maxReward, chartHeight, padding);

        this.drawLine(data, padding, xStep, normalizeY);
        this.drawPoints(data, padding, xStep, normalizeY);
        this.drawLabels(data, episode, minReward, maxReward, padding);
    }

    drawLine(data, padding, xStep, normalizeY) {
        const ctx = this.ctx;

        ctx.strokeStyle = "#0ea5e9";
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(padding, normalizeY(data[0]));

        data.forEach((v, i) => {
            const x = padding + i * xStep;
            const y = normalizeY(v);
            ctx.lineTo(x, y);
        });

        ctx.stroke();
    }

    drawPoints(data, padding, xStep, normalizeY) {
        const ctx = this.ctx;

        ctx.fillStyle = "#f97316";

        data.forEach((v, i) => {
            const x = padding + i * xStep;
            const y = normalizeY(v);

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawLabels(data, episode, min, max, padding) {
        const ctx = this.ctx;

        ctx.fillStyle = "#111";
        ctx.font = "12px monospace";

        const latest = data[data.length - 1];

        const labels = [
            `Episode: ${episode}`,
            `Latest Reward: ${latest.toFixed(2)}`,
            `Max: ${max.toFixed(2)} Min: ${min.toFixed(2)}`
        ];

        labels.forEach((t, i) => {
            ctx.fillText(t, padding, 18 + i * 16);
        });
    }
}

class QConvergenceChart extends BaseChart {
    constructor(options = {}) {
        super({ ...options, canvasId: options.canvasId || "qConvChart" });

        this.threshold = options.threshold || 0.001;

        this.labels = [];
        this.deltas = [];

        this.createCanvas();
    }

    computeMaxDelta(oldQ, newQ) {
        let maxDelta = 0;
        for (const [key, newVal] of newQ.entries()) {
            const oldVal = oldQ.get(key) ?? 0;
            const diff = Math.abs(newVal - oldVal);
            if (diff > maxDelta) {
                maxDelta = diff;
            }
        }
        return maxDelta;
    }

    update(episode, oldQ, newQ) {
        const delta = this.computeMaxDelta(oldQ, newQ);

        this.labels.push(episode);
        this.deltas.push(delta);

        this.render();

        return {
            delta,
            converged: delta < this.threshold
        };
    }

    render() {
        if (!this.ctx || this.deltas.length < 2) return;

        const ctx = this.ctx;
        const padding = 40;

        const max = Math.max(...this.deltas);
        const min = Math.min(...this.deltas);

        const chartWidth = this.width - padding * 2;
        const chartHeight = this.height - padding * 2;

        const xStep = chartWidth / (this.deltas.length - 1);

        this.clear();
        this.drawAxes(padding);

        const normalizeY = (v) =>
            this.normalizeY(v, min, max, chartHeight, padding);

        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(padding, normalizeY(this.deltas[0]));

        this.deltas.forEach((v, i) => {
            const x = padding + i * xStep;
            const y = normalizeY(v);
            ctx.lineTo(x, y);
        });

        ctx.stroke();

        ctx.fillStyle = "#111";
        ctx.font = "12px monospace";

        ctx.fillText(
            `Delta Q latest: ${this.deltas.at(-1).toFixed(6)}`,
            padding,
            20
        );
    }
}

class LiveDashboard {
    constructor(options = {}) {
        this.rootId = options.rootId || "main-frame-error";

        this.rewardChart = new RewardChart({
            rootId: this.rootId,
            canvasId: "rewardChart"
        });

        this.qChart = new QConvergenceChart({
            rootId: this.rootId,
            canvasId: "qConvChart",
            threshold: options.threshold || 0.001
        });

        this.episode = 0;

        this.rewards = [];
        this.oldQ = null;

        this.init();
    }

    init() {
        const root = document.getElementById(this.rootId);

        const title = document.createElement("h3");
        title.innerText = "RL Live Dashboard";
        root.appendChild(title);

        this.rewardChart.createCanvas();
        this.qChart.createCanvas();
    }

    /**
     * Call this after every episode
     */
    update({ reward, newQ }) {
        this.episode += 1;

        // ─────────────────────────────
        // 1. reward tracking
        // ─────────────────────────────
        this.rewards.push(reward);
        this.rewardChart.draw(this.rewards, this.episode);

        // ─────────────────────────────
        // 2. Q convergence tracking
        // ─────────────────────────────
        let result = null;

        if (this.oldQ) {
            result = this.qChart.update(
                this.episode,
                this.oldQ,
                newQ
            );
        }

        this.oldQ = structuredClone(newQ);

        // ─────────────────────────────
        // 3. return diagnostic state
        // ─────────────────────────────
        return {
            episode: this.episode,
            reward,
            avgReward: this.avgReward(),
            qDelta: result?.delta ?? null,
            converged: result?.converged ?? false
        };
    }

    avgReward(window = 50) {
        const slice = this.rewards.slice(-window);

        if (slice.length === 0) return 0;

        return slice.reduce((a, b) => a + b, 0) / slice.length;
    }
}

async function setupTrainingEnv(episodes = 300) {
    const env = new DinoEnv(10);
    const agent = new QLearningAgent(["jump", "do_nothing", "duck_on", "duck_off"], 0.1, 0.1, 0.5);
    browser = new BrowserControls({env, agent, episodes});
    browser.createHTML();
    window.browserControls = browser;
}

// expose globally
window.setupTrainingEnv = setupTrainingEnv;

})();