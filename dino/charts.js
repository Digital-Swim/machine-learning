
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