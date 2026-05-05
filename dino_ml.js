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

        obs.forEach(o => {
            if (o.__id === undefined) {
                o.__id = Date.now() + Math.random();
            }
        });

        const done = r.crashed;

        if (done) {
            return { state: null, reward: -100, done: true };
        }

        let obstacle = null;
        for (let o of obs) {
            if (o.__id !== this.lastPassedObsId) {
                obstacle = o;
                break;
            }
        }

        let reward = 0.1;

        if (obstacle) {
            if (obstacle.xPos + obstacle.width + 5 < player.xPos) {
                reward = 10;
                this.lastPassedObsId = obstacle.__id;
            } else if (player.jumping && obstacle.xPos > 150) {
                reward = -0.1;
            } else if (!player.jumping && obstacle.xPos < 10) {
                reward = -0.1;
            }
        }else{
            if (player.jumping){
                reward = -0.1;
            } 
        }

        const v = Math.round(r.currentSpeed * 10) / 10;

        let pos = obstacle ? obstacle.xPos : null;
        if (pos !== null) pos = Math.min(pos, 150);

        const w = obstacle ? obstacle.width : null;

        const jumpVel =
            player.jumpVelocity < 0 ? -1 :
            player.jumpVelocity > 0 ? 1 : 0;

        return {
            state: [v, pos, w, jumpVel],
            reward,
            done
        };
    }

    step(action) {

        const r = window.Runner.instance_;

        if (action === "jump") {
            if (!r.tRex.jumping && !r.tRex.ducking) {
                r.tRex.startJump(r.currentSpeed);
            }
        }

        if (action === "duck") {
            r.tRex.setDuck(true);
            setTimeout(() => r.tRex.setDuck(false), 300);
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
    debugger;
    const env = new DinoEnv();
    const agent = new QLearningAgent(["jump", "do_nothing"]);

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
            debugger;
            const action = agent.chooseAction(state);

            const stepResult = env.step(action);

            const nextState = stepResult?.state;
            const reward = stepResult?.reward ?? 0;
            done = stepResult?.done ?? true;
            
            console.log(state, action, reward, nextState);
            agent.learn(state, action, reward, nextState);

            state = nextState;
            totalReward += reward;

            await new Promise(r => setTimeout(r, delay));
        }

        console.log(`Episode ${ep} → reward: ${totalReward}`);

        agent.epsilon = Math.max(0.01, agent.epsilon * 0.995);
    }

    window.__agent = agent;

    return "training_done";
}

// expose globally
window.startDinoTraining = train;

})();