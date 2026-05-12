(() => {

class QLearningAgent {


    constructor(getActionFn, alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.getAction = getActionFn.bind(this) ;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.qTable = new Map();
        this.training = true;
        this.prevAction = null;

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

        const actions = this.getAction(this.prevAction);

        if (!state) return actions[0];

        // If training then explore
        if (Math.random() < this.epsilon && this.training) {
            return actions[Math.floor(Math.random() * actions.length)];
        }

        // Exploit
        const values = actions.map(a => this.getQ(state, a));
        const max = Math.max(...values);

        const best = actions.filter(a => this.getQ(state, a) === max);
        return best[Math.floor(Math.random() * best.length)];
    }

    learn(state, action, reward, nextState, done) {

        // Not training then do not learn
        if(!this.training) return;

        if (!state) return;

        let target;

        if(done){
            target = reward;
        }
        else {
            const actions = this.getAction(this.prevAction);
            const maxNext = Math.max(
                ...actions.map(a => this.getQ(nextState, a)));
            target = reward + this.gamma * maxNext;
        }

        const current = this.getQ(state, action);

        const updated =
            current + this.alpha * (target - current);

        this.qTable.set(this.key(state, action), updated);
    }

   
}

class QLearningAgentNStep extends QLearningAgent{
    constructor(getActionFn, alpha = 0.1, gamma = 0.9, epsilon = 0.1, steps = 5) {
        super(getActionFn, alpha, gamma, epsilon);
        this.steps = steps;
        this.memory = [];
    }

    learn(state, action, reward, nextState, done = false) {


        // Not training then do not learn
        if(!this.training) return;

        if (!state) return;

        this.memory.push({ state, action, reward, nextState, done });

        if (this.memory.length < this.steps && !done){
            return
        }

        const steps = Math.min(this.steps, this.memory.length);
        const first = this.memory[0];
        let G = 0;

        // adding all the rewards for n states 
        for(let i = 0; i < steps; i++){
            G += Math.pow(this.gamma, i) * this.memory[i].reward;
        }

        const lastTransition = this.memory[steps - 1];

        // If not terminated, then also add the next state reward 
        if (!lastTransition.done) {
            const actions = this.getAction(this.prevAction);
            const maxNext = Math.max(
                ...actions.map(a =>
                    this.getQ(lastTransition.nextState, a)
                )
            );
            G += Math.pow(this.gamma, steps) * maxNext;
        }

        const current = this.getQ(first.state, first.action);

        const updated =
            current + this.alpha * (G - current);

        this.qTable.set(
            this.key(first.state, first.action),
            updated
        );

        this.memory.shift();

        if (done) {
            this.flush();
        }

    }

    flush() {

        while (this.memory.length > 0) {

            const steps = this.memory.length;
            const first = this.memory[0];

            let G = 0;

            for (let i = 0; i < steps; i++) {
                G += Math.pow(this.gamma, i) * this.memory[i].reward;
            }

            const current = this.getQ(first.state, first.action);

            const updated =
                current + this.alpha * (G - current);

            this.qTable.set(
                this.key(first.state, first.action),
                updated
            );

            this.memory.shift();
        }
    }

   
}

class DinoEnv {


    constructor(frameSkip = 4) {
        this.frameSkip = frameSkip;
        this.stop = false
        this.episodeData = [];
        document.onkeydown = (event) => {
        if (event.key === "a") {
            this.stop = !this.stop
        }
        this.prevPos = null;
    };

    }

    getState() {
        const r = window.Runner?.instance_;
        if (!r) return null;

        const player = r.tRex;
        const obs = r.horizon.obstacles;
        const done = r.crashed;
        const maxDist = 1000000;

        if (done) {
            return { state: null, reward: -5, done: true };
        }

        if(player.distanceRan *  0.025 > maxDist) {
            return { state: null, reward: 2, done: true };
        }

        let obstacle = obs?.[0] ?? null;
        const obsAboveGround = obstacle  ? ( Runner.defaultDimensions.HEIGHT - Runner.config.BOTTOM_PAD -(obstacle.yPos + obstacle.typeConfig.height)) : null;
        //const timeToCollision = obstacle ? this.bucket(obstacle.xPos / r.currentSpeed, 1, 50) : 999;
        const distance = obstacle ?  this.bucket(obstacle.xPos, 5, 40) : 40;
       
        let reward = 0;

        //console.log(obstacle?.xPos, this.prevPos)
        
        if (obstacle) {
            if (
                (obstacle.xPos < player.xPos + 5)
            ) {
                reward = 1;
            }
            else if ((player.jumping || player.ducking ) && distance >= 40) {
                reward = -0.01;
            }
        }
        else {
            if (player.jumping || player.ducking)
            {
                reward = -0.01;
            } 
        }

        
        const speed = Math.round(r.currentSpeed * 10);
        const obsGap = obstacle ?  this.bucket(obstacle.gap, 50, 6) : null;   // Math.min(6, Math.floor(obstacle.gap / 50));
        const obsWidth = obstacle ? obstacle.width : null;
        const obsHeight = obstacle ? obstacle.yPos : null;
        const playerMode =
                        player.ducking ? -1 :
                        player.jumpVelocity === 0 ? 0 :
                        player.jumpVelocity > 0 ? 1 : 2;

        const result = {
        state: [speed, distance, obsWidth, obsHeight, obsGap, playerMode, obsAboveGround],
        reward,
        done
        };

        this.prevPos = obstacle?.xPos ?? null;

        return result;
    }

    getMode(val) {
        switch (val) {
            case -1:
                return "duck";

            case 0:
                return "ground";

            case 1:
                return "jump";

            case 2:
                return "fall";

            default:
                return "unknown";
        }
    }

    step(action) {
        const KEY = {
            up: 32,
            down: 40
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
        case "up":
            fire("keydown", KEY.up);
            fire("keyup", KEY.up);
            break;

        case "down":
            fire("keydown", KEY.down);
            break;

        case "release":
            fire("keyup", KEY.down);
            break;

        case "nothing":
        default:
            break;
    }

        return this.getState();
    }

    reset() {
        const r = window.Runner.instance_;
        r.restart();
        r.tRex.xPos = 18;
        this.prevPos = null;
        //r.currentSpeed = 10;
        return this.getState();
    }

    sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    async run_episode(agent){


        let result = this.reset();
        if (!result) return 0;

        let state = result.state;
        let done = false;
        let totalReward = 0;
        let gotReward = false;

        while (!done) {

            const action = agent.chooseAction(state);
            
            for (let i = 0; i < this.frameSkip; i++) {
                await new Promise(requestAnimationFrame);
            }
            
            const stepResult = await this.step(action);
            
            const nextState = stepResult.state;
            const reward = stepResult.reward;
            done = stepResult.done ;

            if (reward > 0) {
                gotReward = gotReward + 1;
            }

            agent.learn(state, action, reward, nextState, done);

            agent.prevAction = action;
            
            //  viz.addTransition({
            //     state: { speed: state[0], distance: state[1], obsHeight: state[3] },
            //     action,
            //     reward,
            //     q: 0.4,
            //     nextState: { speed: nextState?.[0] ?? null, distance: nextState?.[1] ?? null, obsHeight: nextState?.[3] ?? null }   
            // });

            state = nextState;
            totalReward += reward;

        }

        //viz.endEpisode();
        
        //await this.sleep(5000); // pause before next episode
        
        //this.updateEpsilon(agent);
       
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

    getActions(prevAction){

        if(prevAction == "down"){
            return ["nothing", "release" ];
        }

        return ["up", "down", "nothing"]
    }

}

//const viz = new RLVisualizer();

async function setupTrainingEnv(episodes = 300) {

    const env = new DinoEnv(4);
    const agent = new QLearningAgentNStep(env.getActions, 0.1, 0.1, 0.1, 5);
    browser = new BrowserControls({env, agent, episodes});
    browser.createHTML();
    window.browserControls = browser;
}

// expose globally
window.setupTrainingEnv = setupTrainingEnv;

})();