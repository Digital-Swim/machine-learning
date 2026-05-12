
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

        // [speed, distance, obsWidth, obsHeight, obsGap, playerMode, obsAboveGround]
        const rows = [
            [
                "Speed",
                "Distance",
                "Obs Width",
                "Obs Height",
                "Obs Gap",
                "Player Mode",
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

        // [speed, distance, obsWidth, obsHeight, obsGap, playerMode, obsAboveGround]
        for (let i = 1; i < lines.length; i++) {
            const [
                s,
                t,
                w,
                h,
                gap,
                m,
                a,
                action,
                qValue
            ] = lines[i].split(",");

            const state = [
                Number(s),
                Number(t),
                Number(w),
                Number(h),
                Number(gap),
                Number(m),
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
