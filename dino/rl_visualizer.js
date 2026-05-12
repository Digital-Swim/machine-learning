// RLVisualizer.js

class RLVisualizer {
  constructor(containerId = "rl-visualizer") {
    this.width = window.innerWidth;
    this.height = window.innerHeight;

    this.nodesMap = new Map();
    this.links = [];

    this.currentEpisodeTransitions = [];

    // -----------------------------------
    // CREATE CONTAINER DYNAMICALLY
    // -----------------------------------

    let container = document.getElementById(containerId);

    if (!container) {
      debugger;
      container = document.createElement("div");

      container.id = containerId;

      container.style.position = "fixed";
      container.style.top = "0";
      container.style.left = "0";
      container.style.width = "100vw";
      container.style.height = "100vh";
      container.style.background = "#111";
      container.style.zIndex = "99999";
      container.style.overflow = "hidden";

      document.body.appendChild(container);
    }

    this.container = container;

    // -----------------------------------
    // SVG
    // -----------------------------------

    this.svg = d3
      .select(container)
      .append("svg")
      .attr("width", this.width)
      .attr("height", this.height)
      .style("background", "#111");

    this.initializeDefs();

    this.linkLayer = this.svg.append("g");
    this.nodeLayer = this.svg.append("g");
    this.labelLayer = this.svg.append("g");
    this.linkLabelLayer = this.svg.append("g");

    // -----------------------------------
    // SIMULATION
    // -----------------------------------

    this.simulation = d3
      .forceSimulation()
      .force(
        "link",
        d3
          .forceLink()
          .id((d) => d.id)
          .distance(180),
      )
      .force("charge", d3.forceManyBody().strength(-800))
      .force("center", d3.forceCenter(this.width / 2, this.height / 2));

    // -----------------------------------
    // RESIZE
    // -----------------------------------

    window.addEventListener("resize", () => {
      this.width = window.innerWidth;
      this.height = window.innerHeight;

      this.svg.attr("width", this.width).attr("height", this.height);

      this.simulation.force(
        "center",
        d3.forceCenter(this.width / 2, this.height / 2),
      );

      this.simulation.alpha(1).restart();
    });

    this.render();
  }

  initializeDefs() {
    this.svg
      .append("defs")
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 25)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("fill", "#aaa")
      .attr("d", "M0,-5L10,0L0,5");
  }

  createStateId(state) {
    const speed = Math.floor(state.speed / 5) * 5;
    const distance = Math.floor(state.distance / 10) * 10;
    const height = state.obsHeight || 0;

    return `S${speed}_D${distance}_H${height}`;
  }

  createStateLabel(state) {
    return `
      Speed: ${state.speed}
      Distance: ${state.distance}
      ObsHeight: ${state.obsHeight || 0}
`;
  }

  getRewardColor(reward) {
    if (reward > 0) return "#00ff88";
    if (reward < 0) return "#ff4444";

    return "#888";
  }

  addTransition({ state, action, reward, q = 0, nextState }) {
    const sourceId = this.createStateId(state);
    const targetId = this.createStateId(nextState);

    if (!this.nodesMap.has(sourceId)) {
      this.nodesMap.set(sourceId, {
        id: sourceId,
        label: this.createStateLabel(state),
        visits: 1,
      });
    } else {
      this.nodesMap.get(sourceId).visits++;
    }

    if (!this.nodesMap.has(targetId)) {
      this.nodesMap.set(targetId, {
        id: targetId,
        label: this.createStateLabel(nextState),
        visits: 1,
      });
    } else {
      this.nodesMap.get(targetId).visits++;
    }

    const link = {
      source: sourceId,
      target: targetId,
      action,
      reward,
      q,
      animated: false,
    };

    this.links.push(link);

    this.currentEpisodeTransitions.push(link);

    this.render();
  }

  render() {
    const nodes = Array.from(this.nodesMap.values());

    // LINKS

    this.linkSelection = this.linkLayer.selectAll("line").data(this.links);

    this.linkSelection.exit().remove();

    const linkEnter = this.linkSelection
      .enter()
      .append("line")
      .attr("stroke-width", 2)
      .attr("opacity", 0.7)
      .attr("marker-end", "url(#arrow)");

    this.linkSelection = linkEnter.merge(this.linkSelection);

    this.linkSelection
      .attr("stroke", (d) => this.getRewardColor(d.reward))
      .attr("stroke-width", (d) => Math.max(1, Math.abs(d.q) * 8));

    // NODES

    this.nodeSelection = this.nodeLayer
      .selectAll("circle")
      .data(nodes, (d) => d.id);

    this.nodeSelection.exit().remove();

    const nodeEnter = this.nodeSelection
      .enter()
      .append("circle")
      .attr("r", 10)
      .attr("fill", "#4da3ff")
      .call(
        d3
          .drag()
          .on("start", (event, d) => this.dragStarted(event, d))
          .on("drag", (event, d) => this.dragged(event, d))
          .on("end", (event, d) => this.dragEnded(event, d)),
      );

    nodeEnter.append("title").text((d) => d.label);

    this.nodeSelection = nodeEnter.merge(this.nodeSelection);

    this.nodeSelection.attr("r", (d) => 10 + d.visits);

    // LABELS

    this.labelSelection = this.labelLayer
      .selectAll("text")
      .data(nodes, (d) => d.id);

    this.labelSelection.exit().remove();

    const labelEnter = this.labelSelection
      .enter()
      .append("text")
      .attr("fill", "#fff")
      .attr("font-size", "12px");

    this.labelSelection = labelEnter.merge(this.labelSelection);

    this.labelSelection.text((d) => d.id);

    // LINK LABELS

    this.linkLabelSelection = this.linkLabelLayer
      .selectAll("text")
      .data(this.links);

    this.linkLabelSelection.exit().remove();

    const linkLabelEnter = this.linkLabelSelection
      .enter()
      .append("text")
      .attr("fill", "#ccc")
      .attr("font-size", "11px");

    this.linkLabelSelection = linkLabelEnter.merge(this.linkLabelSelection);

    this.linkLabelSelection.text(
      (d) => `${d.action} | r:${d.reward} | q:${d.q.toFixed(2)}`,
    );

    // SIMULATION

    this.simulation.nodes(nodes).on("tick", () => this.ticked());

    this.simulation.force("link").links(this.links);

    this.simulation.alpha(1).restart();
  }

  ticked() {
    this.linkSelection
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    this.nodeSelection.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

    this.labelSelection.attr("x", (d) => d.x + 15).attr("y", (d) => d.y);

    this.linkLabelSelection
      .attr("x", (d) => (d.source.x + d.target.x) / 2)
      .attr("y", (d) => (d.source.y + d.target.y) / 2);
  }

  async endEpisode(delay = 500) {
    for (const transition of this.currentEpisodeTransitions) {
      await this.animateTransition(transition, delay);
    }

    this.currentEpisodeTransitions = [];
  }

  animateTransition(transition, delay) {
    return new Promise((resolve) => {
      const pulse = this.svg
        .append("circle")
        .attr("r", 8)
        .attr("fill", "#ffff00")
        .attr("cx", transition.source.x)
        .attr("cy", transition.source.y);

      pulse
        .transition()
        .duration(delay)
        .attr("cx", transition.target.x)
        .attr("cy", transition.target.y)
        .on("end", () => {
          pulse.remove();

          resolve();
        });
    });
  }

  clear() {
    this.nodesMap.clear();

    this.links = [];

    this.currentEpisodeTransitions = [];

    this.render();
  }

  destroy() {
    this.simulation.stop();

    this.container.remove();
  }

  dragStarted(event, d) {
    if (!event.active) {
      this.simulation.alphaTarget(0.3).restart();
    }

    d.fx = d.x;
    d.fy = d.y;
  }

  dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  dragEnded(event, d) {
    if (!event.active) {
      this.simulation.alphaTarget(0);
    }

    d.fx = null;
    d.fy = null;
  }
}
