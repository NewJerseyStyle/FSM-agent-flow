/**
 * fsm-agent-flow Visual Editor
 * Custom litegraph.js node types and editor logic.
 */

// ── Utilities ──────────────────────────────────────────────────────────

function toast(message, type = "info", duration = 3000) {
    const container = document.getElementById("toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), duration);
}

async function apiCall(method, path, body = null) {
    const opts = { method, headers: {} };
    if (body) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    return res.json();
}

// ── FSM State Node ─────────────────────────────────────────────────────

function FSMStateNode() {
    this.addInput("in", "transition");
    this.addOutput("out", "transition");

    // Widgets
    this.addWidget("text", "Name", "state", (v) => {
        this.properties.state_name = v;
        this.title = v;
    });
    this.addWidget("text", "Objective", "", (v) => {
        this.properties.objective = v;
    });
    this.addWidget("number", "Max Retries", 3, (v) => {
        this.properties.max_retries = v;
    }, { min: 0, max: 20, step: 1, precision: 0 });
    this.addWidget("toggle", "Initial", false, (v) => {
        this.properties.is_initial = v;
        this.updateAppearance();
    });
    this.addWidget("toggle", "Final", false, (v) => {
        this.properties.is_final = v;
        this.updateAppearance();
    });

    // Properties (not widgets)
    this.properties = {
        state_name: "state",
        objective: "",
        max_retries: 3,
        is_initial: false,
        is_final: false,
        tools: [],
        key_results: [],
        execute_module: null,
    };

    this.title = "state";
    this.size = [260, 180];
    this.color = "#2a2a4a";
    this.bgcolor = "#1a1a2e";
}

FSMStateNode.title = "FSM State";
FSMStateNode.desc = "A workflow state node";

FSMStateNode.prototype.updateAppearance = function () {
    if (this.properties.is_initial && this.properties.is_final) {
        this.color = "#4a3060";
    } else if (this.properties.is_initial) {
        this.color = "#1b4332";
    } else if (this.properties.is_final) {
        this.color = "#3d1224";
    } else {
        this.color = "#2a2a4a";
    }
};

FSMStateNode.prototype.onDrawForeground = function (ctx) {
    // Draw badges
    const badges = [];
    if (this.properties.is_initial) badges.push("START");
    if (this.properties.is_final) badges.push("END");
    if (this.properties.tools.length > 0) badges.push("T:" + this.properties.tools.length);
    if (this.properties.key_results.length > 0) badges.push("KR:" + this.properties.key_results.length);

    if (badges.length > 0) {
        ctx.font = "10px Arial";
        ctx.fillStyle = "#888";
        ctx.fillText(badges.join(" | "), 10, this.size[1] - 8);
    }
};

FSMStateNode.prototype.onDblClick = function () {
    openSidebar(this);
};

// Register node type
LiteGraph.registerNodeType("fsm/State", FSMStateNode);

// ── Graph Setup ────────────────────────────────────────────────────────

const graph = new LGraph();
const canvasEl = document.getElementById("graph-canvas");
const graphCanvas = new LGraphCanvas(canvasEl, graph);

// Configure canvas
graphCanvas.background_image = null;
graphCanvas.render_shadows = false;
graphCanvas.clear_background = true;
graphCanvas.render_canvas_border = false;

function resizeCanvas() {
    const container = document.getElementById("main-container");
    canvasEl.width = container.clientWidth;
    canvasEl.height = container.clientHeight;
    graphCanvas.resize();
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

graph.start();

// ── Sidebar ────────────────────────────────────────────────────────────

let selectedNode = null;

function openSidebar(node) {
    selectedNode = node;
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.remove("hidden");

    document.getElementById("prop-name").value = node.properties.state_name || "";
    document.getElementById("prop-objective").value = node.properties.objective || "";
    document.getElementById("prop-retries").value = node.properties.max_retries || 3;
    document.getElementById("prop-initial").checked = !!node.properties.is_initial;
    document.getElementById("prop-final").checked = !!node.properties.is_final;
    document.getElementById("prop-execute").value = node.properties.execute_module || "";
    document.getElementById("prop-tools").value = (node.properties.tools || []).join(", ");

    renderKeyResults(node.properties.key_results || []);
}

function closeSidebar() {
    document.getElementById("sidebar").classList.add("hidden");
    selectedNode = null;
}

function saveSidebarToNode() {
    if (!selectedNode) return;

    const name = document.getElementById("prop-name").value.trim();
    selectedNode.properties.state_name = name;
    selectedNode.title = name || "state";
    // Also update the Name widget value
    if (selectedNode.widgets) {
        const nameWidget = selectedNode.widgets.find(w => w.name === "Name");
        if (nameWidget) nameWidget.value = name;
        const objWidget = selectedNode.widgets.find(w => w.name === "Objective");
        if (objWidget) objWidget.value = document.getElementById("prop-objective").value;
        const retWidget = selectedNode.widgets.find(w => w.name === "Max Retries");
        if (retWidget) retWidget.value = parseInt(document.getElementById("prop-retries").value) || 3;
        const initWidget = selectedNode.widgets.find(w => w.name === "Initial");
        if (initWidget) initWidget.value = document.getElementById("prop-initial").checked;
        const finalWidget = selectedNode.widgets.find(w => w.name === "Final");
        if (finalWidget) finalWidget.value = document.getElementById("prop-final").checked;
    }

    selectedNode.properties.objective = document.getElementById("prop-objective").value;
    selectedNode.properties.max_retries = parseInt(document.getElementById("prop-retries").value) || 3;
    selectedNode.properties.is_initial = document.getElementById("prop-initial").checked;
    selectedNode.properties.is_final = document.getElementById("prop-final").checked;
    selectedNode.properties.execute_module = document.getElementById("prop-execute").value.trim() || null;

    const toolsStr = document.getElementById("prop-tools").value.trim();
    selectedNode.properties.tools = toolsStr ? toolsStr.split(",").map(t => t.trim()).filter(Boolean) : [];

    // Gather KRs from DOM
    selectedNode.properties.key_results = gatherKeyResults();

    selectedNode.updateAppearance();
    graphCanvas.setDirty(true, true);
}

function renderKeyResults(krs) {
    const container = document.getElementById("kr-list");
    container.innerHTML = "";
    krs.forEach((kr, i) => {
        const item = document.createElement("div");
        item.className = "kr-item";
        item.dataset.index = i;
        item.innerHTML = `
            <div class="kr-header">
                <span class="kr-label">Key Result #${i + 1}</span>
                <button class="kr-remove" onclick="removeKR(${i})">&times;</button>
            </div>
            <input type="text" class="kr-name" placeholder="name" value="${escapeHtml(kr.name || "")}">
            <input type="text" class="kr-desc" placeholder="description" value="${escapeHtml(kr.description || "")}">
            <div class="kr-label">Check expression (optional)</div>
            <input type="text" class="kr-check" placeholder='e.g. len(str(output)) >= 200' value="${escapeHtml(kr.check || "")}">
        `;
        container.appendChild(item);
    });
}

function gatherKeyResults() {
    const items = document.querySelectorAll("#kr-list .kr-item");
    const krs = [];
    items.forEach(item => {
        const name = item.querySelector(".kr-name").value.trim();
        const desc = item.querySelector(".kr-desc").value.trim();
        const check = item.querySelector(".kr-check").value.trim();
        if (name || desc) {
            krs.push({ name, description: desc, check: check || null });
        }
    });
    return krs;
}

function removeKR(index) {
    if (!selectedNode) return;
    selectedNode.properties.key_results.splice(index, 1);
    renderKeyResults(selectedNode.properties.key_results);
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Sidebar event listeners
document.getElementById("sidebar-close").addEventListener("click", () => {
    saveSidebarToNode();
    closeSidebar();
});

document.getElementById("btn-add-kr").addEventListener("click", () => {
    if (!selectedNode) return;
    selectedNode.properties.key_results.push({ name: "", description: "", check: null });
    renderKeyResults(selectedNode.properties.key_results);
});

// Auto-save sidebar on input change
["prop-name", "prop-objective", "prop-retries", "prop-initial", "prop-final", "prop-execute", "prop-tools"].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener("change", saveSidebarToNode);
    if (el.tagName === "INPUT" && (el.type === "text" || el.type === "number")) {
        el.addEventListener("input", saveSidebarToNode);
    }
    if (el.tagName === "TEXTAREA") {
        el.addEventListener("input", saveSidebarToNode);
    }
});

// ── Graph ↔ JSON Conversion ────────────────────────────────────────────

function graphToWorkflow() {
    const workflow = {
        version: "2.0",
        objective: "",
        states: {},
        transitions: {},
        graph_layout: {},
    };

    // Gather all state nodes
    const stateNodes = [];
    const nodeById = {};
    graph._nodes.forEach(node => {
        if (node.type === "fsm/State") {
            stateNodes.push(node);
            nodeById[node.id] = node;
        }
    });

    // Determine workflow objective (from initial state or first state)
    const initial = stateNodes.find(n => n.properties.is_initial) || stateNodes[0];
    if (initial) {
        workflow.objective = initial.properties.objective || "Untitled workflow";
    }

    // Build states and layout
    stateNodes.forEach(node => {
        const name = node.properties.state_name || `state_${node.id}`;
        workflow.states[name] = {
            objective: node.properties.objective || "",
            key_results: (node.properties.key_results || []).map(kr => ({
                name: kr.name,
                description: kr.description,
                check: kr.check || null,
            })),
            tools: node.properties.tools || [],
            max_retries: node.properties.max_retries || 3,
            is_initial: !!node.properties.is_initial,
            is_final: !!node.properties.is_final,
            execute_module: node.properties.execute_module || null,
        };
        workflow.graph_layout[name] = [node.pos[0], node.pos[1]];
    });

    // Build transitions from connections
    stateNodes.forEach(node => {
        const name = node.properties.state_name || `state_${node.id}`;
        // Find output connections from slot 0 ("out")
        const outputLinks = node.outputs[0]?.links || [];
        let target = null;
        if (outputLinks.length > 0) {
            const link = graph.links[outputLinks[0]];
            if (link) {
                const targetNode = graph.getNodeById(link.target_id);
                if (targetNode && targetNode.type === "fsm/State") {
                    target = targetNode.properties.state_name || `state_${targetNode.id}`;
                }
            }
        }
        workflow.transitions[name] = target;
    });

    return workflow;
}

function workflowToGraph(data) {
    graph.clear();

    const stateNames = Object.keys(data.states || {});
    const nodeByName = {};
    const layout = data.graph_layout || {};

    // Create nodes
    stateNames.forEach((name, i) => {
        const stateData = data.states[name];
        const node = LiteGraph.createNode("fsm/State");

        // Set position from layout or auto-layout
        if (layout[name]) {
            node.pos = [layout[name][0], layout[name][1]];
        } else {
            node.pos = [150 + i * 300, 250];
        }

        node.properties.state_name = name;
        node.properties.objective = stateData.objective || "";
        node.properties.max_retries = stateData.max_retries != null ? stateData.max_retries : 3;
        node.properties.is_initial = !!stateData.is_initial;
        node.properties.is_final = !!stateData.is_final;
        node.properties.tools = stateData.tools || [];
        node.properties.key_results = (stateData.key_results || []).map(kr => ({
            name: kr.name || "",
            description: kr.description || "",
            check: kr.check || null,
        }));
        node.properties.execute_module = stateData.execute_module || null;

        node.title = name;

        // Sync widgets
        if (node.widgets) {
            const nameW = node.widgets.find(w => w.name === "Name");
            if (nameW) nameW.value = name;
            const objW = node.widgets.find(w => w.name === "Objective");
            if (objW) objW.value = stateData.objective || "";
            const retW = node.widgets.find(w => w.name === "Max Retries");
            if (retW) retW.value = node.properties.max_retries;
            const initW = node.widgets.find(w => w.name === "Initial");
            if (initW) initW.value = node.properties.is_initial;
            const finalW = node.widgets.find(w => w.name === "Final");
            if (finalW) finalW.value = node.properties.is_final;
        }

        node.updateAppearance();
        graph.add(node);
        nodeByName[name] = node;
    });

    // Create connections from transitions
    const transitions = data.transitions || {};
    Object.entries(transitions).forEach(([src, dst]) => {
        if (dst && nodeByName[src] && nodeByName[dst]) {
            nodeByName[src].connect(0, nodeByName[dst], 0);
        }
    });

    graphCanvas.setDirty(true, true);
}

// ── Toolbar Actions ────────────────────────────────────────────────────

// New
document.getElementById("btn-new").addEventListener("click", () => {
    if (graph._nodes.length > 0 && !confirm("Create new workflow? Unsaved changes will be lost.")) {
        return;
    }
    graph.clear();
    // Add one initial state
    const node = LiteGraph.createNode("fsm/State");
    node.pos = [200, 250];
    node.properties.state_name = "start";
    node.properties.is_initial = true;
    node.title = "start";
    if (node.widgets) {
        const nameW = node.widgets.find(w => w.name === "Name");
        if (nameW) nameW.value = "start";
        const initW = node.widgets.find(w => w.name === "Initial");
        if (initW) initW.value = true;
    }
    node.updateAppearance();
    graph.add(node);
    graphCanvas.setDirty(true, true);
    toast("New workflow created", "info");
});

// Add State
document.getElementById("btn-add-state").addEventListener("click", () => {
    const node = LiteGraph.createNode("fsm/State");
    // Place near center of visible area
    const center = graphCanvas.convertOffsetToCanvas([canvasEl.width / 2, canvasEl.height / 2]);
    node.pos = [center[0] - 130, center[1] - 90];
    const count = graph._nodes.filter(n => n.type === "fsm/State").length;
    const name = `state_${count + 1}`;
    node.properties.state_name = name;
    node.title = name;
    if (node.widgets) {
        const nameW = node.widgets.find(w => w.name === "Name");
        if (nameW) nameW.value = name;
    }
    graph.add(node);
    graphCanvas.setDirty(true, true);
    toast("State added — double-click to edit", "info");
});

// Open
document.getElementById("btn-open").addEventListener("click", () => {
    showModal("Open Workflow", `
        <label style="display:block;margin-bottom:8px;font-size:13px;">File path:</label>
        <input type="text" id="open-path" placeholder="workflow.json" style="width:100%;">
    `, async () => {
        const path = document.getElementById("open-path").value.trim();
        if (!path) return;
        const data = await apiCall("GET", `/api/load?path=${encodeURIComponent(path)}`);
        if (data.error) {
            toast(data.error, "error");
        } else {
            workflowToGraph(data);
            toast("Workflow loaded", "success");
        }
        closeModal();
    });
});

// Save
document.getElementById("btn-save").addEventListener("click", () => {
    const workflow = graphToWorkflow();
    showModal("Save Workflow", `
        <label style="display:block;margin-bottom:8px;font-size:13px;">File path:</label>
        <input type="text" id="save-path" placeholder="workflow.json" value="workflow.json" style="width:100%;">
        <div style="margin-top:8px;font-size:12px;color:#666;">
            Objective: <input type="text" id="save-objective" value="${escapeHtml(workflow.objective)}" style="width:100%;margin-top:4px;">
        </div>
    `, async () => {
        const path = document.getElementById("save-path").value.trim();
        if (!path) return;
        workflow.objective = document.getElementById("save-objective").value.trim();
        const result = await apiCall("POST", "/api/save", { path, workflow });
        if (result.error) {
            toast(result.error, "error");
        } else {
            toast(`Saved to ${result.path}`, "success");
        }
        closeModal();
    });
});

// Validate
document.getElementById("btn-validate").addEventListener("click", async () => {
    const workflow = graphToWorkflow();
    const result = await apiCall("POST", "/api/validate", { workflow });
    if (result.valid) {
        showModal("Validation", `<p class="success-msg">Workflow is valid!</p>`, closeModal);
    } else {
        const errorHtml = result.errors.map(e => `<li>${escapeHtml(e)}</li>`).join("");
        showModal("Validation Errors", `<ul class="error-list">${errorHtml}</ul>`, closeModal);
    }
});

// Export Python
document.getElementById("btn-export").addEventListener("click", async () => {
    const workflow = graphToWorkflow();
    const result = await apiCall("POST", "/api/export-python", { workflow });
    if (result.error) {
        toast(result.error, "error");
        return;
    }
    showModal("Generated Python Code", `
        <pre>${escapeHtml(result.code)}</pre>
        <div style="margin-top:8px;">
            <label style="font-size:13px;">Save to file:</label>
            <input type="text" id="export-path" placeholder="workflow.py" value="workflow.py" style="width:100%;margin-top:4px;">
        </div>
    `, async () => {
        const path = document.getElementById("export-path").value.trim();
        if (path) {
            // Save the Python code via a simple POST
            await apiCall("POST", "/api/save", {
                path,
                workflow: result.code,  // server will write raw string
            });
            toast(`Exported to ${path}`, "success");
        }
        closeModal();
    });
});

// Auto Layout
document.getElementById("btn-auto-layout").addEventListener("click", () => {
    autoLayout();
    toast("Layout applied", "info");
});

function autoLayout() {
    const nodes = graph._nodes.filter(n => n.type === "fsm/State");
    if (nodes.length === 0) return;

    // Build adjacency from connections
    const nameToNode = {};
    nodes.forEach(n => {
        nameToNode[n.properties.state_name || n.id] = n;
    });

    // Find initial node
    let initial = nodes.find(n => n.properties.is_initial) || nodes[0];

    // BFS ordering
    const visited = new Set();
    const order = [];
    const queue = [initial];
    visited.add(initial.id);

    while (queue.length > 0) {
        const current = queue.shift();
        order.push(current);
        // Find connected target
        const outputLinks = current.outputs[0]?.links || [];
        for (const linkId of outputLinks) {
            const link = graph.links[linkId];
            if (link) {
                const target = graph.getNodeById(link.target_id);
                if (target && !visited.has(target.id)) {
                    visited.add(target.id);
                    queue.push(target);
                }
            }
        }
    }

    // Add unvisited nodes
    nodes.forEach(n => {
        if (!visited.has(n.id)) order.push(n);
    });

    // Arrange left to right
    const startX = 100;
    const startY = 200;
    const gapX = 320;
    order.forEach((node, i) => {
        node.pos = [startX + i * gapX, startY];
    });

    graphCanvas.setDirty(true, true);
}

// ── Templates ──────────────────────────────────────────────────────────

async function loadTemplates() {
    const select = document.getElementById("template-select");
    try {
        const templates = await apiCall("GET", "/api/templates");
        templates.forEach(t => {
            const opt = document.createElement("option");
            opt.value = t.name;
            opt.textContent = t.name;
            opt._data = t.data;
            select.appendChild(opt);
        });
    } catch (e) {
        // Templates endpoint not available, skip
    }
}

document.getElementById("template-select").addEventListener("change", function () {
    const opt = this.options[this.selectedIndex];
    if (opt._data) {
        workflowToGraph(opt._data);
        toast(`Template "${opt.value}" loaded`, "success");
    }
    this.selectedIndex = 0;
});

loadTemplates();

// ── Modal ──────────────────────────────────────────────────────────────

let modalOkCallback = null;

function showModal(title, bodyHtml, onOk) {
    document.getElementById("modal-title").textContent = title;
    document.getElementById("modal-body").innerHTML = bodyHtml;
    document.getElementById("modal-overlay").classList.remove("hidden");
    modalOkCallback = onOk;
}

function closeModal() {
    document.getElementById("modal-overlay").classList.add("hidden");
    modalOkCallback = null;
}

document.getElementById("modal-ok").addEventListener("click", () => {
    if (modalOkCallback) modalOkCallback();
});
document.getElementById("modal-cancel").addEventListener("click", closeModal);
document.getElementById("modal-close").addEventListener("click", closeModal);

// Close modal on overlay click
document.getElementById("modal-overlay").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeModal();
});

// ── Keyboard Shortcuts ─────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
    // Ctrl+S = save
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        document.getElementById("btn-save").click();
    }
    // Ctrl+O = open
    if ((e.ctrlKey || e.metaKey) && e.key === "o") {
        e.preventDefault();
        document.getElementById("btn-open").click();
    }
    // Escape = close sidebar/modal
    if (e.key === "Escape") {
        if (!document.getElementById("modal-overlay").classList.contains("hidden")) {
            closeModal();
        } else if (!document.getElementById("sidebar").classList.contains("hidden")) {
            saveSidebarToNode();
            closeSidebar();
        }
    }
});

// ── Load initial file from URL param ───────────────────────────────────

(async function loadInitial() {
    const params = new URLSearchParams(window.location.search);
    const file = params.get("file");
    if (file) {
        const data = await apiCall("GET", `/api/load?path=${encodeURIComponent(file)}`);
        if (!data.error) {
            workflowToGraph(data);
            toast(`Loaded ${file}`, "success");
        }
    }
})();

// ── Status ─────────────────────────────────────────────────────────────

document.getElementById("status-msg").textContent = "Double-click a node to edit properties";
