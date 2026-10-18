/**
 * Run the viewer page: draw the architecture graph and show the layer a click selects.
 *
 * Purpose:
 * - Load graph.json and layer files from inline scripts or by fetch, caching decoded records.
 * - Draw the graph, render the chosen layer in a color mode, and fill the side panel.
 * - Keep the clicked input coordinate across layers, so similarity follows a layer switch.
 *
 * Special Notes:
 * - Layer files load only when their node is clicked, and channel files only in channel mode.
 * - #view carries data-node, data-mode and data-channel once a layer is drawn, for tests.
 * - A mode that cannot apply yet draws PCA with a note in #status, keeping the menu choice.
 * - A compare graph loads both panes of a node and hands drawing to compare.js.
 */

import { base64ToBuffer, parseChannels, parseLayer } from "./format.js";
import { colorsFrom } from "./colors.js";
import { paintCompare, similarityAt } from "./compare.js";
import { renderImage2D, renderTokens, renderVector, renderVolume3D } from "./views2d.js";
import { renderPoints } from "./views3d.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const LAYOUT = { unitX: 220, unitY: 100, nodeW: 120, nodeH: 24, pad: 16 };
const RENDERERS = {
  image2d: renderImage2D, volume3d: renderVolume3D, tokens: renderTokens, vector: renderVector,
  sparse3d: renderPoints, points: renderPoints,
};

const FALLBACK = { similarity: "Showing PCA until a position is clicked",
  channel: "Showing PCA, this layer has no channels", joint: "Showing PCA, no joint colors" };
const cache = new Map();
const app = { graph: null, nodes: new Map(), current: null, click: null, serial: 0, ui: null };

/**
 * Read the graph from the inline #graph script when present, else fetch graph.json.
 *
 * @returns {Promise<object>} The parsed graph.
 * @throws {Error} When graph.json cannot be fetched.
 */
export async function loadGraph() {
  const inline = document.getElementById("graph");
  if (inline) return JSON.parse(inline.textContent);
  const response = await fetch("graph.json");
  if (!response.ok) throw new Error(`graph.json failed to load with HTTP ${response.status}`);
  return response.json();
}

/**
 * Read a bundle file from the inline script with its data-path, else fetch it.
 *
 * @param {string} path Bundle relative path such as layers/L0007.bin.
 * @returns {Promise<ArrayBuffer>} The file bytes.
 * @throws {Error} When the fetch fails.
 */
async function readBytes(path) {
  const inline = document.querySelector(`script[data-path="${CSS.escape(path)}"]`);
  if (inline) return base64ToBuffer(inline.textContent.trim());
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} failed to load with HTTP ${response.status}`);
  return response.arrayBuffer();
}

/**
 * Load and decode a bundle file once, sharing the pending result between callers.
 *
 * @param {string} path Bundle relative path.
 * @param {function(ArrayBuffer): object} parse Decoder from format.js.
 * @returns {Promise<object>} The decoded record. A failed load is dropped from the cache.
 */
function loadCached(path, parse) {
  if (!cache.has(path)) {
    const pending = readBytes(path).then(parse);
    pending.catch(() => cache.delete(path));
    cache.set(path, pending);
  }
  return cache.get(path);
}

/**
 * Load and decode one layer file, cached by path.
 *
 * @param {string} path Bundle relative layer path.
 * @returns {Promise<object>} The record from parseLayer.
 */
export function loadLayer(path) {
  return loadCached(path, parseLayer);
}

/**
 * Create an SVG element with the given attributes.
 *
 * @param {string} tag SVG tag name.
 * @param {object} attrs Attribute names to values.
 * @returns {SVGElement} The element.
 */
function svgElement(tag, attrs = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([name, value]) => element.setAttribute(name, String(value)));
  return element;
}

/**
 * Return a function from layout units to pixels and the pixel size of the whole graph.
 *
 * @param {object} graph Graph with leaf nodes at x, y and boxes at x0, y0, x1, y1.
 * @returns {{px: function(number, number): number[], width: number, height: number}} The map.
 */
function pixelFrame(graph) {
  const leaves = graph.nodes.filter((node) => node.leaf);
  const [halfX, halfY] = [LAYOUT.nodeW / 2 / LAYOUT.unitX, LAYOUT.nodeH / 2 / LAYOUT.unitY];
  const xs = leaves.flatMap((n) => [n.x - halfX, n.x + halfX]);
  const ys = leaves.flatMap((n) => [n.y - halfY, n.y + halfY]);
  graph.boxes.forEach((box) => { xs.push(box.x0, box.x1); ys.push(box.y0, box.y1); });
  const [minX, minY] = [Math.min(...xs), Math.min(...ys)];
  const px = (x, y) => [
    (x - minX) * LAYOUT.unitX + LAYOUT.pad, (y - minY) * LAYOUT.unitY + LAYOUT.pad,
  ];
  const [width, height] = px(Math.max(...xs), Math.max(...ys)).map((v) => v + LAYOUT.pad);
  return { px, width, height };
}

/**
 * Create a clickable group holding a rect and a label, with the full name as a tooltip.
 *
 * @param {string} className Group class, node or box. Box labels sit at the top left.
 * @param {number[]} rect Pixel [x, y, width, height].
 * @param {string} label Text drawn inside the rect.
 * @param {function(): void} onClick Called when the group is clicked.
 * @returns {SVGGElement} The group.
 */
function labelledRect(className, [x, y, width, height], label, onClick) {
  const group = svgElement("g", { class: className });
  const title = svgElement("title");
  const text = svgElement("text", { x: x + 6, y: y + (className === "box" ? 12 : height / 2 + 4) });
  title.textContent = text.textContent = label;
  group.append(title, svgElement("rect", { x, y, width, height }), text);
  // Stopping propagation keeps a click on a node from also selecting its enclosing box.
  group.addEventListener("click", (event) => { event.stopPropagation(); onClick(); });
  return group;
}

/**
 * Create a curved edge from the bottom of one node to the top of another.
 *
 * @param {function(number, number): number[]} px Layout units to pixels.
 * @param {Map<string, object>} byId Graph nodes by id.
 * @param {string[]} edge The [from id, to id] pair.
 * @returns {SVGPathElement} The path.
 * @throws {Error} When the edge names a node the graph lacks.
 */
function edgePath(px, byId, [fromId, toId]) {
  const [from, to] = [byId.get(fromId), byId.get(toId)];
  if (!from || !to) throw new Error(`graph edge ${fromId} to ${toId} names a missing node`);
  const [x1, y1] = px(from.x, from.y).map((v, a) => v + (a ? LAYOUT.nodeH / 2 : 0));
  const [x2, y2] = px(to.x, to.y).map((v, a) => v - (a ? LAYOUT.nodeH / 2 : 0));
  const bend = (y2 - y1) / 2;
  return svgElement("path", {
    class: "edge", d: `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`,
  });
}

/**
 * Draw the graph into an SVG: boxes with container names, leaves with module names, edges.
 *
 * @param {object} graph Graph in the graph.json schema.
 * @param {SVGSVGElement} svg Target element, cleared first.
 * @param {function(string): void} onSelect Called with the node id a click selects. A box
 *   selects the node whose name is the box name.
 */
export function drawGraph(graph, svg, onSelect) {
  const { px, width, height } = pixelFrame(graph);
  svg.replaceChildren();
  Object.entries({ width, height, viewBox: `0 0 ${width} ${height}` })
    .forEach(([name, value]) => svg.setAttribute(name, String(value)));
  const byName = new Map(graph.nodes.map((node) => [node.name, node]));
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  for (const box of [...graph.boxes].sort((a, b) => a.depth - b.depth)) {
    const [[x0, y0], [x1, y1]] = [px(box.x0, box.y0), px(box.x1, box.y1)];
    const own = byName.get(box.name);
    const rect = [x0, y0, x1 - x0, y1 - y0];
    const group = labelledRect("box", rect, box.name, () => own && onSelect(own.id));
    group.dataset.id = own?.id ?? "";
    svg.append(group);
  }
  graph.edges.forEach((edge) => svg.append(edgePath(px, byId, edge)));
  for (const node of graph.nodes.filter((n) => n.leaf)) {
    const [cx, cy] = px(node.x, node.y);
    const rect = [cx - LAYOUT.nodeW / 2, cy - LAYOUT.nodeH / 2, LAYOUT.nodeW, LAYOUT.nodeH];
    const group = labelledRect("node", rect, node.name, () => onSelect(node.id));
    group.dataset.id = node.id;
    group.classList.toggle("empty", !node.file);
    svg.append(group);
  }
}

/**
 * Fill the side panel with one node's name, type, shape, kept positions, and variance.
 *
 * Fractions show as percentages with one decimal, and missing values as none.
 *
 * @param {object} node Graph node.
 */
function fillPanel(node) {
  const percent = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "none");
  const fields = {
    name: node.name, type: node.type ?? "none",
    shape: node.shape?.length ? node.shape.join(" x ") : "none",
    kept: node.kept != null ? `${node.kept} of ${node.total ?? node.kept}` : "none",
    explained: node.explained?.length ? node.explained.map(percent).join(", ") : "none",
    retained: percent(node.retained), skipped: node.skipped ?? "none",
  };
  const rows = Object.entries(fields).flatMap(([field, value]) => {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = field === "explained" ? "explained PC1 to PC3" : field;
    detail.dataset.field = field;
    detail.textContent = String(value);
    return [term, detail];
  });
  app.ui.stats.replaceChildren(...rows);
}

/**
 * Fill the channel select from a node's channel_list, disabling it when there is none.
 *
 * @param {object} node Graph node. Option values are column indices into the channel file.
 */
function fillChannels(node) {
  const list = node.channel_list ?? [];
  const options = list.map((channel, column) => new Option(`channel ${channel}`, String(column)));
  app.ui.channel.replaceChildren(...options);
  app.ui.channel.disabled = list.length === 0;
}

/**
 * Load the node's channel file and return the column of the selected channel.
 *
 * @param {object} node Graph node with an optional channels path.
 * @returns {Promise<Uint8Array|null>} One value per position, or null without channels.
 */
async function channelColumn(node) {
  if (!node.channels || app.ui.channel.disabled) return null;
  const { header, values } = await loadCached(node.channels, parseChannels);
  const column = Number(app.ui.channel.value);
  return values.filter((_, i) => i % header.c === column);
}

/**
 * Return the colors for a view in the chosen mode, falling back to PCA when it cannot apply.
 *
 * @param {object} view Current view with node and record.
 * @param {string} mode Mode chosen in the mode select.
 * @returns {Promise<{mode: string, colors: Uint8Array}>} The mode drawn and its colors.
 */
async function modeColors(view, mode) {
  const { record } = view;
  const extra = mode === "similarity" ? similarityAt(record, app.click)
    : mode === "channel" ? await channelColumn(view.node) : null;
  if (extra || (mode === "joint" && record.rgbJoint)) {
    return { mode, colors: colorsFrom(record, mode, extra ?? undefined) };
  }
  return { mode: "pca", colors: colorsFrom(record, "pca") };
}

/**
 * Render a layer with the view its kind selects, replacing the previous view.
 *
 * @param {object} view Current view with node, record, state and handle.
 * @param {string} mode Color mode drawn, shown by the legend.
 * @param {Uint8Array} colors N*3 colors.
 * @throws {Error} When the layer kind has no view.
 */
function renderView(view, mode, colors) {
  const render = RENDERERS[view.record.header.kind];
  if (!render) throw new Error(`no view draws layer kind ${view.record.header.kind}`);
  view.handle?.dispose?.();
  view.state.mode = mode;
  view.state.onPick = (index) => pickPosition(view, index);
  view.handle = render(app.ui.view, view.record, colors, view.state) ?? null;
}

/**
 * Draw the current layer in the chosen color mode, recoloring through the view handle.
 *
 * Nothing is drawn when a newer selection or paint started while colors were loading.
 * A compare layer, which holds both pane records, is drawn by paintCompare.
 */
async function paint() {
  const [view, chosen] = [app.current, app.ui.mode.value];
  if (!view) return;
  if (view.records) {
    const pick = (record, index) => pickPosition({ record }, index);
    const { ui, click, graph: { meta } } = app;
    return paintCompare(view, { ui, click, labels: meta.labels, renderers: RENDERERS, pick });
  }
  const serial = ++view.paints;
  const { mode, colors } = await modeColors(view, chosen);
  if (view !== app.current || serial !== view.paints) return;
  app.ui.status.textContent = mode === chosen ? "" : FALLBACK[chosen];
  if (view.state.mode === mode && view.handle?.updateColors) view.handle.updateColors(colors);
  else renderView(view, mode, colors);
  Object.assign(app.ui.view.dataset, { node: view.node.id, mode, channel: app.ui.channel.value });
}

/**
 * Store the input coordinate of a clicked position and recolor by similarity to it.
 *
 * @param {object} view View whose position was clicked.
 * @param {number} index Position index in the view's record.
 */
function pickPosition(view, index) {
  const { dim } = view.record.header;
  app.click = Array.from(view.record.positions.subarray(index * dim, index * dim + dim));
  app.ui.click.textContent = app.click.map((v) => Number(v.toFixed(3))).join(", ");
  app.ui.mode.value = "similarity";
  paint().catch(report);
}

/**
 * Stop the current layer view and show a message in its place.
 *
 * @param {string} message Text shown in the view area.
 */
function clearView(message) {
  app.current?.handle?.dispose?.();
  app.current = null;
  app.ui.status.textContent = "";
  app.ui.view.replaceChildren(Object.assign(document.createElement("p"), { textContent: message }));
  delete app.ui.view.dataset.node;
}

/**
 * Select a node: fill the side panel, load its layer, and draw it in the chosen mode.
 *
 * @param {string} nodeId Id of the node to show.
 */
export async function showLayer(nodeId) {
  const node = app.nodes.get(nodeId);
  const serial = ++app.serial;
  app.ui.svg.querySelectorAll("g").forEach((el) => {
    el.classList.toggle("selected", el.dataset.id === nodeId);
  });
  fillPanel(node);
  fillChannels(node);
  clearView(node.file ? "Loading layer" : `No stored data for ${node.name}`);
  if (!node.file) return;
  const record = await loadLayer(node.file);
  const records = node.files?.b ? [record, await loadLayer(node.files.b)] : undefined;
  if (serial === app.serial) {
    app.current = { node, record, records, state: {}, handle: null, paints: 0 };
    await paint();
  }
}

/**
 * Show an error in the status line and the console.
 *
 * @param {Error} error The error.
 */
function report(error) {
  app.ui.status.textContent = error.message;
  console.error(error);
}

/**
 * Collect the page elements, load and draw the graph, and wire the mode and channel controls.
 */
async function main() {
  const byId = (id) => document.getElementById(id);
  app.ui = {
    svg: byId("graph-svg"), view: byId("view"), mode: byId("mode"), channel: byId("channel"),
    stats: byId("stats"), click: byId("click"), status: byId("status"),
  };
  app.graph = await loadGraph();
  app.nodes = new Map(app.graph.nodes.map((node) => [node.id, node]));
  if (app.graph.meta?.mode === "compare") {
    app.ui.mode.querySelector('option[value="joint"]').hidden = false;
  }
  drawGraph(app.graph, app.ui.svg, (id) => showLayer(id).catch(report));
  app.ui.mode.addEventListener("change", () => paint().catch(report));
  app.ui.channel.addEventListener("change", () => paint().catch(report));
}

main().catch(report);
