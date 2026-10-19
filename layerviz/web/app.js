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
 * - Every load error names its file. The serve hint for a split bundle opened from disk lives
 *   in index.html, because browsers never run this module on such a page.
 * - The SVG graph drawing lives in graphview.js.
 */

import { base64ToBuffer, parseChannels, parseLayer } from "./format.js";
import { colorsFrom, legend } from "./colors.js";
import { paintCompare, similarityAt } from "./compare.js";
import { drawGraph } from "./graphview.js";
import { renderImage2D, renderTokens, renderVector, renderVolume3D } from "./views2d.js";
import { renderPoints } from "./views3d.js";

const RENDERERS = {
  image2d: renderImage2D, volume3d: renderVolume3D, tokens: renderTokens, vector: renderVector,
  sparse3d: renderPoints, points: renderPoints,
};

const FALLBACK = { similarity: "Showing PCA until a position is clicked",
  channel: "Showing PCA, this layer has no channels", joint: "Showing PCA, no joint colors" };
const cache = new Map();
const app = { graph: null, nodes: new Map(), current: null, click: null, serial: 0, ui: null };

/**
 * Fetch one bundle file, naming the file in every error.
 *
 * @param {string} path Bundle relative path.
 * @returns {Promise<Response>} The ok response.
 * @throws {Error} When the connection fails or the HTTP status is not ok.
 */
async function fetchFile(path) {
  let response;
  try {
    response = await fetch(path);
  } catch (error) {
    throw new Error(`${path} could not be fetched: ${error.message}`, { cause: error });
  }
  if (!response.ok) throw new Error(`${path} failed to load with HTTP ${response.status}`);
  return response;
}

/**
 * Parse JSON text, naming the file it came from when it is not valid.
 *
 * @param {string} path Bundle relative path of the text.
 * @param {string} text JSON text.
 * @returns {object} The parsed value.
 * @throws {Error} When the text is not valid JSON.
 */
function parseJson(path, text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${path} is not valid JSON: ${error.message}`, { cause: error });
  }
}

/**
 * Read the graph from the inline #graph script when present, else fetch graph.json.
 *
 * @returns {Promise<object>} The parsed graph.
 * @throws {Error} When graph.json cannot be fetched or is not valid JSON, naming the file.
 */
export async function loadGraph() {
  const inline = document.getElementById("graph");
  if (inline) return parseJson("graph.json", inline.textContent);
  const response = await fetchFile("graph.json");
  return parseJson("graph.json", await response.text());
}

/**
 * Decode the base64 text of an inline bundle file, naming the file when it is not valid.
 *
 * @param {string} path Bundle relative path of the inline file.
 * @param {string} text Base64 text.
 * @returns {ArrayBuffer} The file bytes.
 * @throws {Error} When the text is not valid base64.
 */
function decodeInline(path, text) {
  try {
    return base64ToBuffer(text);
  } catch (error) {
    throw new Error(`${path}: ${error.message}`, { cause: error });
  }
}

/**
 * Read a bundle file from the inline script with its data-path, else fetch it.
 *
 * @param {string} path Bundle relative path such as layers/L0007.bin.
 * @returns {Promise<ArrayBuffer>} The file bytes.
 * @throws {Error} When the file cannot be read or decoded, naming the file.
 */
async function readBytes(path) {
  const inline = document.querySelector(`script[data-path="${CSS.escape(path)}"]`);
  if (inline) return decodeInline(path, inline.textContent.trim());
  return (await fetchFile(path)).arrayBuffer();
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
 * Return the colors of one paint, or null when a newer selection or paint replaced it.
 *
 * @param {object} view View the paint draws.
 * @param {string} chosen Mode chosen in the mode select.
 * @param {number} serial Paint number of this paint on the view.
 * @returns {Promise<{mode: string, colors: Uint8Array}|null>} The mode drawn and its colors.
 */
async function currentColors(view, chosen, serial) {
  const stale = () => view !== app.current || serial !== view.paints;
  try {
    const drawn = await modeColors(view, chosen);
    return stale() ? null : drawn;
  } catch (error) {
    // A newer view owns the status line, so a replaced paint's failure is only logged.
    if (!stale()) throw error;
    console.warn("Dropped the failure of a replaced paint:", error);
    return null;
  }
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
 * Recolor a view on its own canvas and set its legend, keeping the camera and slab.
 *
 * @param {object} view Current view whose handle has setColors.
 * @param {string} mode Color mode drawn.
 * @param {Uint8Array} colors N*3 colors.
 */
function recolorView(view, mode, colors) {
  view.state.mode = mode;
  view.handle.setColors(colors, legend(mode));
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
  const drawn = await currentColors(view, chosen, ++view.paints);
  if (!drawn) return;
  const { mode, colors } = drawn;
  app.ui.status.textContent = mode === chosen ? "" : FALLBACK[chosen];
  if (view.handle?.setColors) recolorView(view, mode, colors);
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
  let loaded;
  try {
    loaded = await loadRecords(node);
  } catch (error) {
    // A later selection owns the status line, so a stale failure is only logged.
    if (serial === app.serial) throw error;
    console.warn(`Dropped the failure of replaced layer ${node.file}:`, error);
    return;
  }
  if (serial === app.serial) {
    app.current = { node, ...loaded, state: {}, handle: null, paints: 0 };
    await paint();
  }
}

/**
 * Load the layer of a node, and for a compare node both pane layers.
 *
 * @param {object} node Graph node with a file.
 * @returns {Promise<{record: object, records: (object[]|undefined)}>} The first pane record,
 *   and both pane records when the node pairs two files.
 */
async function loadRecords(node) {
  const record = await loadLayer(node.file);
  return { record, records: node.files?.b ? [record, await loadLayer(node.files.b)] : undefined };
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
