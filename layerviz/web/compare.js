/**
 * Draw a compare layer as two labelled panes with one shared camera and click location.
 *
 * Purpose:
 * - Pick the colors of both panes from one mode: aligned PCA, joint PCA, or similarity.
 * - Render both panes with the view for their kind, sharing state.camera and state.onPick.
 * - Show the layer CKA in the side panel and the joint switch only when the pair has joint colors.
 *
 * Special Notes:
 * - Each pane computes similarity from its own scores at the position nearest the click.
 */

import { colorsFrom, legend } from "./colors.js";
import { cosineSimilarity, nearestIndex, percentileRank } from "./similarity.js";

const PANES = ["a", "b"];
const NOTES = {
  similarity: "Showing PCA until a position is clicked",
  channel: "Showing PCA, channels are not compared",
  joint: "Joint colors are off, this pair has no joint fit",
  dimension: (clicked, layer) => `Showing PCA, clicked in ${clicked}D, this layer is ${layer}D`,
};

/**
 * Compute similarity ranks at the position of a record nearest a click coordinate.
 *
 * Shared with app.js, which colors a single pane the same way.
 *
 * @param {object} record Decoded layer record.
 * @param {number[]|null} click Input coordinate of the click, or null.
 * @returns {Float32Array|null} Percentile rank per position, or null without a usable click.
 */
export function similarityAt(record, click) {
  const { n, dim, k } = record.header;
  if (!click || click.length !== dim) return null;
  const index = nearestIndex(record.positions, dim, click);
  return index < 0 ? null : percentileRank(cosineSimilarity(record.scores, n, k, index));
}

/**
 * Return one pane's colors in a mode, or null when the mode cannot apply to it.
 *
 * @param {object} record Decoded layer record.
 * @param {string} mode One of pca, joint, or similarity. Other modes return null.
 * @param {number[]|null} click Input coordinate of the click, or null.
 * @returns {Uint8Array|null} N*3 colors, or null.
 */
function paneColors(record, mode, click) {
  if (mode === "pca" || (mode === "joint" && record.rgbJoint)) return colorsFrom(record, mode);
  const ranks = mode === "similarity" ? similarityAt(record, click) : null;
  return ranks ? colorsFrom(record, mode, ranks) : null;
}

/**
 * Pick the colors of both panes in the chosen mode, falling back to aligned PCA for both.
 *
 * @param {object[]} records Decoded records of pane a and pane b.
 * @param {string} chosen Mode chosen in the mode select.
 * @param {number[]|null} click Input coordinate of the click, or null.
 * @param {boolean} joint Whether the pair has joint colors.
 * @returns {{mode: string, colors: Uint8Array[]}} The mode drawn and one color array per pane.
 */
function compareColors(records, chosen, click, joint) {
  const mode = chosen === "joint" && !joint ? "pca" : chosen;
  const colors = records.map((record) => paneColors(record, mode, click));
  if (colors.every(Boolean)) return { mode, colors };
  return { mode: "pca", colors: records.map((record) => colorsFrom(record, "pca")) };
}

/**
 * Create one labelled pane in root and return the element its view renders into.
 *
 * @param {HTMLElement} root Element holding the panes.
 * @param {string} pane Pane key, a or b.
 * @param {string} label Model label shown above the pane.
 * @returns {HTMLElement} The view root of the pane.
 */
function paneRoot(root, pane, label) {
  const section = document.createElement("section");
  section.className = "layerviz-pane";
  section.dataset.pane = pane;
  Object.assign(section.style, { flex: "1 1 0", minWidth: "0" });
  const title = Object.assign(document.createElement("h3"), { textContent: label });
  const view = document.createElement("div");
  section.append(title, view);
  root.append(section);
  return view;
}

/**
 * Render one pane of a compare layer and pass the camera it filled on to the next pane.
 *
 * @param {HTMLElement} wrapper Element holding the panes.
 * @param {object} record Decoded record of the pane.
 * @param {number} i Pane index, 0 for pane a and 1 for pane b.
 * @param {object} state Compare state, as renderCompare describes it.
 * @returns {object|null} The handle the view returned, or null.
 * @throws {Error} When the layer kind has no view.
 */
function renderPane(wrapper, record, i, state) {
  const render = state.renderers[record.header.kind];
  if (!render) throw new Error(`no view draws layer kind ${record.header.kind}`);
  const paneState = Object.assign(state.paneStates[i], {
    mode: state.mode, camera: state.camera, onPick: (index) => state.onPick(record, index),
  });
  const handle = render(paneRoot(wrapper, PANES[i], state.labels[i]), record,
    state.colors[i], paneState) ?? null;
  state.camera = paneState.camera;
  return handle;
}

/**
 * Render two panes side by side, sharing one camera and one pick callback.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} recordA Decoded record of pane a.
 * @param {object} recordB Decoded record of pane b.
 * @param {object} state Compare state: labels, mode, colors per pane, renderers by layer kind,
 *   onPick(record, positionIndex), and paneStates kept across renders. state.camera is filled
 *   by the first 3D pane and passed to the second.
 * @returns {{setColors: function(Uint8Array[], string): boolean, dispose: function(): void}} A
 *   handle that recolors both panes in place and sets their legend caption when their views
 *   allow it, and one that stops them.
 * @throws {Error} When the layer kind has no view or a view throws. Pane a is disposed first.
 */
export function renderCompare(root, recordA, recordB, state) {
  const wrapper = Object.assign(document.createElement("div"), { className: "layerviz-compare" });
  Object.assign(wrapper.style, { display: "flex", gap: "12px" });
  root.replaceChildren(wrapper);
  state.paneStates ??= [{}, {}];
  const handles = [];
  try {
    [recordA, recordB].forEach((record, i) => handles.push(renderPane(wrapper, record, i, state)));
  } catch (error) {
    // Pane a may hold a WebGL context or listeners that nothing else would release.
    handles.forEach((handle) => handle?.dispose?.());
    throw error;
  }
  return {
    setColors(colors, caption) {
      if (!handles.every((handle) => handle?.setColors)) return false;
      handles.forEach((handle, i) => handle.setColors(colors[i], caption));
      return true;
    },
    dispose() { handles.forEach((handle) => handle?.dispose?.()); },
  };
}

/**
 * Show a node's CKA with two decimals in the side panel, replacing an earlier CKA row.
 *
 * @param {HTMLElement} stats The side panel dl.
 * @param {object} node Graph node with an optional cka.
 */
function showCka(stats, node) {
  stats.querySelectorAll(".layerviz-cka").forEach((element) => element.remove());
  const term = Object.assign(document.createElement("dt"), { textContent: "CKA" });
  const detail = document.createElement("dd");
  detail.dataset.field = "cka";
  detail.textContent = Number.isFinite(node.cka) ? node.cka.toFixed(2) : "none";
  term.className = detail.className = "layerviz-cka";
  stats.append(term, detail);
}

/**
 * Disable the joint option when the pair has no joint colors, moving a joint choice to PCA.
 *
 * @param {HTMLSelectElement} select The mode select.
 * @param {boolean} joint Whether the pair has joint colors.
 * @returns {boolean} Whether a joint choice was moved to PCA.
 */
function gateJoint(select, joint) {
  select.querySelector('option[value="joint"]').disabled = !joint;
  if (joint || select.value !== "joint") return false;
  select.value = "pca";
  return true;
}

/**
 * Pick the status note that says why the drawn mode differs from the chosen one.
 *
 * @param {object} context Fields forced (a joint choice moved to PCA), chosen (mode select
 *   value), mode (mode drawn), click (input coordinate or null), and dim (layer dimension).
 * @returns {string} The note, empty when the chosen mode was drawn.
 */
export function compareNote({ forced, chosen, mode, click, dim }) {
  if (forced) return NOTES.joint;
  if (mode === chosen) return "";
  if (chosen === "similarity" && click && click.length !== dim) {
    return NOTES.dimension(click.length, dim);
  }
  return NOTES[chosen] ?? "";
}

/**
 * Draw the current compare layer in the chosen mode, recoloring in place when possible.
 *
 * @param {object} view Current view with node, records of pane a and b, and handle.
 * @param {object} env Page context: ui elements, chosen mode, click coordinate or null,
 *   labels, renderers by layer kind, and pick(record, positionIndex).
 */
export function paintCompare(view, env) {
  const { node, records } = view;
  const joint = node.joint !== false;
  const forced = gateJoint(env.ui.mode, joint);
  const chosen = env.ui.mode.value;
  const { mode, colors } = compareColors(records, chosen, env.click, joint);
  view.state.colors = colors;
  const sameMode = view.state.mode === mode && view.handle?.setColors(colors, legend(mode));
  if (!sameMode) {
    view.handle?.dispose?.();
    Object.assign(view.state, { mode, labels: env.labels, renderers: env.renderers,
      onPick: env.pick });
    view.handle = renderCompare(env.ui.view, records[0], records[1], view.state);
  }
  env.ui.status.textContent = compareNote({ forced, chosen, mode, click: env.click,
    dim: records[0].header.dim });
  showCka(env.ui.stats, node);
  Object.assign(env.ui.view.dataset, { node: node.id, mode, channel: env.ui.channel.value });
}
