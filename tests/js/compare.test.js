/**
 * Test the compare page helpers: click similarity, the status note, and pane cleanup.
 *
 * Purpose:
 * - Pin similarityAt for a click of another dimension, a far click, and a normal click.
 * - Check that the status note names both dimensions when a click cannot apply.
 * - Check that renderCompare disposes pane a before it rethrows a pane b error.
 *
 * Special Notes:
 * - The document stub holds only what renderCompare and paneRoot touch.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { compareNote, renderCompare, similarityAt } from "../../layerviz/web/compare.js";

/**
 * Build a 2D record of three positions whose scores make the ranks easy to write out.
 *
 * @returns {object} Record with header, positions, and scores in the format.js shape.
 */
function record2d() {
  return {
    header: { kind: "image2d", n: 3, dim: 2, k: 2 },
    positions: new Float32Array([0, 0, 1, 0, 5, 5]),
    scores: new Float32Array([1, 0, 1, 1, 0, 1]),
  };
}

/**
 * Build a minimal DOM element with the members renderCompare and paneRoot use.
 *
 * @returns {object} Element stub with style, dataset, and child lists.
 */
function elementStub() {
  return {
    style: {},
    dataset: {},
    children: [],
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
  };
}

test("similarityAt ranks every position against the clicked one", () => {
  assert.deepEqual(Array.from(similarityAt(record2d(), [0, 0])), [1, 0.5, 0]);
});

test("similarityAt snaps a click outside every position to the nearest one", () => {
  assert.deepEqual(Array.from(similarityAt(record2d(), [100, 100])), [0, 0.5, 1]);
});

test("similarityAt returns null for a click of another dimension or no click", () => {
  assert.equal(similarityAt(record2d(), [0, 0, 0]), null);
  assert.equal(similarityAt(record2d(), null), null);
});

test("compareNote names both dimensions when a 2D click meets a 3D layer", () => {
  const note = compareNote({ forced: false, chosen: "similarity", mode: "pca",
    click: [3, 4], dim: 3 });
  assert.equal(note, "Showing PCA, clicked in 2D, this layer is 3D");
});

test("compareNote asks for a click when no position was clicked", () => {
  const note = compareNote({ forced: false, chosen: "similarity", mode: "pca",
    click: null, dim: 3 });
  assert.equal(note, "Showing PCA until a position is clicked");
});

test("compareNote reports a forced joint fallback and stays empty when the mode applies", () => {
  assert.equal(compareNote({ forced: true, chosen: "pca", mode: "pca", click: null, dim: 2 }),
    "Joint colors are off, this pair has no joint fit");
  assert.equal(compareNote({ forced: false, chosen: "similarity", mode: "similarity",
    click: [0, 0], dim: 2 }), "");
});

test("renderCompare disposes pane a and rethrows when pane b throws", (t) => {
  globalThis.document = { createElement: elementStub };
  t.after(() => { delete globalThis.document; });
  let disposed = 0;
  let calls = 0;
  const render = () => {
    calls += 1;
    if (calls === 2) throw new Error("pane b failed");
    return { dispose() { disposed += 1; } };
  };
  const state = { renderers: { image2d: render }, labels: ["a", "b"], colors: [null, null],
    mode: "pca", onPick: () => {} };
  assert.throws(() => renderCompare(elementStub(), record2d(), record2d(), state),
    /pane b failed/);
  assert.equal(calls, 2);
  assert.equal(disposed, 1);
});
