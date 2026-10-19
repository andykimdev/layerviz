/**
 * Reset a view root and give it the legend caption for the current color mode.
 *
 * Purpose:
 * - Hold the one resetRoot that the 2D and 3D views share before they draw.
 *
 * Special Notes:
 * - Touches the DOM only through the document global, so a Node test can stub it.
 */

import { legend } from "./colors.js";

/**
 * Empty a root element and give it a legend line for the current colors.
 *
 * @param {HTMLElement} root Element the view renders into.
 * @param {object} state View state, whose optional mode names the color mode.
 */
export function resetRoot(root, state) {
  root.replaceChildren();
  if (state.mode) {
    const caption = document.createElement("div");
    caption.className = "layerviz-legend";
    caption.textContent = legend(state.mode);
    root.append(caption);
  }
}
