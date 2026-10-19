/**
 * Draw the architecture graph as SVG: container boxes, leaf nodes, and curved edges.
 *
 * Purpose:
 * - Map layout units to pixels and size the SVG to the whole graph.
 * - Give every box and node a click handler that selects a node id.
 *
 * Special Notes:
 * - Labels are set through textContent, so module names never parse as markup.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const LAYOUT = { unitX: 220, unitY: 100, nodeW: 120, nodeH: 24, pad: 16 };

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
 * Return the layout unit bounds of every leaf node rect and every box.
 *
 * Loops keep a large graph from overflowing the argument limit that spreads hit.
 *
 * @param {object} graph Graph with leaf nodes at x, y and boxes at x0, y0, x1, y1.
 * @returns {{minX: number, minY: number, maxX: number, maxY: number}} The bounds.
 */
function graphBounds(graph) {
  const [halfX, halfY] = [LAYOUT.nodeW / 2 / LAYOUT.unitX, LAYOUT.nodeH / 2 / LAYOUT.unitY];
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  const extend = (x0, y0, x1, y1) => {
    bounds.minX = Math.min(bounds.minX, x0);
    bounds.minY = Math.min(bounds.minY, y0);
    bounds.maxX = Math.max(bounds.maxX, x1);
    bounds.maxY = Math.max(bounds.maxY, y1);
  };
  for (const node of graph.nodes) {
    if (node.leaf) extend(node.x - halfX, node.y - halfY, node.x + halfX, node.y + halfY);
  }
  for (const box of graph.boxes) extend(box.x0, box.y0, box.x1, box.y1);
  return bounds;
}

/**
 * Return a function from layout units to pixels and the pixel size of the whole graph.
 *
 * @param {object} graph Graph with leaf nodes at x, y and boxes at x0, y0, x1, y1.
 * @returns {{px: function(number, number): number[], width: number, height: number}} The map.
 */
function pixelFrame(graph) {
  const { minX, minY, maxX, maxY } = graphBounds(graph);
  const px = (x, y) => [
    (x - minX) * LAYOUT.unitX + LAYOUT.pad, (y - minY) * LAYOUT.unitY + LAYOUT.pad,
  ];
  const [width, height] = px(maxX, maxY).map((v) => v + LAYOUT.pad);
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
