/**
 * Draw sparse3d and points layers as a rotatable three.js point cloud with a slab control.
 *
 * Purpose:
 * - Render positions as WebGL points colored by any color mode, with picking by click.
 * - Show only the points inside a slab whose axis, center, and thickness are range inputs.
 *
 * Special Notes:
 * - three.js and OrbitControls load from ./vendor/ so bundles work offline.
 * - The pure geometry lives in geometry3d.js, so Node tests it without WebGL.
 */

import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import {
  boundsOf, canvasSize, clampSlab, nearestOnRay, rgbaColors, screenAxes, slabMask, toScreen,
} from "./geometry3d.js";
import { resetRoot } from "./legend.js";

const CLICK_SLOP_PX = 4;
const SLAB_STEPS = 100;

/**
 * Create a camera looking at the point cloud from outside its bounding sphere.
 *
 * @param {{center: number[], radius: number}} bounds Bounds of the points.
 * @returns {THREE.PerspectiveCamera} The camera.
 */
function makeCamera(bounds) {
  const radius = Math.max(bounds.radius, 1e-6);
  const camera = new THREE.PerspectiveCamera(50, 1, radius / 100, radius * 100);
  const [x, y, z] = bounds.center;
  camera.position.set(x, y, z + radius * 3);
  return camera;
}

/**
 * Create the Points object with a position attribute and an RGBA color attribute.
 *
 * @param {Float32Array} positions N*3 coordinates in screen axis order, from toScreen.
 * @param {Uint8Array} rgba N*4 colors from rgbaColors.
 * @param {number} radius Bounding sphere radius, which sets the point size.
 * @returns {THREE.Points} The points, whose alphaTest discards hidden ones.
 */
function makePoints(positions, rgba, radius) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(rgba, 4, true));
  const material = new THREE.PointsMaterial({
    size: Math.max(radius / 60, 1e-6),
    vertexColors: true,
    transparent: true,
    alphaTest: 0.5,
  });
  return new THREE.Points(geometry, material);
}

/**
 * Create a labelled range input that calls onInput with its numeric value.
 *
 * @param {string} label Text shown before the slider.
 * @param {{min: number, max: number, step: number, value: number}} range Slider settings.
 * @param {function(number): void} onInput Called on every input event.
 * @returns {{wrapper: HTMLElement, input: HTMLInputElement}} The label and its slider.
 */
function rangeControl(label, range, onInput) {
  const input = document.createElement("input");
  Object.assign(input, { type: "range", min: range.min, max: range.max, step: range.step });
  input.value = String(range.value);
  input.addEventListener("input", () => onInput(Number(input.value)));
  const wrapper = document.createElement("label");
  wrapper.className = "layerviz-slab";
  wrapper.append(`${label} `, input);
  return { wrapper, input };
}

/**
 * Build the axis, center, and thickness sliders that rewrite state.slab.
 *
 * @param {object} state View state with a valid slab.
 * @param {{min: number[], max: number[], center: number[]}} bounds Bounds of the points.
 * @param {string[]} names Stored column name per screen axis, from screenAxes, shown as the
 *   axis slider title.
 * @param {function(): void} onChange Called after state.slab changes.
 * @returns {HTMLElement} A wrapper holding the three sliders. The axis slider runs over the
 *   screen axes in (z, y, x) order, while state.slab.axis keeps the screen axis index.
 */
function slabControls(state, bounds, names, onChange) {
  const extent = (axis) => Math.max(bounds.max[axis] - bounds.min[axis], 1e-6);
  // Slider value 0 is screen axis 2, so the slider lists z, y, x like a stored voxel index.
  const screenAxisOf = (value) => 2 - value;
  const { axis } = state.slab;
  const center = rangeControl("center", {
    min: bounds.min[axis], max: bounds.max[axis], step: extent(axis) / SLAB_STEPS,
    value: state.slab.center,
  }, (value) => { state.slab.center = value; onChange(); });
  const thickness = rangeControl("thickness", {
    min: 0, max: extent(axis), step: extent(axis) / SLAB_STEPS, value: state.slab.thickness,
  }, (value) => { state.slab.thickness = value; onChange(); });
  const range = { min: 0, max: 2, step: 1, value: screenAxisOf(axis) };
  const axisInput = rangeControl("axis", range, (value) => {
    const next = screenAxisOf(value);
    state.slab = { axis: next, center: bounds.center[next], thickness: extent(next) };
    Object.assign(center.input, { min: bounds.min[next], max: bounds.max[next] });
    center.input.step = String(extent(next) / SLAB_STEPS);
    center.input.value = String(state.slab.center);
    Object.assign(thickness.input, { max: extent(next), step: extent(next) / SLAB_STEPS });
    thickness.input.value = String(state.slab.thickness);
    axisInput.input.title = names[next];
    onChange();
  });
  axisInput.input.title = names[axis];
  const wrapper = document.createElement("div");
  wrapper.className = "layerviz-slab-controls";
  wrapper.append(axisInput.wrapper, center.wrapper, thickness.wrapper);
  return wrapper;
}

/**
 * Call onClick with a pointer event only when the pointer did not drag between down and up.
 *
 * @param {HTMLCanvasElement} canvas Canvas to listen on.
 * @param {function(PointerEvent): void} onClick Called for clicks that are not orbit drags.
 */
function onStillClick(canvas, onClick) {
  let down = null;
  canvas.addEventListener("pointerdown", (event) => { down = [event.clientX, event.clientY]; });
  canvas.addEventListener("pointerup", (event) => {
    if (down && Math.hypot(event.clientX - down[0], event.clientY - down[1]) <= CLICK_SLOP_PX) {
      onClick(event);
    }
    down = null;
  });
}

/**
 * Return the world space ray under a pointer event, as origin and direction arrays.
 *
 * @param {PointerEvent} event Pointer event on the canvas.
 * @param {HTMLCanvasElement} canvas The canvas the event hit.
 * @param {THREE.Camera} camera Camera the canvas renders through.
 * @returns {{origin: number[], direction: number[]}} The ray.
 */
function rayUnder(event, canvas, camera) {
  const box = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - box.left) / box.width) * 2 - 1,
    -((event.clientY - box.top) / box.height) * 2 + 1,
  );
  const caster = new THREE.Raycaster();
  caster.setFromCamera(ndc, camera);
  return { origin: caster.ray.origin.toArray(), direction: caster.ray.direction.toArray() };
}

/**
 * Set the legend line of a view root to a caption, adding the line first when it is missing.
 *
 * @param {HTMLElement} root Element the view renders into.
 * @param {string} caption Legend text for the current color mode.
 */
export function showCaption(root, caption) {
  let line = root.querySelector(".layerviz-legend");
  if (!line) {
    line = document.createElement("div");
    line.className = "layerviz-legend";
    root.prepend(line);
  }
  line.textContent = caption;
}

/**
 * Return the outer height of an element, its border box plus its vertical margins.
 *
 * @param {Element} element Element to measure.
 * @returns {number} Height in CSS pixels.
 */
function outerHeight(element) {
  const style = getComputedStyle(element);
  const margins = (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0);
  return element.getBoundingClientRect().height + margins;
}

/**
 * Measure the pane content box and the room its other content takes beside the canvas.
 *
 * @param {HTMLElement} root Pane holding the canvas.
 * @param {HTMLCanvasElement} canvas Canvas inside the pane.
 * @returns {object} The pane argument of geometry3d.js canvasSize.
 */
function measurePane(root, canvas) {
  const style = getComputedStyle(root);
  const padding = ["Top", "Right", "Bottom", "Left"]
    .map((side) => parseFloat(style[`padding${side}`]) || 0);
  const outer = [...root.children].reduce((sum, child) => sum + outerHeight(child), 0);
  return {
    width: root.clientWidth,
    height: root.clientHeight,
    padding,
    reserved: outer - canvas.getBoundingClientRect().height,
    overflowing: root.scrollHeight > root.clientHeight,
  };
}

/**
 * Size the canvas to its pane now and again whenever the pane resizes.
 *
 * @param {HTMLElement} root Pane holding the canvas and the slab controls.
 * @param {THREE.WebGLRenderer} renderer Renderer whose canvas is sized.
 * @returns {function(): void} Stops following the pane.
 */
function followPane(root, renderer) {
  const canvas = renderer.domElement;
  const fit = () => {
    // A collapsed canvas lets the pane report the room its layout gives it, not the room the
    // old canvas props open, so the canvas shrinks with the window as well as growing.
    canvas.style.height = "0px";
    const size = canvasSize(measurePane(root, canvas));
    renderer.setSize(size.width, size.height, false);
    canvas.style.height = `${size.height}px`;
  };
  fit();
  let frame = 0;
  // Fitting on the next frame keeps a resize inside the callback from looping the observer.
  const observer = new ResizeObserver(() => { frame = requestAnimationFrame(fit); });
  observer.observe(root);
  return () => {
    observer.disconnect();
    cancelAnimationFrame(frame);
  };
}

/**
 * Add a canvas sized to its pane, orbit controls on the camera, and a frame loop drawing scene.
 *
 * @param {HTMLElement} root Pane to append the canvas and then the slab controls to.
 * @param {THREE.Scene} scene Scene to draw every frame.
 * @param {THREE.PerspectiveCamera} camera Camera to orbit, possibly shared with other panes.
 * @param {{center: number[]}} bounds Bounds of the points, whose center the controls orbit.
 * @param {HTMLElement} sliders Slab controls, placed before the canvas is measured.
 * @returns {{renderer: THREE.WebGLRenderer, stop: function(): void}} The running renderer and
 *   a function that stops its loop, controls, and pane resizing.
 */
function startRenderer(root, scene, camera, bounds, sliders) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  if (typeof devicePixelRatio === "number") {
    renderer.setPixelRatio(devicePixelRatio);
  }
  const canvas = renderer.domElement;
  // A zero height canvas lets a pane sized by its content report no room before the first fit.
  Object.assign(canvas.style,
    { display: "block", width: "100%", height: "0px", boxSizing: "border-box" });
  root.append(canvas, sliders);
  const unfollow = followPane(root, renderer);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(...bounds.center);
  controls.update();
  // A frame loop keeps panes that share one camera in step when another pane orbits it.
  renderer.setAnimationLoop(() => {
    camera.aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1) || 1;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  });
  const stop = () => {
    renderer.setAnimationLoop(null);
    unfollow();
    controls.dispose();
  };
  return { renderer, stop };
}

/**
 * Render a sparse3d or points layer as a rotatable point cloud with slab sliders.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} record Decoded layer record whose positions are N*3 in stored column order.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} state View state. state.camera is created when missing and shared by
 *   panes given the same state.camera. state.slab is filled and rewritten by the sliders.
 *   Optional mode and onPick(positionIndex).
 * @returns {{setColors: function(Uint8Array, string=): void,
 *   updateColors: function(Uint8Array): void, dispose: function(): void}} setColors recolors
 *   the points on the same canvas and sets the legend caption when one is given.
 *   updateColors recolors only. dispose stops rendering and frees the WebGL context.
 * @throws {Error} When the positions are not 3 dimensional, the layer is empty, or the
 *   layer kind has no 3D axis order.
 */
export function renderPoints(root, record, colors, state) {
  const { n, dim, kind } = record.header;
  if (dim !== 3) {
    throw new Error(`renderPoints needs 3 dimensional positions, this layer has dim ${dim}`);
  }
  const axes = screenAxes(kind);
  const positions = toScreen(record.positions, n, axes.columns);
  resetRoot(root, state);
  const bounds = boundsOf(positions, n);
  state.slab = clampSlab(state.slab, bounds);
  state.camera ??= makeCamera(bounds);
  let current = colors;
  const maskNow = () => slabMask(positions, n, state.slab.axis, state.slab.center,
    state.slab.thickness);
  let mask = maskNow();
  const points = makePoints(positions, rgbaColors(current, mask, n), bounds.radius);
  const scene = new THREE.Scene();
  scene.add(points);
  const recolor = () => {
    points.geometry.getAttribute("color").array.set(rgbaColors(current, mask, n));
    points.geometry.getAttribute("color").needsUpdate = true;
  };
  const sliders = slabControls(state, bounds, axes.names, () => { mask = maskNow(); recolor(); });
  const { renderer, stop } = startRenderer(root, scene, state.camera, bounds, sliders);
  onStillClick(renderer.domElement, (event) => {
    const { origin, direction } = rayUnder(event, renderer.domElement, state.camera);
    const index = nearestOnRay(positions, n, origin, direction, mask);
    if (index >= 0 && state.onPick) {
      state.onPick(index);
    }
  });
  const setColors = (next, caption) => {
    current = next;
    recolor();
    if (caption !== undefined) {
      showCaption(root, caption);
    }
  };
  return {
    setColors,
    updateColors: (next) => setColors(next),
    dispose() {
      stop();
      points.geometry.dispose();
      points.material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
