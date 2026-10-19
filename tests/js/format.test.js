/**
 * Test the layer and channel file decoder in format.js on hand built buffers.
 *
 * Purpose:
 * - Pin the binary framing that bundle.py must write and format.js must read.
 * - Check float16 decoding, the optional rgb_joint section, and framing errors.
 * - Check that bad header fields, sections, and base64 text raise errors naming the cause.
 *
 * Special Notes:
 * - Buffers are built byte by byte here, never by bundle.py, so the test fixes the layout.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  parseLayer, parseChannels, base64ToBuffer, decodeFloat16,
} from "../../layerviz/web/format.js";

const N = 4;
const DIM = 2;
const K = 3;
const POSITIONS = [0.5, -1.25, 2, 3.75, -4, 5.5, 6.125, -7];
const RGB = [255, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
// Each float16 bit pattern below decodes exactly to the value at the same index.
const SCORE_BITS = [
  0x0000, 0x3c00, 0xc000, 0x3800, 0x7bff, 0x0001,
  0x3400, 0x3e00, 0xba00, 0x4000, 0x4200, 0x7c00,
];
const SCORE_VALUES = [0, 1, -2, 0.5, 65504, 2 ** -24, 0.25, 1.5, -0.75, 2, 3, Infinity];
const RGB_JOINT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const CHANNEL_VALUES = [9, 8, 7, 6, 5, 4, 3, 2];

/**
 * Return the little endian bytes of a typed array.
 */
function bytesOf(typed) {
  return new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
}

/**
 * Round a byte count up to the next multiple of 4.
 */
function align4(count) {
  return Math.ceil(count / 4) * 4;
}

/**
 * Frame a header and its sections the way bundle.py writes a layer or channel file.
 *
 * Sections is a list of [name, dtype, bytes]. Offsets count from the end of the header
 * and each section starts at a multiple of 4 bytes, in list order.
 */
function buildFile(header, sections) {
  const specs = {};
  let offset = 0;
  for (const [name, dtype, bytes] of sections) {
    specs[name] = [offset, bytes.length, dtype];
    offset = align4(offset + bytes.length);
  }
  const json = new TextEncoder().encode(JSON.stringify({ ...header, sections: specs }));
  // Pad with spaces by UTF-8 byte count, not string length, so data starts 4 byte aligned.
  const headerBytes = new Uint8Array(align4(json.length + 4) - 4).fill(0x20);
  headerBytes.set(json);
  const out = new Uint8Array(4 + headerBytes.length + offset);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);
  for (const [name, , bytes] of sections) {
    out.set(bytes, 4 + headerBytes.length + specs[name][0]);
  }
  return out.buffer;
}

/**
 * Build the n = 4, dim = 2, k = 3 layer file, with rgb_joint when asked.
 *
 * Header fields in fields replace the defaults, and an undefined value drops the field.
 */
function buildLayer({ joint = false, fields = {} } = {}) {
  const header = {
    n: N, dim: DIM, k: K, kind: "points", grid: null, scale: [1, 1], labels: ["a", "b", "c", "dä"],
    ...fields,
  };
  const sections = [
    ["positions", "float32", bytesOf(new Float32Array(POSITIONS))],
    ["rgb", "uint8", new Uint8Array(RGB)],
    ["scores", "float16", bytesOf(new Uint16Array(SCORE_BITS))],
  ];
  if (joint) {
    sections.push(["rgb_joint", "uint8", new Uint8Array(RGB_JOINT)]);
  }
  return buildFile(header, sections);
}

/**
 * Build a channel file with n = 4 points and c = 2 channels.
 *
 * Header fields in fields replace the defaults, and sections replaces the values section.
 */
function buildChannels({
  fields = {}, sections = [["values", "uint8", new Uint8Array(CHANNEL_VALUES)]],
} = {}) {
  const header = { n: N, c: 2, channels: [5, 17], ...fields };
  return buildFile(header, sections);
}

/**
 * Frame raw header text with no sections after it.
 */
function frameHeaderText(text) {
  const json = new TextEncoder().encode(text);
  const out = new Uint8Array(4 + json.length);
  new DataView(out.buffer).setUint32(0, json.length, true);
  out.set(json, 4);
  return out.buffer;
}

const BAD_COUNTS = [["missing", undefined], ["negative", -1], ["fractional", 1.5], ["a string", "4"]];

test("parseLayer decodes positions as float32 in point order", () => {
  const record = parseLayer(buildLayer());
  assert.ok(record.positions instanceof Float32Array);
  assert.deepEqual(Array.from(record.positions), POSITIONS);
});

test("parseLayer decodes rgb as uint8 triples", () => {
  const record = parseLayer(buildLayer());
  assert.ok(record.rgb instanceof Uint8Array);
  assert.deepEqual(Array.from(record.rgb), RGB);
});

test("parseLayer decodes float16 scores into a Float32Array", () => {
  const record = parseLayer(buildLayer());
  assert.ok(record.scores instanceof Float32Array);
  assert.deepEqual(Array.from(record.scores), SCORE_VALUES);
});

test("parseLayer keeps every header field", () => {
  const record = parseLayer(buildLayer());
  assert.equal(record.header.n, N);
  assert.equal(record.header.dim, DIM);
  assert.equal(record.header.k, K);
  assert.equal(record.header.kind, "points");
  assert.equal(record.header.grid, null);
  assert.deepEqual(record.header.scale, [1, 1]);
  assert.deepEqual(record.header.labels, ["a", "b", "c", "dä"]);
});

test("parseLayer gives null rgbJoint when the section is missing", () => {
  assert.equal(parseLayer(buildLayer()).rgbJoint, null);
});

test("parseLayer decodes rgb_joint when the section is present", () => {
  const record = parseLayer(buildLayer({ joint: true }));
  assert.ok(record.rgbJoint instanceof Uint8Array);
  assert.deepEqual(Array.from(record.rgbJoint), RGB_JOINT);
});

test("parseLayer decodes float16 NaN and negative zero", () => {
  const header = { n: 1, dim: 1, k: 2, kind: "points", grid: null, scale: null, labels: null };
  const buffer = buildFile(header, [
    ["positions", "float32", bytesOf(new Float32Array([0]))],
    ["rgb", "uint8", new Uint8Array([0, 0, 0])],
    ["scores", "float16", bytesOf(new Uint16Array([0x7e00, 0x8000]))],
  ]);
  const scores = parseLayer(buffer).scores;
  assert.ok(Number.isNaN(scores[0]));
  assert.ok(Object.is(scores[1], -0));
});

test("parseChannels decodes the values and the channel list", () => {
  const result = parseChannels(buildChannels());
  assert.ok(result.values instanceof Uint8Array);
  assert.deepEqual(Array.from(result.values), CHANNEL_VALUES);
  assert.deepEqual(result.header.channels, [5, 17]);
  assert.equal(result.header.n, N);
  assert.equal(result.header.c, 2);
});

test("parseLayer rejects a header length past the end of the buffer", () => {
  const buffer = buildLayer();
  new DataView(buffer).setUint32(0, buffer.byteLength, true);
  assert.throws(() => parseLayer(buffer), /header length/);
});

test("parseLayer rejects a buffer too short to hold the header length", () => {
  assert.throws(() => parseLayer(new ArrayBuffer(2)), /header length/);
});

test("parseLayer rejects a section whose size does not match its shape", () => {
  const header = { n: N, dim: DIM, k: K, kind: "points", grid: null, scale: null, labels: null };
  const buffer = buildFile(header, [
    ["positions", "float32", bytesOf(new Float32Array(POSITIONS.slice(0, 6)))],
    ["rgb", "uint8", new Uint8Array(RGB)],
    ["scores", "float16", bytesOf(new Uint16Array(SCORE_BITS))],
  ]);
  assert.throws(() => parseLayer(buffer), /positions/);
});

test("parseLayer rejects a section whose dtype does not match its role", () => {
  const header = { n: N, dim: DIM, k: K, kind: "points", grid: null, scale: null, labels: null };
  const buffer = buildFile(header, [
    ["positions", "float16", bytesOf(new Uint16Array(SCORE_BITS.slice(0, 8)))],
    ["rgb", "uint8", new Uint8Array(RGB)],
    ["scores", "float16", bytesOf(new Uint16Array(SCORE_BITS))],
  ]);
  assert.throws(() => parseLayer(buffer), /section positions has dtype float16/);
});

test("parseLayer rejects a header without the rgb section", () => {
  const header = { n: N, dim: DIM, k: K, kind: "points", grid: null, scale: null, labels: null };
  const buffer = buildFile(header, [
    ["positions", "float32", bytesOf(new Float32Array(POSITIONS))],
    ["scores", "float16", bytesOf(new Uint16Array(SCORE_BITS))],
  ]);
  assert.throws(() => parseLayer(buffer), /section rgb is missing/);
});

test("parseLayer rejects a header that is not valid JSON", () => {
  assert.throws(() => parseLayer(frameHeaderText("{bad")), /header is not valid JSON/);
});

test("parseLayer rejects a header that is JSON null", () => {
  assert.throws(() => parseLayer(frameHeaderText("null")), {
    message: "header must be a JSON object, got null",
  });
});

for (const field of ["n", "dim", "k"]) {
  for (const [label, value] of BAD_COUNTS) {
    test(`parseLayer rejects a header whose ${field} is ${label}`, () => {
      const buffer = buildLayer({ fields: { [field]: value } });
      assert.throws(() => parseLayer(buffer), {
        message: `header field ${field} must be a non-negative integer, got ${JSON.stringify(value)}`,
      });
    });
  }
}

for (const [label, value] of [["missing", undefined], ["a number", 3], ["null", null]]) {
  test(`parseLayer rejects a header whose kind is ${label}`, () => {
    const buffer = buildLayer({ fields: { kind: value } });
    assert.throws(() => parseLayer(buffer), {
      message: `header field kind must be a string, got ${JSON.stringify(value)}`,
    });
  });
}

test("parseLayer names the section and the buffer end when a section is oversized", () => {
  const full = buildLayer();
  const cut = full.byteLength - 8;
  // Scores is the last section and fills the buffer to its end, so it ends at full.
  assert.throws(() => parseLayer(full.slice(0, cut)), {
    message: `section scores ends at byte ${full.byteLength}, past the buffer end at ${cut} bytes`,
  });
});

test("parseLayer decodes a layer with n = 0 into empty arrays", () => {
  const header = { n: 0, dim: 3, k: 2, kind: "points", grid: null, scale: null, labels: [] };
  const buffer = buildFile(header, [
    ["positions", "float32", new Uint8Array(0)],
    ["rgb", "uint8", new Uint8Array(0)],
    ["scores", "float16", new Uint8Array(0)],
  ]);
  const record = parseLayer(buffer);
  assert.deepEqual(record.positions, new Float32Array(0));
  assert.deepEqual(record.rgb, new Uint8Array(0));
  assert.deepEqual(record.scores, new Float32Array(0));
  assert.equal(record.rgbJoint, null);
});

test("parseLayer decodes a float16 scores section holding -Infinity", () => {
  const header = { n: 1, dim: 1, k: 2, kind: "points", grid: null, scale: null, labels: null };
  const buffer = buildFile(header, [
    ["positions", "float32", bytesOf(new Float32Array([0]))],
    ["rgb", "uint8", new Uint8Array([0, 0, 0])],
    ["scores", "float16", bytesOf(new Uint16Array([0xfc00, 0xbc00]))],
  ]);
  assert.deepEqual(parseLayer(buffer).scores, new Float32Array([-Infinity, -1]));
});

test("decodeFloat16 decodes little endian half values into a Float32Array", () => {
  const bytes = new Uint16Array([0x3c00, 0xfc00, 0x0001, 0xc000]).buffer;
  assert.deepEqual(decodeFloat16(bytes), new Float32Array([1, -Infinity, 2 ** -24, -2]));
});

for (const field of ["n", "c"]) {
  for (const [label, value] of BAD_COUNTS) {
    test(`parseChannels rejects a header whose ${field} is ${label}`, () => {
      const buffer = buildChannels({ fields: { [field]: value } });
      assert.throws(() => parseChannels(buffer), {
        message: `header field ${field} must be a non-negative integer, got ${JSON.stringify(value)}`,
      });
    });
  }
}

test("parseChannels rejects a file without the values section", () => {
  assert.throws(() => parseChannels(buildChannels({ sections: [] })), {
    message: "section values is missing from the header",
  });
});

test("parseChannels rejects a values section whose dtype is not uint8", () => {
  const sections = [["values", "float16", bytesOf(new Uint16Array([1, 2, 3, 4]))]];
  assert.throws(() => parseChannels(buildChannels({ sections })), {
    message: "section values has dtype float16, expected uint8",
  });
});

test("parseChannels rejects a values section whose size does not match n times c", () => {
  const sections = [["values", "uint8", new Uint8Array([1, 2, 3, 4, 5, 6])]];
  assert.throws(() => parseChannels(buildChannels({ sections })), {
    message: "section values has 6 bytes, its shape needs 8",
  });
});

test("parseChannels rejects a values section that runs past the buffer end", () => {
  const full = buildChannels();
  const cut = full.byteLength - 4;
  assert.throws(() => parseChannels(full.slice(0, cut)), {
    message: `section values ends at byte ${full.byteLength}, past the buffer end at ${cut} bytes`,
  });
});

test("base64ToBuffer returns the decoded bytes as an ArrayBuffer", () => {
  const buffer = base64ToBuffer("AQID+g==");
  assert.ok(buffer instanceof ArrayBuffer);
  assert.deepEqual(Array.from(new Uint8Array(buffer)), [1, 2, 3, 250]);
});

test("base64ToBuffer rejects text that is not valid base64", () => {
  assert.throws(() => base64ToBuffer("not base64!"), (error) => {
    assert.equal(error.name, "Error");
    assert.match(error.message, /^base64 text is not valid: /);
    assert.equal(error.cause.name, "InvalidCharacterError");
    assert.ok(error.message.endsWith(error.cause.message));
    return true;
  });
});
