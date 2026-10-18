/**
 * Decode the binary layer and channel files of a layerviz bundle into typed arrays.
 *
 * Purpose:
 * - Turn a layer file into the record every view reads: header, positions, rgb, scores,
 *   and rgbJoint.
 * - Turn a channel file into its header and uint8 values, and decode inline base64 bundles.
 *
 * Special Notes:
 * - Framing, little endian: uint32 header length H, H bytes of UTF-8 JSON, then sections at
 *   offsets counted from the end of the header. bundle.py writes the same layout.
 * - Pure ES module with no DOM access, so Node tests it directly.
 * - float16 decoding is a local copy, so the module does not depend on similarity.js.
 */

const BYTES_PER_ELEMENT = { float32: 4, uint8: 1, float16: 2 };

/**
 * Split a framed file into its parsed JSON header and the byte offset where data starts.
 *
 * @param {ArrayBuffer} buffer Whole layer or channel file.
 * @returns {{buffer: ArrayBuffer, header: object, dataStart: number}} The buffer, its
 *   header object, and the first data byte.
 * @throws {Error} When the header length or the header JSON is unreadable.
 */
function readFrame(buffer) {
  if (buffer.byteLength < 4) {
    throw new Error(`header length needs 4 bytes, buffer has ${buffer.byteLength}`);
  }
  const headerLength = new DataView(buffer).getUint32(0, true);
  if (4 + headerLength > buffer.byteLength) {
    throw new Error(
      `header length ${headerLength} runs past the buffer end at ${buffer.byteLength} bytes`,
    );
  }
  const json = new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength));
  return { buffer, header: parseHeader(json), dataStart: 4 + headerLength };
}

/**
 * Parse the JSON header text of a framed file.
 *
 * @param {string} json UTF-8 decoded header text.
 * @returns {object} The header object.
 * @throws {Error} When the text is not valid JSON, with the parser message attached.
 */
function parseHeader(json) {
  try {
    return JSON.parse(json);
  } catch (error) {
    throw new Error(`header is not valid JSON: ${error.message}`, { cause: error });
  }
}

/**
 * Copy one named section out of a framed file after checking its dtype, size and bounds.
 *
 * @param {{buffer: ArrayBuffer, header: object, dataStart: number}} frame From readFrame.
 * @param {string} name Section name in header.sections.
 * @param {number} count Number of elements the section must hold.
 * @param {string} dtype The dtype this section must have: float32, uint8 or float16.
 * @returns {ArrayBuffer} A fresh, aligned copy of the section bytes.
 * @throws {Error} When the section is missing, has another dtype, the wrong size, or runs
 *   past the buffer end.
 */
function sectionBytes(frame, name, count, dtype) {
  const { buffer } = frame;
  const spec = frame.header.sections?.[name];
  if (!spec) {
    throw new Error(`section ${name} is missing from the header`);
  }
  const [offset, nbytes, actual] = spec;
  if (actual !== dtype) {
    throw new Error(`section ${name} has dtype ${actual}, expected ${dtype}`);
  }
  const expected = count * BYTES_PER_ELEMENT[dtype];
  if (nbytes !== expected) {
    throw new Error(`section ${name} has ${nbytes} bytes, its shape needs ${expected}`);
  }
  const start = frame.dataStart + offset;
  if (start + nbytes > buffer.byteLength) {
    throw new Error(`section ${name} ends at byte ${start + nbytes}, past ${buffer.byteLength}`);
  }
  // Copying keeps typed array views aligned whatever the header length is.
  // Typed array views of the copy assume a little endian host, true of every target platform.
  return buffer.slice(start, start + nbytes);
}

/**
 * Convert one IEEE 754 half precision bit pattern to a JavaScript number.
 *
 * @param {number} bits Unsigned 16 bit pattern.
 * @returns {number} The decoded value, including subnormals, signed zero, Infinity and NaN.
 */
function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    return sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return mantissa ? NaN : sign * Infinity;
  }
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/**
 * Decode a buffer of float16 values into a Float32Array.
 *
 * @param {ArrayBuffer} bytes Little endian float16 data.
 * @returns {Float32Array} One float per half value.
 */
function decodeFloat16(bytes) {
  const halves = new Uint16Array(bytes);
  const out = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i += 1) {
    out[i] = halfToFloat(halves[i]);
  }
  return out;
}

/**
 * Decode a layer file into the record shared by the page and every view.
 *
 * @param {ArrayBuffer} buffer Whole layer file.
 * @returns {{header: object, positions: Float32Array, rgb: Uint8Array,
 *   scores: Float32Array, rgbJoint: Uint8Array|null}} Positions are N*dim, rgb and
 *   rgbJoint N*3, scores N*k decoded from float16.
 * @throws {Error} When the framing or any section does not match the header.
 */
export function parseLayer(buffer) {
  const frame = readFrame(buffer);
  const { n, dim, k } = frame.header;
  const hasJoint = Boolean(frame.header.sections?.rgb_joint);
  return {
    header: frame.header,
    positions: new Float32Array(sectionBytes(frame, "positions", n * dim, "float32")),
    rgb: new Uint8Array(sectionBytes(frame, "rgb", n * 3, "uint8")),
    scores: decodeFloat16(sectionBytes(frame, "scores", n * k, "float16")),
    rgbJoint: hasJoint ? new Uint8Array(sectionBytes(frame, "rgb_joint", n * 3, "uint8")) : null,
  };
}

/**
 * Decode a channel file into its header and per point channel values.
 *
 * @param {ArrayBuffer} buffer Whole channel file.
 * @returns {{header: object, values: Uint8Array}} Values are N*C, header.channels lists
 *   the original channel indices.
 * @throws {Error} When the framing or the values section does not match the header.
 */
export function parseChannels(buffer) {
  const frame = readFrame(buffer);
  const { n, c } = frame.header;
  return {
    header: frame.header,
    values: new Uint8Array(sectionBytes(frame, "values", n * c, "uint8")),
  };
}

/**
 * Decode a base64 string from an inline bundle into an ArrayBuffer.
 *
 * @param {string} text Standard base64 text.
 * @returns {ArrayBuffer} The decoded bytes.
 */
export function base64ToBuffer(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}
