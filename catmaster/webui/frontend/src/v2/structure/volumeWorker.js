export const MAX_VOLUME_BYTES = 512 * 1024 * 1024;
export const MAX_SOURCE_GRID_POINTS = 50_000_000;
export const MAX_INTERACTIVE_GRID_POINTS = 500_000;

const BOHR_TO_ANGSTROM = 0.529177249;
const textDecoder = new TextDecoder();
const atomicNumbers = [
  "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
  "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I", "Xe",
];

function elementFromToken(token) {
  const value = String(token || "");
  if (/^[A-Z][a-z]?$/.test(value)) return value;
  return atomicNumbers[Number(value)] || "X";
}

class ByteScanner {
  constructor(buffer) {
    this.bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    this.position = 0;
  }

  get done() {
    return this.position >= this.bytes.length;
  }

  readLine() {
    if (this.done) return null;
    const start = this.position;
    while (
      this.position < this.bytes.length
      && this.bytes[this.position] !== 10
      && this.bytes[this.position] !== 13
    ) this.position += 1;
    const line = textDecoder.decode(this.bytes.subarray(start, this.position));
    if (this.bytes[this.position] === 13) this.position += 1;
    if (this.bytes[this.position] === 10) this.position += 1;
    return line;
  }

  skipWhitespace() {
    while (this.position < this.bytes.length && this.bytes[this.position] <= 32) {
      this.position += 1;
    }
  }

  readNumber() {
    this.skipWhitespace();
    if (this.done) return null;
    let sign = 1;
    if (this.bytes[this.position] === 45) {
      sign = -1;
      this.position += 1;
    } else if (this.bytes[this.position] === 43) {
      this.position += 1;
    }
    let mantissa = 0;
    let decimalPlaces = 0;
    let sawDigit = false;
    let afterDecimal = false;
    while (this.position < this.bytes.length) {
      const byte = this.bytes[this.position];
      if (byte >= 48 && byte <= 57) {
        sawDigit = true;
        mantissa = mantissa * 10 + byte - 48;
        if (afterDecimal) decimalPlaces += 1;
        this.position += 1;
      } else if (byte === 46 && !afterDecimal) {
        afterDecimal = true;
        this.position += 1;
      } else {
        break;
      }
    }
    if (!sawDigit) {
      while (this.position < this.bytes.length && this.bytes[this.position] > 32) {
        this.position += 1;
      }
      return Number.NaN;
    }
    let exponent = -decimalPlaces;
    const exponentMarker = this.bytes[this.position];
    if (exponentMarker === 69 || exponentMarker === 101 || exponentMarker === 68 || exponentMarker === 100) {
      this.position += 1;
      let exponentSign = 1;
      if (this.bytes[this.position] === 45) {
        exponentSign = -1;
        this.position += 1;
      } else if (this.bytes[this.position] === 43) {
        this.position += 1;
      }
      let exponentValue = 0;
      let exponentDigits = 0;
      while (this.position < this.bytes.length) {
        const byte = this.bytes[this.position];
        if (byte < 48 || byte > 57) break;
        exponentValue = exponentValue * 10 + byte - 48;
        exponentDigits += 1;
        this.position += 1;
      }
      if (!exponentDigits) return Number.NaN;
      exponent += exponentSign * exponentValue;
    }
    while (this.position < this.bytes.length && this.bytes[this.position] > 32) {
      this.position += 1;
    }
    return sign * mantissa * (10 ** exponent);
  }
}

function numbers(line) {
  return String(line || "").trim().split(/\s+/).filter(Boolean).map(Number);
}

function multiply(vector, scalar) {
  return vector.map((value) => Number(value) * scalar);
}

function determinant(matrix) {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

function fractionalToCartesian(matrix, fractional) {
  return [0, 1, 2].map((axis) => fractional.reduce(
    (sum, value, vector) => sum + Number(value) * Number(matrix[vector][axis]),
    0,
  ));
}

function cartesianToFractional(matrix, cartesian) {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
  const det = determinant(matrix);
  if (Math.abs(det) < 1e-12) throw new Error("The volume lattice is singular.");
  const inverse = [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
  return [
    cartesian.reduce((sum, value, index) => sum + value * inverse[index][0], 0),
    cartesian.reduce((sum, value, index) => sum + value * inverse[index][1], 0),
    cartesian.reduce((sum, value, index) => sum + value * inverse[index][2], 0),
  ];
}

function makeSite(element, xyz, abc, index) {
  return {
    species: [{ element, occu: 1, oxidation_state: 0 }],
    xyz,
    abc,
    label: `${element}${index + 1}`,
    properties: {},
  };
}

function targetGridDimensions(dimensions, maxPoints = MAX_INTERACTIVE_GRID_POINTS) {
  const sourceTotal = dimensions.reduce((product, value) => product * value, 1);
  if (!Number.isSafeInteger(sourceTotal) || sourceTotal <= 0 || sourceTotal > MAX_SOURCE_GRID_POINTS) {
    throw new Error(`The source grid has ${sourceTotal.toLocaleString()} points; the safe limit is ${MAX_SOURCE_GRID_POINTS.toLocaleString()}.`);
  }
  if (sourceTotal <= maxPoints) return { dimensions: [...dimensions], factor: 1 };
  let factor = Math.max(2, Math.ceil(Math.cbrt(sourceTotal / maxPoints)));
  const reduced = () => dimensions.map(
    (value) => value === 1 ? 1 : Math.max(2, Math.ceil(value / factor)),
  );
  let output = reduced();
  while (output.reduce((product, value) => product * value, 1) > maxPoints) {
    factor += 1;
    output = reduced();
  }
  return { dimensions: output, factor };
}

function dataRange(grid) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  for (const plane of grid) {
    for (const row of plane) {
      for (const value of row) {
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
        count += 1;
      }
    }
  }
  if (!count) return { min: 0, max: 0, abs_max: 0, mean: 0 };
  return { min, max, abs_max: Math.max(Math.abs(min), Math.abs(max)), mean: sum / count };
}

function readBoundedGrid(scanner, sourceDimensions, dataOrder, divisor = 1) {
  const [nx, ny, nz] = sourceDimensions;
  const sourceTotal = nx * ny * nz;
  const { dimensions: outputDimensions, factor } = targetGridDimensions(sourceDimensions);
  const [ox, oy, oz] = outputDimensions;
  const sums = new Float64Array(ox * oy * oz);
  const counts = new Uint32Array(sums.length);
  for (let flat = 0; flat < sourceTotal; flat += 1) {
    const value = scanner.readNumber();
    if (!Number.isFinite(value)) {
      throw new Error(`The scalar grid ended after ${flat.toLocaleString()} of ${sourceTotal.toLocaleString()} values.`);
    }
    let x;
    let y;
    let z;
    if (dataOrder === "x_fastest") {
      x = flat % nx;
      y = Math.floor(flat / nx) % ny;
      z = Math.floor(flat / (nx * ny));
    } else {
      z = flat % nz;
      y = Math.floor(flat / nz) % ny;
      x = Math.floor(flat / (ny * nz));
    }
    const tx = Math.min(ox - 1, Math.floor((x * ox) / nx));
    const ty = Math.min(oy - 1, Math.floor((y * oy) / ny));
    const tz = Math.min(oz - 1, Math.floor((z * oz) / nz));
    const target = (tx * oy + ty) * oz + tz;
    sums[target] += value / divisor;
    counts[target] += 1;
  }
  const grid = Array.from({ length: ox }, (_, x) => Array.from(
    { length: oy },
    (_, y) => {
      const row = new Float32Array(oz);
      for (let z = 0; z < oz; z += 1) {
        const flat = (x * oy + y) * oz + z;
        row[z] = counts[flat] ? sums[flat] / counts[flat] : 0;
      }
      return row;
    },
  ));
  return {
    grid,
    grid_dims: outputDimensions,
    source_grid_dims: [...sourceDimensions],
    downsample_factor: factor,
    data_range: dataRange(grid),
  };
}

function readRequiredLine(scanner, message) {
  const line = scanner.readLine();
  if (line === null) throw new Error(message);
  return line;
}

function nextNonemptyLine(scanner) {
  while (!scanner.done) {
    const line = scanner.readLine();
    if (line?.trim()) return line;
  }
  return null;
}

function parseCube(buffer) {
  const scanner = new ByteScanner(buffer);
  readRequiredLine(scanner, "The CUBE file is empty.");
  readRequiredLine(scanner, "The CUBE file is missing its second title line.");
  const atomHeader = numbers(readRequiredLine(scanner, "The CUBE atom header is missing."));
  if (atomHeader.length < 4) throw new Error("The CUBE atom header is malformed.");
  const atomCount = Math.abs(Math.trunc(atomHeader[0]));
  const hasOrbitalHeader = atomHeader[0] < 0;
  const rawOrigin = atomHeader.slice(1, 4);
  const axes = [0, 1, 2].map(
    () => numbers(readRequiredLine(scanner, "The CUBE voxel header is incomplete.")),
  );
  if (axes.some((axis) => axis.length < 4 || !Number.isInteger(Math.abs(axis[0])))) {
    throw new Error("The CUBE voxel header is malformed.");
  }
  const sourceDimensions = axes.map((axis) => Math.abs(Math.trunc(axis[0])));
  const scale = axes[0][0] > 0 ? BOHR_TO_ANGSTROM : 1;
  const voxelVectors = axes.map((axis) => multiply(axis.slice(1, 4), scale));
  const lattice = voxelVectors.map((vector, axis) => multiply(vector, sourceDimensions[axis]));
  const origin = multiply(rawOrigin, scale);
  const sites = [];
  for (let index = 0; index < atomCount; index += 1) {
    const atom = numbers(readRequiredLine(scanner, "The CUBE atom list ended early."));
    if (atom.length < 5) throw new Error(`CUBE atom ${index + 1} is malformed.`);
    const rawXyz = multiply(atom.slice(2, 5), scale);
    const xyz = rawXyz.map((value, axis) => value - origin[axis]);
    sites.push(makeSite(elementFromToken(atom[0]), xyz, cartesianToFractional(lattice, xyz), index));
  }
  if (hasOrbitalHeader) readRequiredLine(scanner, "The CUBE orbital header is missing.");
  const reduced = readBoundedGrid(scanner, sourceDimensions, "z_fastest");
  return {
    structure: {
      sites,
      lattice: {
        matrix: lattice,
        pbc: Math.hypot(...origin) < 1e-6 ? [true, true, true] : [false, false, false],
      },
    },
    volumes: [{
      ...reduced,
      lattice,
      origin,
      data_order: "z_fastest",
      periodic: Math.hypot(...origin) < 1e-6,
      label: "volumetric data",
      source: "CUBE",
    }],
  };
}

function matchingGridLine(line, expected = null) {
  const values = numbers(line);
  if (
    values.length !== 3
    || values.some((value) => !Number.isInteger(value) || value <= 0)
  ) return null;
  if (expected && values.some((value, axis) => value !== expected[axis])) return null;
  return values;
}

function findNextGrid(scanner, expected = null) {
  while (!scanner.done) {
    const line = scanner.readLine();
    const dimensions = matchingGridLine(line, expected);
    if (dimensions) return dimensions;
  }
  return null;
}

function parseChgcar(buffer) {
  const scanner = new ByteScanner(buffer);
  readRequiredLine(scanner, "The CHGCAR title is missing.");
  const scale = Number(readRequiredLine(scanner, "The CHGCAR scale is missing.").trim());
  if (!Number.isFinite(scale) || Math.abs(scale) < 1e-15) throw new Error("The CHGCAR scale is invalid.");
  const lattice = [0, 1, 2].map(
    () => multiply(numbers(readRequiredLine(scanner, "The CHGCAR lattice is incomplete.")).slice(0, 3), scale),
  );
  let symbolsOrCounts = readRequiredLine(scanner, "The CHGCAR species header is missing.").trim().split(/\s+/);
  let symbols;
  let counts;
  if (symbolsOrCounts.every((value) => Number.isInteger(Number(value)))) {
    counts = symbolsOrCounts.map(Number);
    symbols = counts.map((_, index) => atomicNumbers[index + 1] || "X");
  } else {
    symbols = symbolsOrCounts.map((value) => elementFromToken(value.split(/[_/]/)[0]));
    counts = numbers(readRequiredLine(scanner, "The CHGCAR atom counts are missing."));
  }
  let coordinateMode = readRequiredLine(scanner, "The CHGCAR coordinate mode is missing.");
  if (coordinateMode.trim().toUpperCase().startsWith("S")) {
    coordinateMode = readRequiredLine(scanner, "The CHGCAR coordinate mode is missing.");
  }
  const direct = coordinateMode.trim().toUpperCase().startsWith("D");
  const sites = [];
  for (let species = 0; species < counts.length; species += 1) {
    for (let offset = 0; offset < counts[species]; offset += 1) {
      const coordinates = numbers(readRequiredLine(scanner, "The CHGCAR atom list ended early.")).slice(0, 3);
      const xyz = direct ? fractionalToCartesian(lattice, coordinates) : multiply(coordinates, scale);
      const abc = direct ? coordinates : cartesianToFractional(lattice, xyz);
      sites.push(makeSite(symbols[species] || "X", xyz, abc, sites.length));
    }
  }
  const firstDimensions = findNextGrid(scanner);
  if (!firstDimensions) throw new Error("No scalar grid was found in the CHGCAR file.");
  const cellVolume = Math.abs(determinant(lattice)) || 1;
  const volumes = [];
  let dimensions = firstDimensions;
  for (let volumeIndex = 0; volumeIndex < 2 && dimensions; volumeIndex += 1) {
    const reduced = readBoundedGrid(scanner, dimensions, "x_fastest", cellVolume);
    volumes.push({
      ...reduced,
      lattice,
      origin: [0, 0, 0],
      data_order: "x_fastest",
      periodic: true,
      label: volumeIndex === 0 ? "charge density" : "magnetization density",
      source: "CHGCAR",
    });
    dimensions = findNextGrid(scanner, firstDimensions);
  }
  return {
    structure: { sites, lattice: { matrix: lattice, pbc: [true, true, true] } },
    volumes,
  };
}

function parseXsf(buffer) {
  const scanner = new ByteScanner(buffer);
  let lattice = null;
  const sites = [];
  const volumes = [];
  while (!scanner.done) {
    const rawLine = scanner.readLine();
    if (rawLine === null) break;
    const header = rawLine.trim().toUpperCase();
    if (header === "PRIMVEC") {
      lattice = [0, 1, 2].map(
        () => numbers(readRequiredLine(scanner, "The XSF PRIMVEC block is incomplete.")).slice(0, 3),
      );
    } else if (header === "PRIMCOORD") {
      const count = Math.trunc(numbers(readRequiredLine(scanner, "The XSF PRIMCOORD count is missing."))[0] || 0);
      for (let index = 0; index < count; index += 1) {
        const tokens = readRequiredLine(scanner, "The XSF atom list ended early.").trim().split(/\s+/);
        const xyz = tokens.slice(1, 4).map(Number);
        sites.push(makeSite(elementFromToken(tokens[0]), xyz, xyz, sites.length));
      }
    } else if (header.startsWith("BEGIN_DATAGRID_3D")) {
      const label = rawLine.trim().replace(/^BEGIN_DATAGRID_3D_?/i, "") || `volume ${volumes.length + 1}`;
      const dimensions = matchingGridLine(readRequiredLine(scanner, "The XSF grid dimensions are missing."));
      if (!dimensions) throw new Error("The XSF grid dimensions are invalid.");
      const origin = numbers(readRequiredLine(scanner, "The XSF grid origin is missing.")).slice(0, 3);
      const gridLattice = [0, 1, 2].map(
        () => numbers(readRequiredLine(scanner, "The XSF grid lattice is incomplete.")).slice(0, 3),
      );
      const reduced = readBoundedGrid(scanner, dimensions, "x_fastest");
      volumes.push({
        ...reduced,
        lattice: gridLattice,
        origin,
        data_order: "x_fastest",
        periodic: true,
        label,
        source: "XSF",
      });
    }
  }
  if (!volumes.length) throw new Error("No BEGIN_DATAGRID_3D block was found in this XSF file.");
  const activeLattice = lattice || volumes[0].lattice;
  for (const site of sites) site.abc = cartesianToFractional(activeLattice, site.xyz);
  return {
    structure: { sites, lattice: { matrix: activeLattice, pbc: [true, true, true] } },
    volumes,
  };
}

function formatFrom(filename, buffer) {
  const lower = String(filename || "").toLowerCase().replace(/\.(gz|bz2|xz)$/i, "");
  if (lower.endsWith(".xsf")) return "xsf";
  if (lower.endsWith(".cube") || lower.endsWith(".cub")) return "cube";
  if (/(^|[/_.-])(chgcar|parchg|locpot|elfcar|aeccar)([/_.-]|$)/i.test(lower)) return "chgcar";
  const prefix = textDecoder.decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 64 * 1024)));
  if (/BEGIN_DATAGRID_3D/i.test(prefix)) return "xsf";
  const lines = prefix.split(/\r?\n/, 8);
  if (lines.length >= 5 && numbers(lines[2]).length === 4 && numbers(lines[3]).length === 4) return "cube";
  return "chgcar";
}

export function parseVolumeBuffer(buffer, filename = "") {
  if (!(buffer instanceof ArrayBuffer)) throw new TypeError("Volume parsing requires an ArrayBuffer.");
  if (buffer.byteLength > MAX_VOLUME_BYTES) {
    throw new Error(`The scalar-field file exceeds the ${(MAX_VOLUME_BYTES / 1024 / 1024).toFixed(0)} MB safety limit.`);
  }
  const format = formatFrom(filename, buffer);
  if (format === "xsf") return parseXsf(buffer);
  if (format === "cube") return parseCube(buffer);
  return parseChgcar(buffer);
}

function abortError() {
  return new DOMException("Volume loading was cancelled.", "AbortError");
}

export async function readResponseBuffer(response, {
  signal,
  onProgress = () => {},
  maxBytes = MAX_VOLUME_BYTES,
} = {}) {
  if (!response.ok) throw new Error(`Volume download failed (${response.status}).`);
  const declared = Number(response.headers.get("content-length") || 0);
  if (declared > maxBytes) {
    throw new Error(`The scalar-field file exceeds the ${(maxBytes / 1024 / 1024).toFixed(0)} MB safety limit.`);
  }
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength > maxBytes) throw new Error("The scalar-field file exceeds the safe size limit.");
    onProgress(buffer.byteLength, declared);
    return buffer;
  }
  const reader = response.body.getReader();
  let output = declared > 0 ? new Uint8Array(declared) : new Uint8Array(1024 * 1024);
  let loaded = 0;
  try {
    while (true) {
      if (signal?.aborted) throw abortError();
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > maxBytes) {
        throw new Error(`The scalar-field stream exceeds the ${(maxBytes / 1024 / 1024).toFixed(0)} MB safety limit.`);
      }
      if (loaded + value.byteLength > output.byteLength) {
        let capacity = output.byteLength;
        while (capacity < loaded + value.byteLength) capacity = Math.min(maxBytes, capacity * 2);
        const grown = new Uint8Array(capacity);
        grown.set(output.subarray(0, loaded));
        output = grown;
      }
      output.set(value, loaded);
      loaded += value.byteLength;
      onProgress(loaded, declared);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (signal?.aborted) throw abortError();
  return loaded === output.byteLength ? output.buffer : output.buffer.slice(0, loaded);
}

let activeController = null;
const workerScope = typeof self === "undefined" ? null : self;
if (workerScope) {
  workerScope.onmessage = async (event) => {
    if (event.data?.type === "cancel") {
      activeController?.abort();
      return;
    }
    if (event.data?.type !== "load") return;
    activeController?.abort();
    const controller = new AbortController();
    activeController = controller;
    try {
      const response = await fetch(event.data.url, {
        credentials: "same-origin",
        signal: controller.signal,
      });
      const buffer = await readResponseBuffer(response, {
        signal: controller.signal,
        onProgress: (loaded, total) => workerScope.postMessage({ type: "progress", loaded, total }),
      });
      workerScope.postMessage({ type: "stage", message: "Parsing and averaging the scalar field off the main thread…" });
      const parsed = parseVolumeBuffer(buffer, String(event.data.filename || ""));
      workerScope.postMessage({ type: "result", ...parsed });
    } catch (error) {
      workerScope.postMessage({
        type: error?.name === "AbortError" ? "cancelled" : "error",
        message: error?.message || String(error),
      });
    } finally {
      if (activeController === controller) activeController = null;
    }
  };
}
