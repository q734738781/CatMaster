export function cloneValue(value) {
  return value == null ? value : structuredClone(value);
}

export function matrixVector(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, item, index) => sum + Number(item) * Number(vector[index]), 0));
}

export function invertMatrix3(matrix) {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix.map((row) => row.map(Number));
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(determinant) < 1e-12) throw new Error("The cell matrix is singular.");
  return [
    [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
    [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
    [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
  ];
}

export function latticeMatrix(structure) {
  const matrix = structure?.lattice?.matrix;
  return Array.isArray(matrix) && matrix.length === 3 ? matrix.map((row) => row.map(Number)) : null;
}

export function latticeParameters(matrix) {
  const vectors = matrix.map((row) => row.map(Number));
  const lengths = vectors.map((vector) => Math.hypot(...vector));
  const angle = (left, right) => {
    const cosine = Math.max(-1, Math.min(1, left.reduce((sum, value, index) => sum + value * right[index], 0)
      / (Math.hypot(...left) * Math.hypot(...right))));
    return Math.acos(cosine) * 180 / Math.PI;
  };
  return [
    ...lengths,
    angle(vectors[1], vectors[2]),
    angle(vectors[0], vectors[2]),
    angle(vectors[0], vectors[1]),
  ];
}

export function latticeFromParameters(parameters) {
  const [a, b, c, alphaDegrees, betaDegrees, gammaDegrees] = parameters.map(Number);
  if (![a, b, c].every((value) => value > 0)) throw new Error("Cell lengths must be positive.");
  const [alpha, beta, gamma] = [alphaDegrees, betaDegrees, gammaDegrees].map((value) => value * Math.PI / 180);
  if (![alpha, beta, gamma].every((value) => value > 0 && value < Math.PI)) {
    throw new Error("Cell angles must be between 0° and 180°.");
  }
  const sinGamma = Math.sin(gamma);
  if (Math.abs(sinGamma) < 1e-9) throw new Error("Gamma makes the cell singular.");
  const cx = c * Math.cos(beta);
  const cy = c * (Math.cos(alpha) - Math.cos(beta) * Math.cos(gamma)) / sinGamma;
  const czSquared = c * c - cx * cx - cy * cy;
  if (czSquared <= 1e-12) throw new Error("These lengths and angles do not form a valid cell.");
  return [
    [a, 0, 0],
    [b * Math.cos(gamma), b * sinGamma, 0],
    [cx, cy, Math.sqrt(czSquared)],
  ];
}

export function crystallographicViewDirection(structure, indices, kind = "uvw") {
  const matrix = latticeMatrix(structure);
  if (!matrix) throw new Error("Crystallographic views require a periodic lattice.");
  const vector = (indices || []).map(Number);
  if (vector.length !== 3 || vector.some((value) => !Number.isFinite(value))) {
    throw new Error("Enter exactly three finite crystallographic indices.");
  }
  let direction;
  if (kind === "hkl") {
    const volume = dot(matrix[0], cross(matrix[1], matrix[2]));
    if (Math.abs(volume) < 1e-12) throw new Error("The cell matrix is singular.");
    const reciprocal = [
      cross(matrix[1], matrix[2]).map((value) => value / volume),
      cross(matrix[2], matrix[0]).map((value) => value / volume),
      cross(matrix[0], matrix[1]).map((value) => value / volume),
    ];
    direction = [0, 1, 2].map((axis) => reciprocal.reduce(
      (sum, basis, basisIndex) => sum + vector[basisIndex] * basis[axis],
      0,
    ));
  } else {
    direction = [0, 1, 2].map((axis) => matrix.reduce(
      (sum, basis, basisIndex) => sum + vector[basisIndex] * basis[axis],
      0,
    ));
  }
  return normalize(direction);
}

export function siteSpecies(site) {
  const species = Array.isArray(site?.species) ? site.species : [];
  return species
    .map((item) => ({
      element: String(item?.element || "").trim(),
      occupancy: Number(item?.occu ?? 1),
    }))
    .filter((item) => item.element && Number.isFinite(item.occupancy) && item.occupancy > 0);
}

export function siteElement(site) {
  const species = siteSpecies(site);
  const dominant = species.reduce(
    (best, item) => item.occupancy > (best?.occupancy ?? -1) ? item : best,
    null,
  );
  return String(dominant?.element || site?.label || "X");
}

function compactCount(value) {
  if (Math.abs(value - Math.round(value)) < 1e-9) return String(Math.round(value));
  return Number(value.toFixed(4)).toString();
}

export function siteSpeciesLabel(site) {
  const species = siteSpecies(site);
  if (!species.length) return siteElement(site);
  if (species.length === 1 && Math.abs(species[0].occupancy - 1) < 1e-9) return species[0].element;
  return species.map((item) => `${item.element} ${compactCount(item.occupancy)}`).join(" + ");
}

export function formulaFromSites(structure) {
  const counts = new Map();
  for (const site of structure?.sites || []) {
    for (const item of siteSpecies(site)) {
      counts.set(item.element, (counts.get(item.element) || 0) + item.occupancy);
    }
  }
  return [...counts.entries()]
    .map(([element, count]) => `${element}${Math.abs(count - 1) < 1e-9 ? "" : compactCount(count)}`)
    .join(" ") || "Empty structure";
}

export function fractionalToCartesian(matrix, fractional) {
  // Pymatgen stores lattice vectors as rows: cart = frac @ matrix.
  return [
    fractional.reduce((sum, value, index) => sum + Number(value) * Number(matrix[index][0]), 0),
    fractional.reduce((sum, value, index) => sum + Number(value) * Number(matrix[index][1]), 0),
    fractional.reduce((sum, value, index) => sum + Number(value) * Number(matrix[index][2]), 0),
  ];
}

export function cartesianToFractional(matrix, cartesian) {
  const inverse = invertMatrix3(matrix);
  return [
    cartesian.reduce((sum, value, index) => sum + Number(value) * Number(inverse[index][0]), 0),
    cartesian.reduce((sum, value, index) => sum + Number(value) * Number(inverse[index][1]), 0),
    cartesian.reduce((sum, value, index) => sum + Number(value) * Number(inverse[index][2]), 0),
  ];
}

function minimumImageContext(matrix) {
  const numeric = matrix.map((row) => row.map(Number));
  const inverse = invertMatrix3(numeric);
  const scale = Math.max(...numeric.map((row) => norm(row)), 1);
  return {
    matrix: numeric,
    orthogonal: (
      Math.abs(dot(numeric[0], numeric[1])) <= 1e-12 * scale * scale
      && Math.abs(dot(numeric[0], numeric[2])) <= 1e-12 * scale * scale
      && Math.abs(dot(numeric[1], numeric[2])) <= 1e-12 * scale * scale
    ),
    // If ||x @ A|| <= r, then |x_i| <= r ||column_i(A^-1)||.  These
    // bounds make the finite image search below exact for any nonsingular
    // triclinic cell, including cells where rounding each fractional
    // coordinate independently does not find the closest image.
    inverseColumnNorms: [0, 1, 2].map((column) => Math.hypot(
      inverse[0][column],
      inverse[1][column],
      inverse[2][column],
    )),
  };
}

function closestImageWithContext(context, fractionalDelta) {
  const reduced = fractionalDelta.map((value) => {
    const numeric = Number(value);
    return numeric - Math.round(numeric);
  });
  let best = fractionalToCartesian(context.matrix, reduced);
  let bestSquared = dot(best, best);
  if (context.orthogonal) return best;
  if (bestSquared <= 1e-28) return [0, 0, 0];

  const radius = Math.sqrt(bestSquared);
  const ranges = reduced.map((value, axis) => {
    const bound = radius * context.inverseColumnNorms[axis];
    const tolerance = 1e-12 * Math.max(1, Math.abs(value), bound);
    return [
      Math.ceil(value - bound - tolerance),
      Math.floor(value + bound + tolerance),
    ];
  });

  for (let i = ranges[0][0]; i <= ranges[0][1]; i += 1) {
    for (let j = ranges[1][0]; j <= ranges[1][1]; j += 1) {
      for (let k = ranges[2][0]; k <= ranges[2][1]; k += 1) {
        if (i === 0 && j === 0 && k === 0) continue;
        const candidate = fractionalToCartesian(
          context.matrix,
          [reduced[0] - i, reduced[1] - j, reduced[2] - k],
        );
        const squared = dot(candidate, candidate);
        if (squared < bestSquared) {
          best = candidate;
          bestSquared = squared;
        }
      }
    }
  }
  return best;
}

export function minimumImageDisplacement(matrix, fractionalDelta) {
  return closestImageWithContext(minimumImageContext(matrix), fractionalDelta);
}

function updatedSite(site, vector, coordinateMode, matrix) {
  const next = cloneValue(site);
  const values = vector.map(Number);
  if (matrix && coordinateMode === "fractional") {
    next.abc = values;
    next.xyz = fractionalToCartesian(matrix, values);
  } else if (matrix) {
    next.xyz = values;
    next.abc = cartesianToFractional(matrix, values);
  } else {
    next.xyz = values;
    next.abc = values;
  }
  return next;
}

export function setSiteCoordinate(structure, index, vector, coordinateMode = "cartesian") {
  const next = cloneValue(structure);
  if (!next?.sites?.[index]) return next;
  next.sites[index] = updatedSite(next.sites[index], vector, coordinateMode, latticeMatrix(next));
  return next;
}

export function translateSites(structure, indices, vector) {
  let next = cloneValue(structure);
  for (const index of indices) {
    const site = next?.sites?.[index];
    if (!site) continue;
    next = setSiteCoordinate(next, index, site.xyz.map((value, axis) => Number(value) + Number(vector[axis])), "cartesian");
  }
  return next;
}

function normalize(vector) {
  const length = Math.hypot(...vector);
  if (length < 1e-12) throw new Error("Rotation axis needs a non-zero direction.");
  return vector.map((value) => Number(value) / length);
}

export function rotateSites(structure, indices, degrees, axis, originMode = "centroid") {
  const next = cloneValue(structure);
  const selected = indices.map((index) => next?.sites?.[index]).filter(Boolean);
  if (!selected.length) return next;
  const origin = Array.isArray(originMode)
    ? originMode.map(Number)
    : originMode === "world"
      ? [0, 0, 0]
      : [0, 1, 2].map((axisIndex) => selected.reduce((sum, site) => sum + Number(site.xyz[axisIndex]), 0) / selected.length);
  const [ux, uy, uz] = normalize(axis);
  const angle = Number(degrees) * Math.PI / 180;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const rotation = [
    [cosine + ux * ux * (1 - cosine), ux * uy * (1 - cosine) - uz * sine, ux * uz * (1 - cosine) + uy * sine],
    [uy * ux * (1 - cosine) + uz * sine, cosine + uy * uy * (1 - cosine), uy * uz * (1 - cosine) - ux * sine],
    [uz * ux * (1 - cosine) - uy * sine, uz * uy * (1 - cosine) + ux * sine, cosine + uz * uz * (1 - cosine)],
  ];
  let transformed = next;
  for (const index of indices) {
    const site = transformed?.sites?.[index];
    if (!site) continue;
    const relative = site.xyz.map((value, coordinate) => Number(value) - origin[coordinate]);
    const rotated = matrixVector(rotation, relative).map((value, coordinate) => value + origin[coordinate]);
    transformed = setSiteCoordinate(transformed, index, rotated, "cartesian");
  }
  return transformed;
}

export function alignSites(structure, indices, kind, targetDirection) {
  const minimum = kind === "plane" ? 3 : 2;
  if (indices.length < minimum) throw new Error(`Select at least ${minimum} atoms to align a ${kind}.`);
  const points = indices.slice(0, minimum).map((index) => structure?.sites?.[index]?.xyz);
  if (points.some((point) => !point)) throw new Error("The selected alignment atoms are unavailable.");
  const source = kind === "plane"
    ? cross(subtract(points[1], points[0]), subtract(points[2], points[0]))
    : subtract(points[1], points[0]);
  const from = normalize(source);
  const to = normalize(targetDirection);
  let axis = cross(from, to);
  const cosine = Math.max(-1, Math.min(1, dot(from, to)));
  if (norm(axis) < 1e-10) {
    if (cosine > 0.999999) return cloneValue(structure);
    const helper = Math.abs(from[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0];
    axis = cross(from, helper);
  }
  const degrees = Math.acos(cosine) * 180 / Math.PI;
  return rotateSites(structure, indices, degrees, axis, points[0]);
}

export function wrapSites(structure, indices = []) {
  const matrix = latticeMatrix(structure);
  if (!matrix) return cloneValue(structure);
  const targets = indices.length ? new Set(indices) : null;
  const next = cloneValue(structure);
  next.sites = next.sites.map((site, index) => {
    if (targets && !targets.has(index)) return site;
    const fractional = site.abc.map((value) => ((Number(value) % 1) + 1) % 1);
    return updatedSite(site, fractional, "fractional", matrix);
  });
  return next;
}

export function unwrapSites(structure, indices = []) {
  const matrix = latticeMatrix(structure);
  const targets = indices.length ? indices : structure?.sites?.map((_, index) => index) || [];
  if (!matrix || targets.length < 2) return cloneValue(structure);
  const context = minimumImageContext(matrix);
  const anchor = structure.sites[targets[0]];
  let next = cloneValue(structure);
  for (const index of targets.slice(1)) {
    const site = structure.sites[index];
    const delta = site.abc.map((value, axis) => Number(value) - Number(anchor.abc[axis]));
    const closest = closestImageWithContext(context, delta);
    const xyz = anchor.xyz.map((value, axis) => Number(value) + closest[axis]);
    next = setSiteCoordinate(next, index, xyz, "cartesian");
  }
  return next;
}

export function centerSites(structure, indices = []) {
  const targets = indices.length ? indices : structure?.sites?.map((_, index) => index) || [];
  if (!targets.length) return cloneValue(structure);
  const matrix = latticeMatrix(structure);
  const selected = targets.map((index) => structure.sites[index]).filter(Boolean);
  const center = [0, 1, 2].map((axis) => selected.reduce((sum, site) => sum + Number(site.xyz[axis]), 0) / selected.length);
  const target = matrix
    ? fractionalToCartesian(matrix, [0.5, 0.5, 0.5])
    : [0, 0, 0];
  return translateSites(structure, targets, target.map((value, axis) => value - center[axis]));
}

export function deleteSites(structure, indices) {
  const removed = new Set(indices);
  const next = cloneValue(structure);
  const indexMap = new Map();
  next.sites = next.sites.filter((_, index) => {
    if (removed.has(index)) return false;
    indexMap.set(index, indexMap.size);
    return true;
  });
  const bonds = next?.properties?.bonds;
  if (Array.isArray(bonds)) {
    next.properties.bonds = bonds
      .filter((bond) => !removed.has(bond.site_idx_1) && !removed.has(bond.site_idx_2))
      .map((bond) => ({
        ...bond,
        site_idx_1: indexMap.get(bond.site_idx_1),
        site_idx_2: indexMap.get(bond.site_idx_2),
      }));
  }
  return next;
}

export function addSite(structure, element, vector, coordinateMode = "fractional") {
  const symbol = String(element || "").trim();
  if (!/^[A-Z][a-z]?$/.test(symbol)) throw new Error("Enter a valid element symbol such as C, Fe, or Pt.");
  const next = cloneValue(structure);
  const matrix = latticeMatrix(next);
  const site = {
    species: [{ element: symbol, occu: 1, oxidation_state: 0 }],
    label: symbol,
    properties: matrix ? { selective_dynamics: [true, true, true] } : {},
    abc: [0, 0, 0],
    xyz: [0, 0, 0],
  };
  next.sites = [...(next.sites || []), updatedSite(site, vector, coordinateMode, matrix)];
  return next;
}

export function duplicateSites(structure, indices, offset = [0.3, 0.3, 0.3]) {
  const next = cloneValue(structure);
  const matrix = latticeMatrix(next);
  for (const index of indices) {
    const site = next.sites[index];
    if (!site) continue;
    const duplicate = updatedSite(
      site,
      site.xyz.map((value, axis) => Number(value) + Number(offset[axis])),
      "cartesian",
      matrix,
    );
    if (duplicate.properties) delete duplicate.properties._catmaster_mol_atom_index;
    next.sites.push(duplicate);
  }
  return next;
}

export function replaceSpecies(structure, indices, element) {
  const symbol = String(element || "").trim();
  if (!/^[A-Z][a-z]?$/.test(symbol)) throw new Error("Enter a valid element symbol such as C, Fe, or Pt.");
  const next = cloneValue(structure);
  for (const index of indices) {
    const site = next?.sites?.[index];
    if (!site) continue;
    site.species = [{ element: symbol, occu: 1, oxidation_state: 0 }];
    site.label = symbol;
  }
  return next;
}

export function setMobility(structure, indices, mobility) {
  const next = cloneValue(structure);
  for (const [index, site] of next.sites.entries()) {
    site.properties = { ...(site.properties || {}) };
    const current = Array.isArray(site.properties.selective_dynamics)
      ? site.properties.selective_dynamics.map(Boolean)
      : [true, true, true];
    if (indices.includes(index)) site.properties.selective_dynamics = mobility.map(Boolean);
    else site.properties.selective_dynamics = current;
  }
  return next;
}

export function parseIndexSelection(expression, atomCount) {
  const selected = new Set();
  for (const token of String(expression || "").split(/[\s,;]+/).filter(Boolean)) {
    const range = token.match(/^(\d+)-(\d+)$/);
    if (range) {
      const start = Number(range[1]);
      const end = Number(range[2]);
      for (let value = Math.min(start, end); value <= Math.max(start, end); value += 1) {
        if (value >= 1 && value <= atomCount) selected.add(value - 1);
      }
      continue;
    }
    const value = Number(token);
    if (Number.isInteger(value) && value >= 1 && value <= atomCount) selected.add(value - 1);
  }
  return [...selected].sort((left, right) => left - right);
}

export function selectByElement(structure, query) {
  const wanted = new Set(String(query || "").split(/[\s,;]+/).filter(Boolean).map((item) => item.toLowerCase()));
  return (structure?.sites || []).flatMap((site, index) => (
    siteSpecies(site).some((item) => wanted.has(item.element.toLowerCase())) ? [index] : []
  ));
}

export function selectByLayer(structure, axis, center, tolerance) {
  const coordinate = Math.max(0, Math.min(2, Number(axis)));
  return (structure?.sites || []).flatMap((site, index) => (
    Math.abs(Number(site.xyz?.[coordinate]) - Number(center)) <= Number(tolerance) ? [index] : []
  ));
}

export function selectByRadius(structure, seed, radius, useMic = true) {
  if (!Number.isInteger(seed) || !structure?.sites?.[seed]) return [];
  const limit = Number(radius);
  if (!Number.isFinite(limit) || limit < 0) return [];
  const selected = [];
  const limitSquared = limit * limit;
  const origin = structure.sites[seed];
  const matrix = useMic ? latticeMatrix(structure) : null;
  const micContext = matrix ? minimumImageContext(matrix) : null;
  for (let index = 0; index < structure.sites.length; index += 1) {
    const site = structure.sites[index];
    let x;
    let y;
    let z;
    if (micContext) {
      [x, y, z] = closestImageWithContext(
        micContext,
        site.abc.map((value, axis) => Number(value) - Number(origin.abc[axis])),
      );
    } else {
      x = Number(site.xyz[0]) - Number(origin.xyz[0]);
      y = Number(site.xyz[1]) - Number(origin.xyz[1]);
      z = Number(site.xyz[2]) - Number(origin.xyz[2]);
    }
    if (x * x + y * y + z * z <= limitSquared) selected.push(index);
  }
  return selected;
}

export function explicitBonds(structure) {
  return Array.isArray(structure?.properties?.bonds) ? structure.properties.bonds : [];
}

export function connectedFragment(structure, seed) {
  if (!Number.isInteger(seed) || seed < 0) return [];
  const adjacency = new Map();
  for (const bond of explicitBonds(structure)) {
    const left = Number(bond.site_idx_1);
    const right = Number(bond.site_idx_2);
    adjacency.set(left, [...(adjacency.get(left) || []), right]);
    adjacency.set(right, [...(adjacency.get(right) || []), left]);
  }
  const visited = new Set([seed]);
  const queue = [seed];
  while (queue.length) {
    for (const neighbor of adjacency.get(queue.shift()) || []) {
      if (visited.has(neighbor)) continue;
      visited.add(neighbor);
      queue.push(neighbor);
    }
  }
  return [...visited].sort((left, right) => left - right);
}

function subtract(left, right) {
  return left.map((value, index) => Number(value) - Number(right[index]));
}

function dot(left, right) {
  return left.reduce((sum, value, index) => sum + Number(value) * Number(right[index]), 0);
}

function cross(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function norm(vector) {
  return Math.hypot(...vector);
}

export function displacement(structure, leftIndex, rightIndex, useMic = false) {
  const left = structure?.sites?.[leftIndex];
  const right = structure?.sites?.[rightIndex];
  if (!left || !right) return null;
  const matrix = latticeMatrix(structure);
  if (useMic && matrix) {
    return minimumImageDisplacement(matrix, subtract(right.abc, left.abc));
  }
  return subtract(right.xyz, left.xyz);
}

export function measurement(structure, indices, mode = "distance", useMic = false) {
  if (mode.startsWith("cell_")) {
    const axis = { cell_a: 0, cell_b: 1, cell_c: 2 }[mode];
    const vector = latticeMatrix(structure)?.[axis];
    return vector ? { label: `${mode.slice(-1)} cell vector`, value: norm(vector), unit: "Å" } : null;
  }
  if (mode === "coordination" && indices.length >= 1) {
    const cutoff = 3;
    const count = structure.sites.reduce((total, _, index) => (
      index !== indices[0] && norm(displacement(structure, indices[0], index, useMic)) <= cutoff ? total + 1 : total
    ), 0);
    return { label: "Coordination within 3 Å", value: count, unit: "neighbors" };
  }
  if (mode === "distance" && indices.length >= 2) {
    return { label: "Distance", value: norm(displacement(structure, indices[0], indices[1], useMic)), unit: "Å" };
  }
  if (mode === "angle" && indices.length >= 3) {
    const left = displacement(structure, indices[1], indices[0], useMic);
    const right = displacement(structure, indices[1], indices[2], useMic);
    const leftLength = norm(left);
    const rightLength = norm(right);
    if (leftLength < 1e-12 || rightLength < 1e-12) return null;
    const cosine = Math.max(-1, Math.min(1, dot(left, right) / (leftLength * rightLength)));
    return { label: "Angle", value: Math.acos(cosine) * 180 / Math.PI, unit: "°" };
  }
  if (mode === "dihedral" && indices.length >= 4) {
    const b0 = displacement(structure, indices[1], indices[0], useMic);
    const b1 = displacement(structure, indices[1], indices[2], useMic);
    const b2 = displacement(structure, indices[2], indices[3], useMic);
    if (norm(b0) < 1e-12 || norm(b1) < 1e-12 || norm(b2) < 1e-12) return null;
    const b1n = normalize(b1);
    const v = subtract(b0, b1n.map((value) => dot(b0, b1n) * value));
    const w = subtract(b2, b1n.map((value) => dot(b2, b1n) * value));
    if (norm(v) < 1e-12 || norm(w) < 1e-12) return null;
    const angle = Math.atan2(dot(cross(b1n, v), w), dot(v, w)) * 180 / Math.PI;
    return { label: "Dihedral", value: angle, unit: "°" };
  }
  return null;
}

export function canonicalPeriodicSnapshot(originalSnapshot, viewerStructure) {
  const base = cloneValue(originalSnapshot);
  const original = base?.payload?.pymatgen || {};
  base.payload = {
    pymatgen: {
      ...original,
      lattice: cloneValue(viewerStructure.lattice),
      sites: cloneValue(viewerStructure.sites || []),
    },
  };
  return base;
}

export function viewerFromTrajectoryFrame(frame) {
  const sites = (frame?.symbols || []).map((element, index) => ({
    species: [{ element, occu: 1, oxidation_state: 0 }],
    xyz: cloneValue(frame.positions[index]),
    abc: frame?.pbc?.some(Boolean)
      ? cartesianToFractional(frame.cell, frame.positions[index])
      : cloneValue(frame.positions[index]),
    label: element,
    properties: {},
  }));
  if (frame?.pbc?.some(Boolean)) {
    return {
      sites,
      lattice: {
        matrix: cloneValue(frame.cell),
        pbc: cloneValue(frame.pbc),
      },
    };
  }
  return { sites };
}

export function defaultSavePath(path, mode) {
  const source = String(path || "structure");
  const slash = source.lastIndexOf("/");
  const directory = slash >= 0 ? source.slice(0, slash + 1) : "files/";
  const filename = slash >= 0 ? source.slice(slash + 1) : source;
  if (/^(POSCAR|CONTCAR)$/i.test(filename)) return `${directory}${filename}_edited`;
  const dot = filename.lastIndexOf(".");
  const stem = dot > 0 ? filename.slice(0, dot) : filename;
  const suffix = dot > 0 ? filename.slice(dot) : mode === "molecule" ? ".sdf" : ".vasp";
  return `${directory}${stem}_edited${suffix}`;
}
