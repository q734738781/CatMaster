// MatterViz still performs lattice, symmetry, and mesh work on every displayed
// site.  These tiers keep the renderer bounded without turning a 50k-site
// structure into an uninformative handful of atoms.
export const LOD_RENDER_TIERS = Object.freeze([
  Object.freeze({ upTo: 1_000, budget: 1_000 }),
  Object.freeze({ upTo: 10_000, budget: 2_500 }),
  Object.freeze({ upTo: 50_000, budget: 5_000 }),
  Object.freeze({ upTo: Number.POSITIVE_INFINITY, budget: 6_000 }),
]);

export const MAX_RENDER_SITES = 6_000;
const MAX_NEIGHBORHOOD_SEEDS = 32;
const NEIGHBORS_PER_SELECTED_SITE = 24;

function siteElements(site) {
  const species = Array.isArray(site?.species) ? site.species : [];
  const elements = species.map((item) => String(item?.element || "").trim()).filter(Boolean);
  return elements.length ? elements : [String(site?.label || "?")];
}

function validUniqueIndices(values, count) {
  return [...new Set((values || []).map(Number))]
    .filter((value) => Number.isInteger(value) && value >= 0 && value < count);
}

function projectBonds(bonds, baseToDisplay) {
  return (Array.isArray(bonds) ? bonds : []).flatMap((bond) => {
    const left = baseToDisplay.get(Number(bond?.site_idx_1));
    const right = baseToDisplay.get(Number(bond?.site_idx_2));
    if (left === undefined || right === undefined) return [];
    return [{ ...bond, site_idx_1: left, site_idx_2: right }];
  });
}

function renderBudget(total, requestedBudget) {
  if (Number.isFinite(requestedBudget)) {
    return Math.max(1, Math.min(total, Math.floor(requestedBudget)));
  }
  return Math.min(
    total,
    LOD_RENDER_TIERS.find((tier) => total <= tier.upTo)?.budget || MAX_RENDER_SITES,
  );
}

function sitePosition(site) {
  const preferred = Array.isArray(site?.abc) && site.abc.length === 3 ? site.abc : site?.xyz;
  if (!Array.isArray(preferred) || preferred.length !== 3) return null;
  const values = preferred.map(Number);
  return values.every(Number.isFinite) ? values : null;
}

function distanceSquared(left, right) {
  return (left[0] - right[0]) ** 2
    + (left[1] - right[1]) ** 2
    + (left[2] - right[2]) ** 2;
}

function addSelectedNeighborhoods(chosen, sites, selected, budget) {
  for (const selectedIndex of selected.slice(0, MAX_NEIGHBORHOOD_SEEDS)) {
    if (chosen.size >= budget) return;
    const center = sitePosition(sites[selectedIndex]);
    if (!center) continue;
    const nearest = [];
    for (let index = 0; index < sites.length; index += 1) {
      if (index === selectedIndex || chosen.has(index)) continue;
      const position = sitePosition(sites[index]);
      if (!position) continue;
      const entry = { index, distance: distanceSquared(center, position) };
      if (nearest.length < NEIGHBORS_PER_SELECTED_SITE) {
        nearest.push(entry);
        nearest.sort((left, right) => right.distance - left.distance || right.index - left.index);
      } else if (
        entry.distance < nearest[0].distance
        || (entry.distance === nearest[0].distance && entry.index < nearest[0].index)
      ) {
        nearest[0] = entry;
        nearest.sort((left, right) => right.distance - left.distance || right.index - left.index);
      }
    }
    nearest
      .sort((left, right) => left.distance - right.distance || left.index - right.index)
      .forEach(({ index }) => {
        if (chosen.size < budget) chosen.add(index);
      });
  }
}

function normalizedPositions(sites) {
  const positions = sites.map(sitePosition);
  const bounds = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
  for (const position of positions) {
    if (!position) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      bounds[axis][0] = Math.min(bounds[axis][0], position[axis]);
      bounds[axis][1] = Math.max(bounds[axis][1], position[axis]);
    }
  }
  return positions.map((position) => {
    if (!position) return null;
    return position.map((value, axis) => {
      const width = bounds[axis][1] - bounds[axis][0];
      return width > 1e-12 ? (value - bounds[axis][0]) / width : 0.5;
    });
  });
}

function addSpatialLevel(chosen, positions, side, budget) {
  const represented = new Set();
  const keyFor = (position) => position?.map(
    (value) => Math.min(side - 1, Math.max(0, Math.floor(value * side))),
  ).join(":");

  for (const index of chosen) {
    const key = keyFor(positions[index]);
    if (key) represented.add(key);
  }
  for (let index = 0; index < positions.length && chosen.size < budget; index += 1) {
    const key = keyFor(positions[index]);
    if (!key || represented.has(key)) continue;
    represented.add(key);
    chosen.add(index);
  }
}

function addStratifiedRemainder(chosen, total, budget) {
  const missing = budget - chosen.size;
  if (missing <= 0) return;
  // Two interleaved phases avoid a prefix bias when sites are grouped by
  // species or layer in the source file.
  for (let slot = 0; slot < missing * 2 && chosen.size < budget; slot += 1) {
    chosen.add(Math.min(total - 1, Math.floor(((slot + 0.5) * total) / (missing * 2))));
  }
  for (let index = 0; index < total && chosen.size < budget; index += 1) chosen.add(index);
}

export function buildLargeStructureView(
  structure,
  bonds,
  selection = [],
  requestedBudget,
) {
  const sites = Array.isArray(structure?.sites) ? structure.sites : [];
  const total = sites.length;
  const budget = renderBudget(total, requestedBudget);
  if (total <= budget) {
    return {
      structure,
      bonds,
      selection: validUniqueIndices(selection, total),
      displayToBase: null,
      isLod: false,
      total,
      visible: total,
    };
  }

  const selected = validUniqueIndices(selection, total);
  const chosen = new Set(selected.slice(0, budget));
  addSelectedNeighborhoods(chosen, sites, selected, budget);

  // Every represented species gets a deterministic exemplar before density
  // sampling, so dilute dopants and adsorbates are not erased by LOD.
  const seenElements = new Set();
  for (let index = 0; index < sites.length && chosen.size < budget; index += 1) {
    const elements = siteElements(sites[index]);
    if (elements.some((element) => !seenElements.has(element))) {
      elements.forEach((element) => seenElements.add(element));
      chosen.add(index);
    }
  }

  // Coarse-to-fine voxel passes form an auditable density hierarchy: global
  // shape first, progressively denser spatial coverage second.
  const positions = normalizedPositions(sites);
  for (const side of [4, 8, 16, 32, 64]) {
    if (chosen.size >= budget) break;
    addSpatialLevel(chosen, positions, side, budget);
  }
  addStratifiedRemainder(chosen, total, budget);

  const displayToBase = [...chosen].slice(0, budget).sort((left, right) => left - right);
  const baseToDisplay = new Map(displayToBase.map((baseIndex, displayIndex) => [baseIndex, displayIndex]));
  const sourceBonds = Array.isArray(bonds) ? bonds : structure?.properties?.bonds;
  const displayBonds = projectBonds(sourceBonds, baseToDisplay);
  const properties = { ...(structure?.properties || {}) };
  if (Array.isArray(properties.bonds)) properties.bonds = displayBonds;

  return {
    structure: {
      ...structure,
      properties,
      sites: displayToBase.map((index) => sites[index]),
    },
    bonds: displayBonds,
    selection: selected.flatMap((baseIndex) => {
      const displayIndex = baseToDisplay.get(baseIndex);
      return displayIndex === undefined ? [] : [displayIndex];
    }),
    displayToBase,
    isLod: true,
    total,
    visible: displayToBase.length,
  };
}

export function displayIndicesToBase(indices, displayToBase) {
  if (!displayToBase) return validUniqueIndices(indices, Number.MAX_SAFE_INTEGER);
  return validUniqueIndices(indices, displayToBase.length).map((index) => displayToBase[index]);
}
