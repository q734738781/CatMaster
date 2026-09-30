import assert from "node:assert/strict";
import test from "node:test";

import {
  alignSites,
  cartesianToFractional,
  crystallographicViewDirection,
  displacement,
  formulaFromSites,
  fractionalToCartesian,
  latticeFromParameters,
  latticeParameters,
  measurement,
  minimumImageDisplacement,
  selectByRadius,
  selectByElement,
  siteSpeciesLabel,
  unwrapSites,
} from "./structureModel.js";

function periodicFixture() {
  const matrix = latticeFromParameters([5.1, 6.2, 7.3, 72, 81, 67]);
  const fractional = [
    [0.95, 0.2, 0.3],
    [0.05, 0.2, 0.3],
    [0.25, 0.4, 0.7],
  ];
  return {
    lattice: { matrix, pbc: [true, true, true] },
    sites: fractional.map((abc, index) => ({
      species: [{ element: index ? "O" : "Si", occu: 1 }],
      abc,
      xyz: fractionalToCartesian(matrix, abc),
      properties: {},
    })),
  };
}

test("triclinic cell parameters and coordinate conversions round trip", () => {
  const parameters = [5.1, 6.2, 7.3, 72, 81, 67];
  const matrix = latticeFromParameters(parameters);
  latticeParameters(matrix).forEach((value, index) => assert.ok(Math.abs(value - parameters[index]) < 1e-9));
  const fractional = [0.17, 0.41, 0.83];
  const restored = cartesianToFractional(matrix, fractionalToCartesian(matrix, fractional));
  restored.forEach((value, index) => assert.ok(Math.abs(value - fractional[index]) < 1e-10));
});

test("[uvw] and (hkl) camera directions follow direct and reciprocal triclinic bases", () => {
  const structure = periodicFixture();
  const uvw = crystallographicViewDirection(structure, [1, 0, 0], "uvw");
  const directA = structure.lattice.matrix[0];
  const directANorm = Math.hypot(...directA);
  uvw.forEach((value, axis) => assert.ok(Math.abs(value - directA[axis] / directANorm) < 1e-10));

  const hkl = crystallographicViewDirection(structure, [1, 0, 0], "hkl");
  assert.ok(Math.abs(hkl.reduce((sum, value, axis) => sum + value * structure.lattice.matrix[1][axis], 0)) < 1e-10);
  assert.ok(Math.abs(hkl.reduce((sum, value, axis) => sum + value * structure.lattice.matrix[2][axis], 0)) < 1e-10);
  assert.ok(hkl.reduce((sum, value, axis) => sum + value * directA[axis], 0) > 0);
});

test("periodic radius selection, MIC, and unwrap operate on base atoms", () => {
  const structure = periodicFixture();
  assert.equal(Math.hypot(...displacement(structure, 0, 1, false)) > 3, true);
  assert.equal(Math.hypot(...displacement(structure, 0, 1, true)) < 1, true);
  assert.deepEqual(selectByRadius(structure, 0, 1, true), [0, 1]);
  const unwrapped = unwrapSites(structure, [0, 1]);
  assert.equal(Math.hypot(...unwrapped.sites[1].xyz.map((value, axis) => value - unwrapped.sites[0].xyz[axis])) < 1, true);
});

test("triclinic MIC finds the true closest lattice image instead of rounding each axis", () => {
  const matrix = latticeFromParameters([4.2, 5.1, 6.3, 78, 83, 72]);
  const delta = [-0.2021076, 0.5078919, -0.2316460];
  const exact = minimumImageDisplacement(matrix, delta);
  assert.ok(Math.abs(Math.hypot(...exact) - 2.632655988) < 1e-9);

  const componentWrapped = fractionalToCartesian(
    matrix,
    delta.map((value) => value - Math.round(value)),
  );
  assert.ok(Math.abs(Math.hypot(...componentWrapped) - 3.505821440) < 1e-9);

  const structure = {
    lattice: { matrix, pbc: [true, true, true] },
    sites: [
      { species: [{ element: "Si", occu: 1 }], abc: [0, 0, 0], xyz: [0, 0, 0], properties: {} },
      {
        species: [{ element: "O", occu: 1 }],
        abc: delta,
        xyz: fractionalToCartesian(matrix, delta),
        properties: {},
      },
    ],
  };
  assert.ok(Math.abs(Math.hypot(...displacement(structure, 0, 1, true)) - 2.632655988) < 1e-9);
  assert.deepEqual(selectByRadius(structure, 0, 3.0, true), [0, 1]);
  assert.ok(Math.abs(measurement(structure, [0, 1], "distance", true).value - 2.632655988) < 1e-9);
});

test("degenerate angle and dihedral selections are explicitly not measurable", () => {
  const structure = {
    sites: [
      { xyz: [0, 0, 0] },
      { xyz: [0, 0, 0] },
      { xyz: [1, 0, 0] },
      { xyz: [2, 0, 0] },
    ],
  };
  assert.equal(measurement(structure, [0, 1, 2], "angle"), null);
  assert.equal(measurement(structure, [0, 1, 2, 3], "dihedral"), null);

  const collinear = {
    sites: [
      { xyz: [-1, 0, 0] },
      { xyz: [0, 0, 0] },
      { xyz: [1, 0, 0] },
      { xyz: [2, 1, 0] },
    ],
  };
  assert.equal(measurement(collinear, [0, 1, 2, 3], "dihedral"), null);
});

test("disordered sites preserve occupancies in labels, formula, and element selection", () => {
  const structure = {
    sites: [
      {
        species: [
          { element: "Na", occu: 0.5 },
          { element: "K", occu: 0.5 },
        ],
      },
      { species: [{ element: "Cl", occu: 1 }] },
    ],
  };
  assert.equal(siteSpeciesLabel(structure.sites[0]), "Na 0.5 + K 0.5");
  assert.equal(formulaFromSites(structure), "Na0.5 K0.5 Cl");
  assert.deepEqual(selectByElement(structure, "K"), [0]);
  assert.deepEqual(selectByElement(structure, "Na Cl"), [0, 1]);
});

test("pair and plane alignment use the requested direction", () => {
  const structure = periodicFixture();
  const pair = alignSites(structure, [0, 1], "pair", [0, 0, 1]);
  const pairVector = pair.sites[1].xyz.map((value, axis) => value - pair.sites[0].xyz[axis]);
  assert.ok(Math.abs(pairVector[0]) < 1e-9);
  assert.ok(Math.abs(pairVector[1]) < 1e-9);
  assert.ok(pairVector[2] > 0);

  const plane = alignSites(structure, [0, 1, 2], "plane", [1, 0, 0]);
  const left = plane.sites[1].xyz.map((value, axis) => value - plane.sites[0].xyz[axis]);
  const right = plane.sites[2].xyz.map((value, axis) => value - plane.sites[0].xyz[axis]);
  const normal = [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
  assert.ok(normal[0] > 0);
  assert.ok(Math.abs(normal[1]) < 1e-8);
  assert.ok(Math.abs(normal[2]) < 1e-8);
});

test("50k base-atom radius selection stays below the interaction gate", () => {
  const sites = Array.from({ length: 50_000 }, (_, index) => ({
    species: [{ element: "H", occu: 1 }],
    abc: [(index % 100) / 100, (Math.floor(index / 100) % 100) / 100, Math.floor(index / 10_000) / 5],
    xyz: [index % 100, Math.floor(index / 100) % 100, Math.floor(index / 10_000)],
    properties: {},
  }));
  const structure = {
    lattice: { matrix: [[100, 0, 0], [0, 100, 0], [0, 0, 5]], pbc: [true, true, true] },
    sites,
  };
  const started = performance.now();
  const selected = selectByRadius(structure, 0, 1.01, true);
  const elapsed = performance.now() - started;
  assert.ok(selected.length > 1);
  assert.ok(elapsed < 200, `50k selection took ${elapsed.toFixed(1)} ms`);
});
