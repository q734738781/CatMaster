import assert from "node:assert/strict";
import test from "node:test";

import {
  advanceUnwrappedTrajectory,
  createCanonicalDocument,
  createDisplayedTrajectoryFrame,
  createHistory,
  pushHistory,
  redoHistory,
  replaceHistoryPresent,
  undoHistory,
} from "./history.js";
import {
  addSite,
  deleteSites,
  replaceSpecies,
  setMobility,
  translateSites,
} from "./structureModel.js";

test("bounded structure history restores a mixed edit sequence", () => {
  const initialViewer = {
    lattice: { matrix: [[5, 0, 0], [0, 5, 0], [0, 0, 5]], pbc: [true, true, true] },
    sites: [{
      species: [{ element: "C", occu: 1 }],
      label: "C",
      abc: [0.2, 0.2, 0.2],
      xyz: [1, 1, 1],
      properties: { selective_dynamics: [true, true, true] },
    }],
  };
  const initial = createCanonicalDocument({
    snapshot: {
      mode: "periodic",
      format: "poscar",
      path: "files/POSCAR",
      source_version: { mtime_ns: 101, size: 202 },
      payload: { pymatgen: initialViewer },
    },
    viewer_structure: initialViewer,
    summary: {
      formula: "C",
      atom_count: 1,
      space_group: { symbol: "P1", number: 1 },
      symmetry_groups: [[0]],
    },
  });
  let history = createHistory(initial);
  for (let index = 0; index < 50; index += 1) {
    let next = history.present.viewer;
    if (index % 5 === 0) next = translateSites(next, [0], [0.05, 0, 0]);
    if (index % 5 === 1) next = addSite(next, "O", [0.4, 0.4, 0.4], "fractional");
    if (index % 5 === 2) next = replaceSpecies(next, [0], index % 10 === 2 ? "N" : "C");
    if (index % 5 === 3) next = setMobility(next, [0], index % 10 === 3 ? [false, false, false] : [true, true, true]);
    if (index % 5 === 4 && next.sites.length > 1) next = deleteSites(next, [next.sites.length - 1]);
    history = pushHistory(history, createCanonicalDocument(history.present, {
      snapshot: {
        ...history.present.snapshot,
        payload: { pymatgen: next },
      },
      viewer: next,
      summary: {
        formula: `step-${index}`,
        atom_count: next.sites.length,
        space_group: null,
        symmetry_groups: [],
      },
      symmetryBroken: true,
      version: history.present.version,
      modified: true,
    }));
  }
  const final = structuredClone(history.present);
  assert.equal(history.past.length, 50);
  for (let index = 0; index < 50; index += 1) history = undoHistory(history);
  assert.deepEqual(history.present, initial);
  for (let index = 0; index < 50; index += 1) history = redoHistory(history);
  assert.deepEqual(history.present, final);
});

test("candidate, Ketcher, and 3D edits share one canonical undo stack", () => {
  const initial = createCanonicalDocument({
    snapshot: {
      mode: "molecule",
      format: "sdf",
      path: "files/source.sdf",
      source_version: { mtime_ns: 7, size: 11 },
      payload: { molblock: "initial molblock" },
    },
    viewer_structure: { sites: [{ label: "C", xyz: [0, 0, 0] }], properties: { bonds: [] } },
    summary: { formula: "C", atom_count: 1 },
  });
  let history = createHistory(initial);
  const ketcher = createCanonicalDocument(history.present, {
    snapshot: {
      ...history.present.snapshot,
      payload: { molblock: "edited bond molblock" },
    },
    viewer: history.present.viewer,
    molblock: "edited bond molblock",
    summary: history.present.summary,
    version: history.present.version,
    modified: true,
    moleculeAuthority: "molblock",
  });
  history = pushHistory(history, ketcher);
  history = replaceHistoryPresent(history, createCanonicalDocument(ketcher, {
    snapshot: ketcher.snapshot,
    viewer: {
      sites: [{ label: "C", xyz: [0, 0, 0] }, { label: "O", xyz: [1.2, 0, 0] }],
      properties: { bonds: [{ site_idx_1: 0, site_idx_2: 1, order: 2 }] },
    },
    molblock: ketcher.molblock,
    summary: { formula: "CO", atom_count: 2 },
    version: ketcher.version,
    modified: true,
    moleculeAuthority: "synchronized",
  }));
  const afterKetcher = structuredClone(history.present);
  history = pushHistory(history, createCanonicalDocument(history.present, {
    snapshot: history.present.snapshot,
    viewer: {
      ...history.present.viewer,
      sites: history.present.viewer.sites.map((site, index) => ({
        ...site,
        xyz: [index + 3, 2, 1],
      })),
    },
    summary: history.present.summary,
    version: history.present.version,
    modified: true,
    moleculeAuthority: "viewer",
  }));
  const after3d = structuredClone(history.present);
  history = pushHistory(history, createCanonicalDocument(history.present, {
    snapshot: {
      ...history.present.snapshot,
      payload: { molblock: "candidate molblock" },
    },
    viewer: { sites: [{ label: "N", xyz: [0, 0, 0] }], properties: { bonds: [] } },
    molblock: "candidate molblock",
    summary: { formula: "N", atom_count: 1 },
    version: history.present.version,
    modified: true,
    moleculeAuthority: "synchronized",
  }));
  const candidate = structuredClone(history.present);

  history = undoHistory(history);
  assert.deepEqual(history.present, after3d);
  history = undoHistory(history);
  assert.deepEqual(history.present, afterKetcher);
  history = undoHistory(history);
  assert.deepEqual(history.present, initial);
  history = redoHistory(redoHistory(redoHistory(history)));
  assert.deepEqual(history.present, candidate);
});

test("undo restores symmetry groups cleared by an edit", () => {
  const initial = createCanonicalDocument({
    snapshot: {
      mode: "periodic",
      payload: { pymatgen: { sites: [] } },
      source_version: { mtime_ns: 1, size: 1 },
    },
    viewer_structure: { sites: [] },
    summary: {
      space_group: { symbol: "Fm-3m", number: 225 },
      symmetry_groups: [[0, 1]],
    },
  });
  const edited = createCanonicalDocument(initial, {
    snapshot: initial.snapshot,
    viewer: initial.viewer,
    summary: { ...initial.summary, space_group: null, symmetry_groups: [] },
    symmetryBroken: true,
    version: initial.version,
    modified: true,
  });
  let history = pushHistory(createHistory(initial), edited);
  assert.deepEqual(history.present.summary.symmetry_groups, []);
  history = undoHistory(history);
  assert.deepEqual(history.present.summary.symmetry_groups, [[0, 1]]);
  assert.equal(history.present.symmetryBroken, false);
});

test("trajectory unwrap advances through skipped frames and extraction uses displayed geometry", () => {
  const frameViewer = (fractional) => ({
    lattice: { matrix: [[10, 0, 0], [0, 10, 0], [0, 0, 10]], pbc: [true, true, true] },
    sites: [{
      species: [{ element: "H", occu: 1 }],
      label: "H",
      abc: [fractional, 0, 0],
      xyz: [fractional * 10, 0, 0],
      properties: {},
    }],
  });
  let advanced = advanceUnwrappedTrajectory(frameViewer(0.9));
  for (const fractional of [0.1, 0.3, 0.55]) {
    advanced = advanceUnwrappedTrajectory(frameViewer(fractional), advanced.state);
  }
  assert.ok(Math.abs(advanced.viewer.sites[0].abc[0] - 1.55) < 1e-12);
  assert.ok(Math.abs(advanced.viewer.sites[0].xyz[0] - 15.5) < 1e-12);

  const rawFrame = {
    index: 3,
    atom_count: 1,
    formula: "H",
    snapshot: {
      mode: "periodic",
      format: "trajectory-frame",
      path: "run.traj · frame 3",
      source_version: { mtime_ns: 12, size: 34 },
      payload: { pymatgen: frameViewer(0.55) },
    },
    viewer_structure: frameViewer(0.55),
  };
  const displayed = createDisplayedTrajectoryFrame(rawFrame, advanced.viewer);
  assert.deepEqual(displayed.viewer_structure, advanced.viewer);
  assert.deepEqual(displayed.snapshot.payload.pymatgen, advanced.viewer);
  assert.notDeepEqual(displayed.viewer_structure, rawFrame.viewer_structure);
});

test("new edits clear redo without mutating stored snapshots", () => {
  const original = { sites: [{ xyz: [0, 0, 0] }] };
  let history = pushHistory(createHistory(original), { sites: [{ xyz: [1, 0, 0] }] });
  history = undoHistory(history);
  original.sites[0].xyz[0] = 99;
  history = pushHistory(history, { sites: [{ xyz: [2, 0, 0] }] });
  assert.equal(history.future.length, 0);
  assert.deepEqual(history.past[0].sites[0].xyz, [0, 0, 0]);
});
