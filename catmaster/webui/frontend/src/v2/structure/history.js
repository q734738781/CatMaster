import {
  canonicalPeriodicSnapshot,
  fractionalToCartesian,
  formulaFromSites,
} from "./structureModel.js";

export const MAX_STRUCTURE_HISTORY = 50;

export function createHistory(value) {
  return { past: [], present: structuredClone(value), future: [] };
}

export function pushHistory(history, value, limit = MAX_STRUCTURE_HISTORY) {
  const serializedCurrent = JSON.stringify(history.present);
  const serializedNext = JSON.stringify(value);
  if (serializedCurrent === serializedNext) return history;
  return {
    past: [...history.past, structuredClone(history.present)].slice(-limit),
    present: structuredClone(value),
    future: [],
  };
}

export function undoHistory(history) {
  if (!history.past.length) return history;
  return {
    past: history.past.slice(0, -1),
    present: structuredClone(history.past.at(-1)),
    future: [structuredClone(history.present), ...history.future],
  };
}

export function redoHistory(history) {
  if (!history.future.length) return history;
  return {
    past: [...history.past, structuredClone(history.present)],
    present: structuredClone(history.future[0]),
    future: history.future.slice(1),
  };
}

export function replaceHistoryPresent(history, value) {
  if (!history) return createHistory(value);
  return {
    ...history,
    present: structuredClone(value),
  };
}

export function createCanonicalDocument(payload, overrides = {}) {
  const snapshot = structuredClone(overrides.snapshot || payload?.snapshot || {});
  const viewer = structuredClone(
    overrides.viewer
      || payload?.viewer
      || payload?.viewer_structure
      || {},
  );
  const summary = structuredClone(overrides.summary || payload?.summary || {});
  const version = structuredClone(
    overrides.version
      || payload?.version
      || payload?.source_version
      || snapshot?.source_version
      || { mtime_ns: 0, size: 0 },
  );
  const molblock = snapshot?.mode === "molecule"
    ? String(overrides.molblock ?? payload?.molblock ?? snapshot?.payload?.molblock ?? "")
    : "";
  if (snapshot?.mode === "molecule") {
    snapshot.payload = {
      ...(snapshot.payload || {}),
      molblock,
    };
  }
  if (snapshot && typeof snapshot === "object") {
    snapshot.source_version = structuredClone(version);
  }
  return {
    snapshot,
    viewer,
    molblock,
    summary,
    symmetryBroken: Boolean(overrides.symmetryBroken ?? payload?.symmetryBroken),
    version,
    modified: Boolean(overrides.modified ?? payload?.modified),
    moleculeAuthority: snapshot?.mode === "molecule"
      ? String(overrides.moleculeAuthority ?? payload?.moleculeAuthority ?? "synchronized")
      : "",
  };
}

export function advanceUnwrappedTrajectory(viewer, previousState = null) {
  const rawViewer = structuredClone(viewer || {});
  const sites = Array.isArray(rawViewer.sites) ? rawViewer.sites : [];
  const matrix = rawViewer?.lattice?.matrix;
  const compatible = Boolean(
    matrix
      && previousState?.rawViewer?.lattice?.matrix
      && previousState.rawViewer.sites?.length === sites.length
      && previousState.offsets?.length === sites.length,
  );
  const offsets = compatible
    ? previousState.offsets.map((row) => [...row])
    : sites.map(() => [0, 0, 0]);
  const unwrapped = structuredClone(rawViewer);

  if (compatible) {
    for (let siteIndex = 0; siteIndex < sites.length; siteIndex += 1) {
      const current = sites[siteIndex].abc;
      const last = previousState.rawViewer.sites[siteIndex].abc;
      if (!Array.isArray(current) || !Array.isArray(last)) continue;
      for (let axis = 0; axis < 3; axis += 1) {
        const delta = Number(current[axis]) - Number(last[axis]);
        if (delta > 0.5) offsets[siteIndex][axis] -= 1;
        if (delta < -0.5) offsets[siteIndex][axis] += 1;
      }
      unwrapped.sites[siteIndex].abc = current.map(
        (value, axis) => Number(value) + offsets[siteIndex][axis],
      );
      unwrapped.sites[siteIndex].xyz = fractionalToCartesian(
        matrix,
        unwrapped.sites[siteIndex].abc,
      );
    }
  }

  return {
    viewer: unwrapped,
    state: {
      rawViewer,
      offsets,
    },
  };
}

export function createDisplayedTrajectoryFrame(frame, viewer) {
  const displayedViewer = structuredClone(viewer || {});
  return {
    ...structuredClone(frame || {}),
    atom_count: displayedViewer?.sites?.length || Number(frame?.atom_count || 0),
    formula: formulaFromSites(displayedViewer) || String(frame?.formula || ""),
    viewer_structure: displayedViewer,
    snapshot: frame?.snapshot?.mode === "periodic"
      ? canonicalPeriodicSnapshot(frame.snapshot, displayedViewer)
      : structuredClone(frame?.snapshot || {}),
  };
}
