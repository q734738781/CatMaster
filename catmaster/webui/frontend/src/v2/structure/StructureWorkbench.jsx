import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  ArrowLeftRight,
  Check,
  ChevronLeft,
  Download,
  FlaskConical,
  Redo2,
  Save,
  Undo2,
  X,
} from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import {
  createCanonicalDocument,
  createHistory,
  pushHistory,
  redoHistory,
  replaceHistoryPresent,
  undoHistory,
} from "./history";
import MatterVizBridge from "./MatterVizBridge";
import RendererBoundary from "./RendererBoundary";
import TrajectoryPanel from "./TrajectoryPanel";
import useVolumeLoader from "./useVolumeLoader";
import { MAX_RENDER_SITES } from "./largeStructureView";
import {
  addSite,
  alignSites,
  canonicalPeriodicSnapshot,
  cartesianToFractional,
  centerSites,
  cloneValue,
  connectedFragment,
  crystallographicViewDirection,
  defaultSavePath,
  deleteSites,
  duplicateSites,
  explicitBonds,
  formulaFromSites,
  latticeFromParameters,
  latticeParameters,
  measurement,
  parseIndexSelection,
  replaceSpecies,
  rotateSites,
  selectByElement,
  selectByLayer,
  selectByRadius,
  setMobility,
  setSiteCoordinate,
  siteSpeciesLabel,
  translateSites,
  unwrapSites,
  wrapSites,
} from "./structureModel";

const KetcherEditor = lazy(() => import("./KetcherEditor"));

function useNarrowWorkbench() {
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.matchMedia("(max-width: 760px)").matches);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 760px)");
    const update = () => setNarrow(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return narrow;
}

function vectorFromText(value, integer = false) {
  const values = String(value || "").trim().split(/[\s,;]+/).filter(Boolean).map(Number);
  if (values.length !== 3 || values.some((item) => !Number.isFinite(item))) {
    throw new Error("Enter exactly three numeric values.");
  }
  return integer ? values.map((item) => Math.trunc(item)) : values;
}

function matrixFromText(value, integer = false) {
  const rows = String(value || "").trim().split(/\n|;/).map((row) => row.trim()).filter(Boolean);
  if (rows.length !== 3) throw new Error("Enter three matrix rows separated by new lines.");
  return rows.map((row) => vectorFromText(row, integer));
}

function Section({ title, children, open = false }) {
  return (
    <details className="v2-workbench-section" open={open}>
      <summary>{title}</summary>
      <div className="v2-workbench-section-body">{children}</div>
    </details>
  );
}

function VectorInputs({ value, onChange, step = "0.01", labels = ["x", "y", "z"] }) {
  return (
    <div className="v2-vector-inputs">
      {labels.map((label, axis) => (
        <label key={label}>
          <span>{label}</span>
          <input
            type="number"
            step={step}
            value={Number(value?.[axis] ?? 0)}
            onChange={(event) => {
              const next = [...(value || [0, 0, 0])];
              next[axis] = Number(event.target.value);
              onChange(next);
            }}
          />
        </label>
      ))}
    </div>
  );
}

function ScientificStructureSummary({
  structure,
  summary,
  symmetryBroken = false,
  includeSpaceGroup = false,
  compact = false,
}) {
  const parameters = structure?.lattice?.matrix
    ? latticeParameters(structure.lattice.matrix)
    : null;
  return (
    <dl className={`v2-workbench-summary ${compact ? "compact" : ""}`} aria-label="Scientific structure summary">
      <div><dt>Formula</dt><dd>{formulaFromSites(structure)}</dd></div>
      <div><dt>Sites</dt><dd>{(structure?.sites?.length || 0).toLocaleString()}</dd></div>
      <div><dt>PBC</dt><dd>{structure?.lattice ? "x · y · z" : "None"}</dd></div>
      {parameters ? (
        <div>
          <dt>Cell</dt>
          <dd>{parameters.slice(0, 3).map((value) => value.toFixed(3)).join(" × ")} Å · {parameters.slice(3).map((value) => `${value.toFixed(2)}°`).join(" / ")}</dd>
        </div>
      ) : null}
      {includeSpaceGroup ? (
        <div>
          <dt>Space group</dt>
          <dd>{symmetryBroken ? "Recalculate before save" : `${summary?.space_group?.symbol || "—"} ${summary?.space_group?.number || ""}`}</dd>
        </div>
      ) : null}
    </dl>
  );
}

function CandidateTray({ candidates, previewIndex, onPreview, onApply, onClear }) {
  if (!candidates?.length) return null;
  return (
    <section className="v2-candidate-tray" aria-label="Generated structure candidates">
      <div className="v2-candidate-head">
        <div>
          <strong>{candidates.length} candidates</strong>
          <span>Preview first; only the selected candidate becomes the editable structure.</span>
        </div>
        <button type="button" className="v2-icon-btn" onClick={onClear} aria-label="Close candidate tray"><X size={15} /></button>
      </div>
      <div className="v2-candidate-list">
        {candidates.map((candidate, index) => (
          <article key={`${candidate.label}-${index}`} className={previewIndex === index ? "active" : ""}>
            <button type="button" className="v2-candidate-preview" onClick={() => onPreview(index)}>
              <strong>{candidate.label || `Candidate ${index + 1}`}</strong>
              <span>{candidate.summary?.formula || ""} · {candidate.summary?.atom_count || 0} atoms</span>
              {Number.isFinite(candidate.surface_area) ? <span>Area {candidate.surface_area.toFixed(2)} Å²</span> : null}
              {candidate.top_composition ? <span>Top {candidate.top_composition} · bottom {candidate.bottom_composition}</span> : null}
              {Number.isFinite(candidate.energy_kcal_mol) ? <span>{candidate.energy_kcal_mol.toFixed(3)} kcal mol⁻¹</span> : null}
              {Number.isFinite(candidate.change?.before_atoms) ? <span>{candidate.change.before_atoms} → {candidate.change.after_atoms} atoms</span> : null}
              {candidate.change?.before_space_group ? <span>{candidate.change.before_space_group.symbol} → {candidate.change.after_space_group?.symbol || "unresolved"}</span> : null}
              {candidate.atom_mapping?.length ? <span>{candidate.atom_mapping.length} mapped atom references</span> : null}
            </button>
            <button type="button" onClick={() => onApply(index)}><Check size={14} /> Apply</button>
          </article>
        ))}
      </div>
    </section>
  );
}

function CoordinateTable({ structure, selection, coordinateMode, onCoordinate, readOnly = false }) {
  const PAGE_SIZE = 100;
  const siteCount = structure?.sites?.length || 0;
  const selectedMode = selection.length > 0;
  const totalRows = selectedMode ? selection.length : siteCount;
  const pageCount = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));
  const [page, setPage] = useState(0);
  const [jumpSite, setJumpSite] = useState("");
  const [jumpMessage, setJumpMessage] = useState("");
  const selectionSignature = selectedMode
    ? `${selection.length}:${selection[0]}:${selection.at(-1)}`
    : "";
  useEffect(() => {
    setPage(0);
    setJumpMessage("");
  }, [siteCount, selectionSignature]);
  useEffect(() => {
    if (page >= pageCount) setPage(pageCount - 1);
  }, [page, pageCount]);
  const start = page * PAGE_SIZE;
  const visibleCount = Math.max(0, Math.min(PAGE_SIZE, totalRows - start));
  const indices = Array.from(
    { length: visibleCount },
    (_, offset) => selectedMode ? selection[start + offset] : start + offset,
  );
  const showMobility = Boolean(structure?.lattice);
  const jumpToSite = () => {
    const oneBased = Number.parseInt(jumpSite, 10);
    if (!Number.isInteger(oneBased) || oneBased < 1 || oneBased > siteCount) {
      setJumpMessage(`Enter a site from 1 to ${siteCount.toLocaleString()}.`);
      return;
    }
    const siteIndex = oneBased - 1;
    const rowIndex = selectedMode ? selection.indexOf(siteIndex) : siteIndex;
    if (rowIndex < 0) {
      setJumpMessage(`Site ${oneBased.toLocaleString()} is not in the current selection.`);
      return;
    }
    setPage(Math.floor(rowIndex / PAGE_SIZE));
    setJumpMessage(`Showing site ${oneBased.toLocaleString()}.`);
  };
  const mobilityLabel = (site) => {
    const mobility = Array.isArray(site?.properties?.selective_dynamics)
      ? site.properties.selective_dynamics.map(Boolean)
      : [true, true, true];
    if (mobility.every(Boolean)) return "Free";
    if (mobility.every((value) => !value)) return "Locked";
    return `Move ${["x", "y", "z"].filter((_, axis) => mobility[axis]).join("/") || "none"}`;
  };
  return (
    <div className="v2-coordinate-shell">
      <div className="v2-coordinate-note">
        {selection.length ? `${selection.length} selected` : `${structure?.sites?.length || 0} sites`}
        {totalRows > PAGE_SIZE
          ? ` · rows ${(start + 1).toLocaleString()}–${(start + visibleCount).toLocaleString()}`
          : ""}
      </div>
      {totalRows > PAGE_SIZE ? (
        <div className="v2-coordinate-pager" aria-label="Coordinate table pages">
          <button type="button" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button>
          <span>Page {(page + 1).toLocaleString()} of {pageCount.toLocaleString()}</span>
          <button type="button" disabled={page + 1 >= pageCount} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>Next</button>
          <label>
            Site
            <input
              type="number"
              min="1"
              max={siteCount}
              value={jumpSite}
              aria-label="Go to site number"
              onChange={(event) => setJumpSite(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") jumpToSite();
              }}
            />
          </label>
          <button type="button" onClick={jumpToSite}>Go</button>
          {jumpMessage ? <span role="status">{jumpMessage}</span> : null}
        </div>
      ) : null}
      <div className="v2-coordinate-table" role="table" aria-label={`${coordinateMode} coordinates`}>
        <div className={`v2-coordinate-row header ${showMobility ? "with-mobility" : ""}`} role="row">
          <span>#</span><span>Species</span>
          {(coordinateMode === "fractional" ? ["a", "b", "c"] : ["x Å", "y Å", "z Å"]).map((label) => <span key={label}>{label}</span>)}
          {showMobility ? <span>Mobility</span> : null}
        </div>
        {indices.map((index) => {
          const site = structure.sites[index];
          const values = coordinateMode === "fractional" ? site.abc : site.xyz;
          return (
            <div className={`v2-coordinate-row ${showMobility ? "with-mobility" : ""}`} role="row" key={`${index}-${values.join("-")}`}>
              <span>{index + 1}</span>
              <span>{siteSpeciesLabel(site)}</span>
              {[0, 1, 2].map((axis) => (
                <input
                  key={axis}
                  type="number"
                  step="0.001"
                  disabled={readOnly}
                  defaultValue={Number(values[axis]).toFixed(6)}
                  aria-label={`Site ${index + 1} ${coordinateMode} coordinate ${axis + 1}`}
                  onBlur={(event) => {
                    const next = [...values];
                    next[axis] = Number(event.target.value);
                    onCoordinate(index, next);
                  }}
                />
              ))}
              {showMobility ? <span aria-label={`Site ${index + 1} mobility`}>{mobilityLabel(site)}</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function VolumeWorkbench({ preview, onClose, fallback }) {
  const sourceUrl = preview?.volume?.source_url || preview?.download_url || "";
  const loader = useVolumeLoader(sourceUrl, preview?.name || preview?.path || "volume");
  const [sliceEnabled, setSliceEnabled] = useState(false);
  const [sliceMode, setSliceMode] = useState("hkl");
  const [sliceHkl, setSliceHkl] = useState([0, 0, 1]);
  const [sliceDistance, setSliceDistance] = useState(0.5);
  const [slicePoint, setSlicePoint] = useState([0, 0, 0]);
  const [sliceNormal, setSliceNormal] = useState([0, 0, 1]);
  const [rendererError, setRendererError] = useState("");
  useEffect(() => {
    loader.load();
  }, [sourceUrl]);
  return (
    <div className="v2-structure-workbench" role="dialog" aria-modal="true" aria-label={`Volume workbench for ${preview?.name || "scalar field"}`}>
      <header className="v2-workbench-topbar">
        <div>
          <strong>{preview?.name || "Scalar field"}</strong>
          <span className="v2-workbench-badge">Volume</span>
        </div>
        <button type="button" className="v2-icon-btn" onClick={onClose} aria-label="Close volume workbench"><X size={18} /></button>
      </header>
      <main className="v2-volume-workbench-grid">
        <section className="v2-workbench-canvas">
          {loader.status === "ready" ? (
            <RendererBoundary fallback={fallback} onError={setRendererError}>
              <MatterVizBridge
                structure={loader.structure}
                volumeData={loader.volumes}
                readOnly
                slice={{
                  enabled: sliceEnabled,
                  mode: sliceMode,
                  hkl: sliceHkl,
                  distance: sliceDistance,
                  point: slicePoint,
                  normal: sliceNormal,
                }}
                onError={setRendererError}
              />
            </RendererBoundary>
          ) : (
            <div className="v2-workbench-loading">
              <strong>{loader.message || "Preparing the scalar field…"}</strong>
              {loader.status === "loading" ? <progress max="1" value={loader.progress || undefined} /> : null}
              {loader.status === "loading" ? <button type="button" onClick={loader.cancel}>Cancel</button> : null}
              {["error", "cancelled"].includes(loader.status) ? <button type="button" onClick={loader.load}>Try again</button> : null}
            </div>
          )}
        </section>
        <aside className="v2-workbench-right">
          <h2>Scalar fields</h2>
          {rendererError ? <div className="v2-error compact">{rendererError}</div> : null}
          {(loader.volumes || []).map((volume, index) => (
            <article className="v2-volume-record" key={`${volume.label}-${index}`}>
              <strong>{volume.label || `Field ${index + 1}`}</strong>
              <span>Range {Number(volume.data_range?.min).toPrecision(4)} to {Number(volume.data_range?.max).toPrecision(4)}</span>
              <span>Grid {(volume.source_grid_dims || volume.grid_dims || []).join(" × ")}</span>
              {volume.downsample_factor > 1 ? <span>Interactive mesh uses {volume.downsample_factor}× downsampling</span> : null}
            </article>
          ))}
          <Section title="Volume slice" open>
            <label><input type="checkbox" checked={sliceEnabled} onChange={(event) => setSliceEnabled(event.target.checked)} /> Show scalar-field slice</label>
            <label>Plane definition
              <select value={sliceMode} onChange={(event) => setSliceMode(event.target.value)}>
                <option value="hkl">Crystallographic (hkl)</option>
                <option value="cartesian">Arbitrary Cartesian plane</option>
              </select>
            </label>
            {sliceMode === "hkl" ? (
              <>
                <VectorInputs value={sliceHkl} step="1" labels={["h", "k", "l"]} onChange={setSliceHkl} />
                <label>Plane distance <input type="number" step="0.05" value={sliceDistance} onChange={(event) => setSliceDistance(Number(event.target.value))} /></label>
              </>
            ) : (
              <>
                <label>Point on plane (Å)</label>
                <VectorInputs value={slicePoint} labels={["x", "y", "z"]} onChange={setSlicePoint} />
                <label>Plane normal</label>
                <VectorInputs value={sliceNormal} labels={["nx", "ny", "nz"]} onChange={setSliceNormal} />
              </>
            )}
          </Section>
          <p className="v2-muted">Positive and negative isosurfaces, field selection, level, color, opacity, and display range are available in the 3D canvas controls.</p>
        </aside>
      </main>
    </div>
  );
}

function StructureDocumentWorkbench({
  workspaceName,
  path,
  preview,
  onClose,
  onSaved,
  fallback,
}) {
  const narrow = useNarrowWorkbench();
  const workbenchRef = useRef(null);
  const closeButtonRef = useRef(null);
  const previousFocusRef = useRef(null);
  const saveInputRef = useRef(null);
  const moleculeSyncSequenceRef = useRef(0);
  const moleculeBusySequenceRef = useRef(0);
  const [document, setDocument] = useState(null);
  const [history, setHistory] = useState(null);
  const [selection, setSelection] = useState([]);
  const [measuredSelection, setMeasuredSelection] = useState([]);
  const [coordinateMode, setCoordinateMode] = useState("fractional");
  const [warnings, setWarnings] = useState([]);
  const [error, setError] = useState("");
  const [rendererError, setRendererError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [moleculeSyncStatus, setMoleculeSyncStatus] = useState("idle");
  const [compare, setCompare] = useState(false);
  const [closeWarning, setCloseWarning] = useState(false);
  const [candidateState, setCandidateState] = useState(null);
  const [candidatePreview, setCandidatePreview] = useState(-1);
  const [destination, setDestination] = useState(defaultSavePath(path, "periodic"));
  const [overwrite, setOverwrite] = useState(false);
  const [overwriteConfirmation, setOverwriteConfirmation] = useState(null);
  const [formatLoss, setFormatLoss] = useState([]);
  const [selectionExpression, setSelectionExpression] = useState("");
  const [selectionGesture, setSelectionGesture] = useState("");
  const [elementExpression, setElementExpression] = useState("");
  const [layerSettings, setLayerSettings] = useState({ axis: 2, center: 0, tolerance: 0.6 });
  const [radius, setRadius] = useState(3);
  const [coordinationRadius, setCoordinationRadius] = useState(3);
  const [newAtom, setNewAtom] = useState({ element: "C", coords: [0.5, 0.5, 0.5] });
  const [replacement, setReplacement] = useState("C");
  const [translation, setTranslation] = useState([0, 0, 0.2]);
  const [rotation, setRotation] = useState({ degrees: 90, axis: [0, 0, 1], origin: "centroid" });
  const [alignmentTarget, setAlignmentTarget] = useState([0, 0, 1]);
  const [supercellMatrix, setSupercellMatrix] = useState("2 0 0\n0 2 0\n0 0 1");
  const [cellMatrix, setCellMatrix] = useState("");
  const [cellParameters, setCellParameters] = useState([1, 1, 1, 90, 90, 90]);
  const [cellKeep, setCellKeep] = useState("fractional");
  const [symmetrySettings, setSymmetrySettings] = useState({ symprec: 0.01, angle_tolerance: 5 });
  const [slabSettings, setSlabSettings] = useState({
    miller_index: "1 1 1",
    min_slab_size: 10,
    min_vacuum_size: 15,
    center_slab: true,
    symmetrize: false,
    orthogonal: false,
    lll_reduce: false,
  });
  const [surfaceSupercell, setSurfaceSupercell] = useState("1 0 0\n0 1 0\n0 0 1");
  const [crystalView, setCrystalView] = useState({ kind: "uvw", indices: [0, 0, 1] });
  const [viewDirection, setViewDirection] = useState(undefined);
  const [defectSettings, setDefectSettings] = useState({
    kind: "vacancy",
    new_species: "",
    site_index: -1,
    coordinates: [0.5, 0.5, 0.5],
    coordinate_type: "fractional",
  });
  const [adsorbateMolblock, setAdsorbateMolblock] = useState("");
  const [adsorptionHeight, setAdsorptionHeight] = useState(2);
  const [adsorptionOrientation, setAdsorptionOrientation] = useState([0, 0, 0]);
  const [moleculeView, setMoleculeView] = useState("2d");
  const [representation, setRepresentation] = useState("ball-stick");
  const [measurementMode, setMeasurementMode] = useState("distance");
  const [useMic, setUseMic] = useState(false);
  const [trajectoryViewer, setTrajectoryViewer] = useState(null);
  const currentDocument = history?.present || null;
  const currentSnapshot = currentDocument?.snapshot || null;
  const currentViewer = currentDocument?.viewer || null;
  const currentSummary = currentDocument?.summary || {};
  const modified = Boolean(currentDocument?.modified);
  const symmetryBroken = Boolean(currentDocument?.symmetryBroken);
  const isTrajectory = Boolean(document?.capabilities?.trajectory);
  const isVibration = Boolean(document?.capabilities?.vibration_fallback);
  const isReadOnlyDocument = Boolean(document && document.capabilities?.editable === false);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    apiFetch("/api/structures/open", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({ workspace: workspaceName, path }),
    })
      .then((payload) => {
        if (cancelled) return;
        setDocument(payload);
        setCoordinateMode(payload.snapshot.mode === "periodic" ? "fractional" : "cartesian");
        setHistory(createHistory(createCanonicalDocument(payload)));
        setWarnings(payload.warnings || []);
        if (payload.capabilities?.trajectory || payload.capabilities?.vibration_fallback) setMoleculeView("3d");
        setDestination(defaultSavePath(path, payload.snapshot.mode));
        setOverwriteConfirmation(null);
        setMoleculeSyncStatus("idle");
        const matrix = payload.viewer_structure?.lattice?.matrix;
        if (matrix) {
          setCellMatrix(matrix.map((row) => row.map((item) => Number(item).toFixed(8)).join(" ")).join("\n"));
          setCellParameters(latticeParameters(matrix));
        }
      })
      .catch((reason) => {
        if (!cancelled) setError(reason.message || String(reason));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [workspaceName, path]);

  useEffect(() => {
    previousFocusRef.current = window.document.activeElement;
    return () => {
      const previous = previousFocusRef.current;
      if (previous && window.document.contains(previous)) previous.focus();
    };
  }, []);

  useEffect(() => {
    if (!loading) closeButtonRef.current?.focus();
  }, [loading]);

  useEffect(() => {
    function keydown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        if (modified) setCloseWarning(true);
        else onClose?.();
        return;
      }
      if (event.key !== "Tab") return;
      const root = workbenchRef.current;
      if (!root) return;
      const focusable = [...root.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )].filter((element) => (
        element.getAttribute("aria-hidden") !== "true"
        && element.getClientRects().length > 0
      ));
      if (!focusable.length) {
        event.preventDefault();
        root.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable.at(-1);
      const active = window.document.activeElement;
      if (!root.contains(active) || (event.shiftKey && active === first)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  }, [modified, onClose]);

  const activeViewer = useMemo(() => {
    if (candidatePreview >= 0) {
      const candidate = candidateState?.candidates?.[candidatePreview];
      return candidate?.viewer_structure || candidate?.snapshot?.payload?.pymatgen || currentViewer;
    }
    return trajectoryViewer || currentViewer;
  }, [candidatePreview, candidateState, trajectoryViewer, currentViewer]);
  const largeStructure = (activeViewer?.sites?.length || 0) > MAX_RENDER_SITES;

  useEffect(() => {
    if (largeStructure && representation !== "ball-stick") {
      setRepresentation("ball-stick");
    }
  }, [largeStructure, representation]);

  const snapshot = currentSnapshot;

  const selectedForMeasure = measuredSelection.length ? measuredSelection : selection;
  const measured = useMemo(
    () => measurement(currentViewer, selectedForMeasure, measurementMode, useMic),
    [currentViewer, selectedForMeasure.join(","), measurementMode, useMic],
  );

  function applyViewer(next, { breaksSymmetry = true } = {}) {
    if (!next || !currentDocument || isReadOnlyDocument) return;
    const periodic = currentSnapshot?.mode === "periodic";
    const nextSymmetryBroken = periodic && breaksSymmetry
      ? true
      : currentDocument.symmetryBroken;
    const nextSummary = {
      ...currentSummary,
      formula: formulaFromSites(next),
      atom_count: next?.sites?.length || 0,
      pbc: next?.lattice?.pbc || currentSummary.pbc || [],
      ...(periodic && breaksSymmetry
        ? { symmetry_groups: [], space_group: null }
        : {}),
    };
    const nextSnapshot = periodic
      ? canonicalPeriodicSnapshot(currentSnapshot, next)
      : currentSnapshot;
    const nextDocument = createCanonicalDocument(currentDocument, {
      snapshot: nextSnapshot,
      viewer: next,
      summary: nextSummary,
      symmetryBroken: nextSymmetryBroken,
      version: currentDocument.version,
      modified: true,
      moleculeAuthority: periodic ? "" : "viewer",
    });
    setHistory((current) => pushHistory(current, nextDocument));
    setTrajectoryViewer(null);
    setCandidatePreview(-1);
  }

  function applyHistory(direction) {
    if (isReadOnlyDocument || moleculeSyncStatus === "pending") return;
    moleculeSyncSequenceRef.current += 1;
    setMoleculeSyncStatus("idle");
    setHistory((current) => direction === "undo" ? undoHistory(current) : redoHistory(current));
    setTrajectoryViewer(null);
    setCandidatePreview(-1);
  }

  async function runTransform(operation, params, { retainCandidates = false } = {}) {
    if (!snapshot || isReadOnlyDocument) return;
    setBusy(operation);
    setError("");
    try {
      let transformDocument = currentDocument;
      let transformInput = currentSnapshot;
      if (snapshot.mode === "molecule" && operation === "molecule_conformers") {
        const direction = currentDocument.moleculeAuthority === "molblock"
          ? "from_2d"
          : "from_3d";
        const synchronized = currentDocument.moleculeAuthority === "synchronized"
          ? { document: currentDocument }
          : await synchronizeMolecule(direction, {
          requestedDocument: currentDocument,
          manageBusy: false,
        });
        if (!synchronized) return;
        transformDocument = synchronized.document;
        transformInput = transformDocument.snapshot;
      }
      const result = await apiFetch("/api/structures/transform", {
        method: "POST",
        body: JSON.stringify({ operation, input: transformInput, params }),
      });
      setWarnings(result.warnings || []);
      if (result.kind === "candidates") {
        setCandidateState(result);
        setCandidatePreview(result.candidates?.length ? 0 : -1);
      } else if (result.snapshot) {
        const viewer = result.snapshot.mode === "periodic"
          ? result.snapshot.payload.pymatgen
          : result.viewer_structure;
        setCandidateState({
          kind: "candidates",
          candidate_type: operation,
          candidates: [{
            label: `${operation.replaceAll("_", " ")} preview`,
            snapshot: result.snapshot,
            viewer_structure: viewer,
            summary: result.summary,
            atom_mapping: result.atom_mapping || [],
            change: result.change || {},
            warnings: result.warnings || [],
          }],
        });
        setCandidatePreview(0);
      }
    } catch (reason) {
      setError(reason.message || String(reason));
    } finally {
      setBusy("");
    }
  }

  function applyCandidate(index) {
    const candidate = candidateState?.candidates?.[index];
    if (!candidate?.snapshot || !currentDocument || isReadOnlyDocument) return;
    const viewer = candidate.viewer_structure || candidate.snapshot.payload?.pymatgen;
    const operation = candidateState.candidate_type;
    const preservesSymmetry = ["conformer", "primitive", "conventional", "standardize", "symmetrize", "make_supercell"].includes(operation);
    const nextSymmetryBroken = candidate.snapshot.mode === "periodic"
      && !candidate.change?.after_space_group
      && !preservesSymmetry;
    const nextSummary = nextSymmetryBroken
      ? { ...(candidate.summary || {}), symmetry_groups: [], space_group: null }
      : candidate.summary;
    const nextDocument = createCanonicalDocument(currentDocument, {
      snapshot: candidate.snapshot,
      viewer,
      summary: nextSummary,
      symmetryBroken: nextSymmetryBroken,
      version: candidate.snapshot.source_version || currentDocument.version,
      modified: true,
      moleculeAuthority: candidate.snapshot.mode === "molecule" ? "synchronized" : "",
    });
    moleculeSyncSequenceRef.current += 1;
    setMoleculeSyncStatus("idle");
    setHistory((current) => pushHistory(current, nextDocument));
    setTrajectoryViewer(null);
    setCandidateState(null);
    setCandidatePreview(-1);
    const selectedSources = new Set(selection);
    const mappedSelection = Array.isArray(candidate.atom_mapping)
      ? candidate.atom_mapping.flatMap((sourceIndex, newIndex) => selectedSources.has(sourceIndex) ? [newIndex] : [])
      : [];
    setSelection(mappedSelection);
  }

  async function save(acceptFormatLoss = false, confirmedVersion = null) {
    if (!currentDocument || isReadOnlyDocument || moleculeSyncStatus === "pending") return;
    setBusy("save");
    setError("");
    setFormatLoss([]);
    try {
      let documentToSave = currentDocument;
      if (
        documentToSave.snapshot.mode === "molecule"
        && documentToSave.moleculeAuthority !== "synchronized"
      ) {
        const direction = documentToSave.moleculeAuthority === "molblock"
          ? "from_2d"
          : "from_3d";
        const synchronized = await synchronizeMolecule(direction, {
          requestedDocument: documentToSave,
          manageBusy: false,
        });
        if (!synchronized) return;
        documentToSave = synchronized.document;
      }
      const approvedVersion = confirmedVersion
        || (
          overwriteConfirmation?.path === destination
            ? overwriteConfirmation.source_version
            : null
        );
      const expectedVersion = approvedVersion
        || (
          destination === documentToSave.snapshot.path
            ? documentToSave.version
            : { mtime_ns: 0, size: 0 }
        );
      const result = await apiFetch("/api/structures/save", {
        method: "POST",
        body: JSON.stringify({
          workspace: workspaceName,
          destination_path: destination,
          snapshot: documentToSave.snapshot,
          viewer_structure: documentToSave.snapshot.mode === "molecule"
            ? documentToSave.viewer
            : {},
          overwrite,
          expected_source_version: expectedVersion,
          accept_format_loss: acceptFormatLoss,
          cif_symprec: symmetrySettings.symprec,
          cif_angle_tolerance: symmetrySettings.angle_tolerance,
        }),
      });
      if (result.requires_overwrite_confirmation) {
        setOverwriteConfirmation({
          path: result.path,
          source_version: result.source_version,
        });
        return;
      }
      const savedDocument = createCanonicalDocument(result, {
        symmetryBroken: false,
        modified: false,
        moleculeAuthority: result.snapshot.mode === "molecule" ? "synchronized" : "",
      });
      setHistory(createHistory(savedDocument));
      setWarnings(result.warnings || []);
      setMoleculeSyncStatus("idle");
      setOverwrite(false);
      setOverwriteConfirmation(null);
      setDestination(defaultSavePath(result.path, result.snapshot.mode));
      onSaved?.(result);
    } catch (reason) {
      if (reason.status === 422 && Array.isArray(reason.details?.warnings)) {
        setFormatLoss(reason.details.warnings);
      } else {
        setError(reason.message || String(reason));
      }
    } finally {
      setBusy("");
    }
  }

  function updateMoleculeMolblock(molblock) {
    if (
      !currentDocument
      || currentSnapshot?.mode !== "molecule"
      || isReadOnlyDocument
      || molblock === currentDocument.molblock
    ) return;
    moleculeSyncSequenceRef.current += 1;
    setMoleculeSyncStatus("idle");
    setHistory((current) => {
      const base = current.present;
      const nextSnapshot = {
        ...base.snapshot,
        payload: {
          ...(base.snapshot.payload || {}),
          molblock,
        },
      };
      return pushHistory(current, createCanonicalDocument(base, {
        snapshot: nextSnapshot,
        viewer: base.viewer,
        molblock,
        summary: base.summary,
        symmetryBroken: base.symmetryBroken,
        version: base.version,
        modified: true,
        moleculeAuthority: "molblock",
      }));
    });
  }

  async function synchronizeMolecule(
    direction,
    {
      requestedDocument,
      manageBusy = true,
    } = {},
  ) {
    const baseDocument = requestedDocument || currentDocument;
    const input = baseDocument?.snapshot;
    if (!input || input.mode !== "molecule") return null;
    const sequence = ++moleculeSyncSequenceRef.current;
    const operation = direction === "from_3d" ? "molecule_from_viewer" : "molecule_refresh";
    setMoleculeSyncStatus("pending");
    if (manageBusy) {
      moleculeBusySequenceRef.current = sequence;
      setBusy("molecule_sync");
    }
    setRendererError("");
    try {
      const result = await apiFetch("/api/structures/transform", {
        method: "POST",
        body: JSON.stringify({
          operation,
          input,
          params: direction === "from_3d"
            ? { viewer_structure: baseDocument.viewer || {} }
            : {},
        }),
      });
      if (sequence !== moleculeSyncSequenceRef.current) return null;
      const synchronizedDocument = createCanonicalDocument(baseDocument, {
        snapshot: result.snapshot,
        viewer: result.viewer_structure,
        summary: result.summary,
        symmetryBroken: baseDocument.symmetryBroken,
        version: baseDocument.version,
        modified: baseDocument.modified,
        moleculeAuthority: "synchronized",
      });
      setHistory((current) => replaceHistoryPresent(current, synchronizedDocument));
      setWarnings(result.warnings || []);
      setMoleculeSyncStatus("idle");
      return { result, document: synchronizedDocument };
    } catch (reason) {
      if (sequence === moleculeSyncSequenceRef.current) {
        setRendererError(reason.message || String(reason));
        setMoleculeSyncStatus("error");
      }
      return null;
    } finally {
      if (manageBusy && moleculeBusySequenceRef.current === sequence) {
        moleculeBusySequenceRef.current = 0;
        setBusy("");
      }
    }
  }

  useEffect(() => {
    if (
      currentDocument?.snapshot?.mode === "molecule"
      && currentDocument.moleculeAuthority === "molblock"
      && moleculeSyncStatus === "idle"
    ) {
      void synchronizeMolecule("from_2d", {
        requestedDocument: currentDocument,
        manageBusy: false,
      });
    }
  }, [
    currentDocument?.molblock,
    currentDocument?.moleculeAuthority,
    moleculeSyncStatus,
  ]);

  useEffect(() => {
    const siteCount = currentViewer?.sites?.length || 0;
    setSelection((current) => current.filter((index) => index >= 0 && index < siteCount));
    setMeasuredSelection((current) => current.filter((index) => index >= 0 && index < siteCount));
  }, [currentViewer?.sites?.length]);

  function applySelectionAction(action) {
    try {
      const structure = currentViewer;
      if (action === "indices") setSelection(parseIndexSelection(selectionExpression, structure.sites.length));
      if (action === "elements") setSelection(selectByElement(structure, elementExpression));
      if (action === "layer") setSelection(selectByLayer(structure, layerSettings.axis, layerSettings.center, layerSettings.tolerance));
      if (action === "radius") setSelection(selectByRadius(structure, selection[0], radius, mode === "periodic"));
      if (action === "coordination") {
        if (!selection.length) throw new Error("Select a centre atom before selecting its coordination shell.");
        setSelection(selectByRadius(
          structure,
          selection[0],
          coordinationRadius,
          mode === "periodic",
        ));
      }
      if (action === "symmetry") {
        const group = (currentSummary.symmetry_groups || []).find((indices) => indices.some((index) => selection.includes(index)));
        if (!group) throw new Error("Select a periodic atom with a resolved symmetry-equivalent group first.");
        setSelection(group);
      }
      if (action === "fragment") setSelection(connectedFragment(structure, selection[0]));
      if (action === "invert") {
        const current = new Set(selection);
        setSelection(structure.sites.flatMap((_, index) => current.has(index) ? [] : [index]));
      }
    } catch (reason) {
      setError(reason.message || String(reason));
    }
  }

  function mutate(action) {
    if (!currentViewer || narrow || isReadOnlyDocument) return;
    try {
      let next = currentViewer;
      if (action === "add") next = addSite(next, newAtom.element, newAtom.coords, coordinateMode);
      if (action === "delete") next = deleteSites(next, selection);
      if (action === "duplicate") next = duplicateSites(next, selection);
      if (action === "replace") next = replaceSpecies(next, selection, replacement);
      if (action === "translate") next = translateSites(next, selection, translation);
      if (action === "rotate") {
        const useSelectedPair = rotation.origin === "selected_pair";
        if (useSelectedPair && selection.length < 2) {
          throw new Error("Select two atoms to define the rotation axis.");
        }
        const axis = useSelectedPair
          ? currentViewer.sites[selection[1]].xyz.map(
            (value, coordinate) => Number(value) - Number(currentViewer.sites[selection[0]].xyz[coordinate]),
          )
          : rotation.axis;
        const origin = useSelectedPair ? currentViewer.sites[selection[0]].xyz : rotation.origin;
        next = rotateSites(next, selection, rotation.degrees, axis, origin);
      }
      if (action === "align_pair") next = alignSites(next, selection, "pair", alignmentTarget);
      if (action === "align_plane") next = alignSites(next, selection, "plane", alignmentTarget);
      if (action === "wrap") next = wrapSites(next, selection);
      if (action === "unwrap") next = unwrapSites(next, selection);
      if (action === "center") next = centerSites(next, selection);
      applyViewer(next);
      if (["delete", "duplicate", "add"].includes(action)) setSelection([]);
    } catch (reason) {
      setError(reason.message || String(reason));
    }
  }

  function handleClose() {
    if (modified) setCloseWarning(true);
    else onClose?.();
  }

  async function exportPng() {
    try {
      const canvas = window.document.querySelector(
        ".v2-structure-workbench .v2-matterviz-bridge canvas",
      );
      if (!canvas) throw new Error("Open the 3D view before exporting a PNG.");
      const { export_canvas_as_png, scene_registry } = await import("matterviz/io");
      const binding = scene_registry.get(canvas);
      const basename = String(currentSnapshot?.path || path || "structure")
        .split("/")
        .at(-1)
        .replace(/\.[^.]+$/, "");
      export_canvas_as_png(
        canvas,
        `${basename || "structure"}.png`,
        150,
        binding?.scene,
        binding?.camera,
      );
    } catch (reason) {
      setError(reason.message || String(reason));
    }
  }

  if (loading) {
    return (
      <div ref={workbenchRef} className="v2-structure-workbench" role="dialog" aria-modal="true" aria-label="Loading structure workbench" tabIndex="-1">
        <div className="v2-workbench-loading">Opening the scientific structure document…</div>
      </div>
    );
  }
  if (error && !document) {
    return (
      <div ref={workbenchRef} className="v2-structure-workbench" role="dialog" aria-modal="true" aria-label="Structure workbench error" tabIndex="-1">
        <div className="v2-workbench-loading"><div className="v2-error">{error}</div><button type="button" onClick={onClose}>Close</button></div>
      </div>
    );
  }
  if (!document || !history) return null;

  const mode = currentSnapshot.mode;
  return (
    <div ref={workbenchRef} className="v2-structure-workbench" role="dialog" aria-modal="true" aria-label={`Structure workbench for ${currentSnapshot.path || path}`} tabIndex="-1">
      <header className="v2-workbench-topbar">
        <div className="v2-workbench-file">
          <strong>{currentSnapshot.path || path}</strong>
          <span className={`v2-workbench-badge ${modified ? "modified" : ""}`}>{modified ? "Modified" : "Saved"}</span>
          <span className="v2-workbench-badge">{mode === "periodic" ? "Periodic" : "Molecule"}</span>
          {mode === "periodic" ? <span className="v2-workbench-badge">Display copies: view only</span> : null}
          {isVibration ? <span className="v2-workbench-badge warning">Vibration · read-only JSmol</span> : null}
          {symmetryBroken ? <span className="v2-workbench-badge warning">Symmetry changed</span> : null}
        </div>
        <div className="v2-workbench-actions">
          {!narrow ? (
            <>
              {!isReadOnlyDocument ? <button type="button" disabled={!history.past.length || moleculeSyncStatus === "pending"} onClick={() => applyHistory("undo")} title="Undo"><Undo2 size={16} /> Undo</button> : null}
              {!isReadOnlyDocument ? <button type="button" disabled={!history.future.length || moleculeSyncStatus === "pending"} onClick={() => applyHistory("redo")} title="Redo"><Redo2 size={16} /> Redo</button> : null}
              <button type="button" aria-pressed={compare} onClick={() => setCompare((value) => !value)}><ArrowLeftRight size={16} /> Compare</button>
              {!isVibration ? <button type="button" onClick={() => void exportPng()}><Download size={16} /> PNG</button> : null}
              {!isReadOnlyDocument ? <button type="button" disabled={moleculeSyncStatus === "pending"} onClick={() => saveInputRef.current?.focus()}><Save size={16} /> Save As</button> : null}
            </>
          ) : <span className="v2-muted">Read-only preview on narrow screens</span>}
          <button ref={closeButtonRef} type="button" className="v2-icon-btn" onClick={handleClose} aria-label="Close structure workbench"><X size={18} /></button>
        </div>
      </header>

      {closeWarning ? (
        <div className="v2-workbench-confirm" role="alertdialog" aria-label="Unsaved changes">
          <span>This structure has unsaved changes.</span>
          <button type="button" onClick={() => setCloseWarning(false)}>Keep editing</button>
          <button type="button" className="danger" onClick={onClose}>Discard and close</button>
        </div>
      ) : null}

      <main className={`v2-workbench-grid ${narrow ? "narrow" : ""} ${isReadOnlyDocument ? "read-only" : ""}`}>
        {!narrow && !isReadOnlyDocument ? (
          <aside className="v2-workbench-left" aria-label="Structure tools">
            <h2>Tools</h2>
            <Section title="Selection" open>
              <label>Indices or ranges
                <input value={selectionExpression} onChange={(event) => setSelectionExpression(event.target.value)} placeholder="1-12, 18, 24" />
              </label>
              <button type="button" onClick={() => applySelectionAction("indices")}>Select indices</button>
              <label>Elements
                <input value={elementExpression} onChange={(event) => setElementExpression(event.target.value)} placeholder="O Pt" />
              </label>
              <button type="button" onClick={() => applySelectionAction("elements")}>Select species</button>
              <div className="v2-inline-fields">
                <label>Axis
                  <select value={layerSettings.axis} onChange={(event) => setLayerSettings({ ...layerSettings, axis: Number(event.target.value) })}>
                    <option value="0">x</option><option value="1">y</option><option value="2">z</option>
                  </select>
                </label>
                <label>Center <input type="number" step="0.1" value={layerSettings.center} onChange={(event) => setLayerSettings({ ...layerSettings, center: Number(event.target.value) })} /></label>
                <label>± Å <input type="number" min="0" step="0.1" value={layerSettings.tolerance} onChange={(event) => setLayerSettings({ ...layerSettings, tolerance: Number(event.target.value) })} /></label>
              </div>
              <button type="button" onClick={() => applySelectionAction("layer")}>Select layer</button>
              <div className="v2-inline-fields">
                <label>Radius Å <input type="number" min="0.01" step="0.1" value={radius} onChange={(event) => setRadius(Number(event.target.value))} /></label>
                <button type="button" disabled={!selection.length} onClick={() => applySelectionAction("radius")}>Around first selected</button>
              </div>
              <div className="v2-inline-fields">
                <label>Coordination cutoff Å
                  <input type="number" min="0.01" step="0.1" value={coordinationRadius} onChange={(event) => setCoordinationRadius(Number(event.target.value))} />
                </label>
                <button type="button" disabled={!selection.length} onClick={() => applySelectionAction("coordination")}>Select coordination shell</button>
              </div>
              <small>Periodic coordination selection uses the model minimum-image radius helper.</small>
              <div className="v2-button-row">
                <button
                  type="button"
                  aria-pressed={selectionGesture === "box"}
                  onClick={() => setSelectionGesture((current) => current === "box" ? "" : "box")}
                >Box select</button>
                <button
                  type="button"
                  aria-pressed={selectionGesture === "lasso"}
                  onClick={() => setSelectionGesture((current) => current === "lasso" ? "" : "lasso")}
                >Lasso select</button>
                <button type="button" disabled={!selection.length || !explicitBonds(currentViewer).length} onClick={() => applySelectionAction("fragment")}>Connected fragment</button>
                {mode === "periodic" ? <button type="button" disabled={!selection.length || !currentSummary.symmetry_groups?.length} onClick={() => applySelectionAction("symmetry")}>Symmetry equivalents</button> : null}
                <button type="button" onClick={() => applySelectionAction("invert")}>Invert</button>
                <button type="button" onClick={() => setSelection([])}>Clear</button>
              </div>
              <small>{selection.length} base atom{selection.length === 1 ? "" : "s"} selected. Display copies always map to these base indices.</small>
            </Section>

            {mode === "periodic" ? (
              <Section title="Crystallographic view">
                <label>Direction convention
                  <select value={crystalView.kind} onChange={(event) => setCrystalView({ ...crystalView, kind: event.target.value })}>
                    <option value="uvw">Direct-lattice [uvw]</option>
                    <option value="hkl">Plane normal (hkl)</option>
                  </select>
                </label>
                <VectorInputs
                  value={crystalView.indices}
                  step="1"
                  labels={crystalView.kind === "uvw" ? ["u", "v", "w"] : ["h", "k", "l"]}
                  onChange={(indices) => setCrystalView({ ...crystalView, indices })}
                />
                <div className="v2-button-row">
                  <button type="button" onClick={() => {
                    try {
                      setViewDirection(crystallographicViewDirection(currentViewer, crystalView.indices, crystalView.kind));
                    } catch (reason) {
                      setError(reason.message || String(reason));
                    }
                  }}>Apply view</button>
                  <button type="button" disabled={!viewDirection} onClick={() => setViewDirection(undefined)}>Reset view</button>
                </div>
              </Section>
            ) : null}

            <Section title="Build and move">
              <label>New atom <input value={newAtom.element} onChange={(event) => setNewAtom({ ...newAtom, element: event.target.value })} /></label>
              <VectorInputs value={newAtom.coords} onChange={(coords) => setNewAtom({ ...newAtom, coords })} labels={coordinateMode === "fractional" ? ["a", "b", "c"] : ["x", "y", "z"]} />
              <button type="button" onClick={() => mutate("add")}>Add atom</button>
              <label>Change selected to <input value={replacement} onChange={(event) => setReplacement(event.target.value)} /></label>
              <div className="v2-button-row">
                <button type="button" disabled={!selection.length} onClick={() => mutate("replace")}>Replace</button>
                <button type="button" disabled={!selection.length} onClick={() => mutate("duplicate")}>Duplicate</button>
                <button type="button" className="danger" disabled={!selection.length} onClick={() => mutate("delete")}>Delete</button>
              </div>
              <label>Translate selected (Å)</label>
              <VectorInputs value={translation} onChange={setTranslation} />
              <button type="button" disabled={!selection.length} onClick={() => mutate("translate")}>Translate</button>
              <label>Rotate selected</label>
              {rotation.origin !== "selected_pair" ? (
                <VectorInputs value={rotation.axis} onChange={(axis) => setRotation({ ...rotation, axis })} labels={["axis x", "axis y", "axis z"]} />
              ) : <small>The first two selected atoms define the axis and its origin.</small>}
              <div className="v2-inline-fields">
                <label>Degrees <input type="number" value={rotation.degrees} onChange={(event) => setRotation({ ...rotation, degrees: Number(event.target.value) })} /></label>
                <label>Origin
                  <select value={rotation.origin} onChange={(event) => setRotation({ ...rotation, origin: event.target.value })}>
                    <option value="centroid">Selection centroid</option>
                    <option value="world">World origin</option>
                    <option value="selected_pair">First selected atom pair</option>
                  </select>
                </label>
              </div>
              <button type="button" disabled={!selection.length} onClick={() => mutate("rotate")}>Rotate</button>
              <label>Align direction</label>
              <VectorInputs value={alignmentTarget} onChange={setAlignmentTarget} labels={["target x", "target y", "target z"]} />
              <div className="v2-button-row">
                <button type="button" disabled={selection.length < 2} onClick={() => mutate("align_pair")}>Align atom pair</button>
                <button type="button" disabled={selection.length < 3} onClick={() => mutate("align_plane")}>Align plane normal</button>
              </div>
              <div className="v2-button-row">
                <button type="button" onClick={() => mutate("wrap")}>Wrap</button>
                <button type="button" onClick={() => mutate("unwrap")}>Unwrap selected</button>
                <button type="button" onClick={() => mutate("center")}>Center</button>
              </div>
            </Section>

            {mode === "periodic" ? (
              <>
                <Section title="Scientific supercell">
                  <small>Display replication in the 3D controls is view-only. This operation materializes a new cell and atom list.</small>
                  <textarea rows="3" value={supercellMatrix} onChange={(event) => setSupercellMatrix(event.target.value)} />
                  <button type="button" disabled={Boolean(busy)} onClick={() => {
                    try {
                      void runTransform("make_supercell", { matrix: matrixFromText(supercellMatrix, true) });
                    } catch (reason) { setError(reason.message); }
                  }}>Preview and make supercell</button>
                </Section>
                <Section title="Cell and symmetry">
                  <label>3 × 3 lattice matrix (Å)</label>
                  <textarea rows="4" value={cellMatrix} onChange={(event) => setCellMatrix(event.target.value)} />
                  <div className="v2-cell-parameters" aria-label="Cell lengths and angles">
                    {["a Å", "b Å", "c Å", "α °", "β °", "γ °"].map((label, index) => (
                      <label key={label}>{label}
                        <input
                          type="number"
                          min={index < 3 ? "0.000001" : "0.001"}
                          max={index < 3 ? undefined : "179.999"}
                          step="0.01"
                          value={Number(cellParameters[index]).toFixed(5)}
                          onChange={(event) => {
                            const next = [...cellParameters];
                            next[index] = Number(event.target.value);
                            setCellParameters(next);
                          }}
                        />
                      </label>
                    ))}
                  </div>
                  <label>Keep
                    <select value={cellKeep} onChange={(event) => setCellKeep(event.target.value)}>
                      <option value="fractional">Fractional coordinates</option>
                      <option value="cartesian">Cartesian coordinates</option>
                    </select>
                  </label>
                  <button type="button" onClick={() => {
                    try { void runTransform("set_cell", { matrix: matrixFromText(cellMatrix), keep: cellKeep }); }
                    catch (reason) { setError(reason.message); }
                  }}>Preview cell change</button>
                  <button type="button" onClick={() => {
                    try {
                      const matrix = latticeFromParameters(cellParameters);
                      setCellMatrix(matrix.map((row) => row.map((item) => item.toFixed(8)).join(" ")).join("\n"));
                      void runTransform("set_cell", { matrix, keep: cellKeep });
                    } catch (reason) { setError(reason.message); }
                  }}>Preview lengths and angles</button>
                  <div className="v2-inline-fields">
                    <label>symprec <input type="number" min="0.000001" step="0.001" value={symmetrySettings.symprec} onChange={(event) => setSymmetrySettings({ ...symmetrySettings, symprec: Number(event.target.value) })} /></label>
                    <label>Angle tol. <input type="number" min="0.1" step="0.5" value={symmetrySettings.angle_tolerance} onChange={(event) => setSymmetrySettings({ ...symmetrySettings, angle_tolerance: Number(event.target.value) })} /></label>
                  </div>
                  <div className="v2-button-grid">
                    {["primitive", "conventional", "standardize", "symmetrize"].map((operation) => (
                      <button type="button" key={operation} onClick={() => void runTransform(operation, symmetrySettings)}>
                        {operation[0].toUpperCase() + operation.slice(1)}
                      </button>
                    ))}
                  </div>
                </Section>
                <Section title="Slab terminations">
                  <label>Miller index <input value={slabSettings.miller_index} onChange={(event) => setSlabSettings({ ...slabSettings, miller_index: event.target.value })} /></label>
                  <div className="v2-inline-fields">
                    <label>Slab Å <input type="number" min="0.1" value={slabSettings.min_slab_size} onChange={(event) => setSlabSettings({ ...slabSettings, min_slab_size: Number(event.target.value) })} /></label>
                    <label>Vacuum Å <input type="number" min="0" value={slabSettings.min_vacuum_size} onChange={(event) => setSlabSettings({ ...slabSettings, min_vacuum_size: Number(event.target.value) })} /></label>
                  </div>
                  {[
                    ["center_slab", "Center slab"],
                    ["symmetrize", "Symmetric terminations"],
                    ["orthogonal", "Orthogonal c"],
                    ["lll_reduce", "LLL reduce"],
                  ].map(([key, label]) => (
                    <label key={key}><input type="checkbox" checked={slabSettings[key]} onChange={(event) => setSlabSettings({ ...slabSettings, [key]: event.target.checked })} /> {label}</label>
                  ))}
                  <label>Surface supercell (3 × 3 integers)
                    <textarea rows="3" value={surfaceSupercell} onChange={(event) => setSurfaceSupercell(event.target.value)} />
                  </label>
                  <button type="button" onClick={() => {
                    try {
                      void runTransform("slab_candidates", {
                        ...slabSettings,
                        miller_index: vectorFromText(slabSettings.miller_index, true),
                        surface_supercell: matrixFromText(surfaceSupercell, true),
                      });
                    } catch (reason) { setError(reason.message); }
                  }}>Generate all terminations</button>
                </Section>
                <Section title="Defect candidates">
                  <label>Defect
                    <select value={defectSettings.kind} onChange={(event) => setDefectSettings({ ...defectSettings, kind: event.target.value })}>
                      <option value="vacancy">Vacancy</option>
                      <option value="substitution">Substitution</option>
                      <option value="interstitial">Interstitial</option>
                    </select>
                  </label>
                  {defectSettings.kind !== "vacancy" ? <label>New species <input value={defectSettings.new_species} onChange={(event) => setDefectSettings({ ...defectSettings, new_species: event.target.value })} /></label> : null}
                  {defectSettings.kind === "interstitial" ? (
                    <>
                      <label>Coordinates
                        <select
                          value={defectSettings.coordinate_type}
                          onChange={(event) => setDefectSettings({ ...defectSettings, coordinate_type: event.target.value })}
                        >
                          <option value="fractional">Fractional</option>
                          <option value="cartesian">Cartesian (Å)</option>
                        </select>
                      </label>
                      <VectorInputs
                        value={defectSettings.coordinates}
                        labels={defectSettings.coordinate_type === "fractional" ? ["a", "b", "c"] : ["x Å", "y Å", "z Å"]}
                        onChange={(coordinates) => setDefectSettings({ ...defectSettings, coordinates })}
                      />
                      <button
                        type="button"
                        aria-pressed={selectionGesture === "point"}
                        onClick={() => setSelectionGesture((current) => current === "point" ? "" : "point")}
                      >Pick position in 3D view</button>
                      <small>Click to define the sightline through the unit cell, then set its true depth or enter exact fractional coordinates with the keyboard.</small>
                    </>
                  ) : (
                    <label>Specific site (−1 = all inequivalent) <input type="number" min="-1" value={defectSettings.site_index} onChange={(event) => setDefectSettings({ ...defectSettings, site_index: Number(event.target.value) })} /></label>
                  )}
                  <button type="button" onClick={() => void runTransform("defect_candidates", {
                    ...defectSettings,
                    symprec: symmetrySettings.symprec,
                    angle_tolerance: symmetrySettings.angle_tolerance,
                  })}>Generate inequivalent candidates</button>
                </Section>
                <Section title="Adsorption candidates">
                  <label>Adsorbate MolBlock
                    <textarea rows="5" value={adsorbateMolblock} onChange={(event) => setAdsorbateMolblock(event.target.value)} placeholder="Paste an SDF/MOL connection table" />
                  </label>
                  <label>Height (Å) <input type="number" min="0.1" step="0.1" value={adsorptionHeight} onChange={(event) => setAdsorptionHeight(Number(event.target.value))} /></label>
                  <label>Orientation before placement (Euler degrees)</label>
                  <VectorInputs value={adsorptionOrientation} onChange={setAdsorptionOrientation} labels={["x °", "y °", "z °"]} step="1" />
                  <button type="button" disabled={!adsorbateMolblock.trim()} onClick={() => void runTransform("adsorption_candidates", {
                    adsorbate_molblock: adsorbateMolblock,
                    distance: adsorptionHeight,
                    site_kinds: ["ontop", "bridge", "hollow"],
                    reorient: false,
                    orientation_euler_deg: adsorptionOrientation,
                  })}>Generate site previews</button>
                </Section>
              </>
            ) : (
              <Section title="Molecule conformers" open>
                <p>Connectivity, bond order, aromaticity, stereo, and formal charge remain owned by the 2D molecule document.</p>
                <button type="button" onClick={() => void runTransform("molecule_conformers", {
                  count: 10,
                  random_seed: 42,
                  optimize: "mmff",
                  prune_rms_threshold: 0.35,
                })}>Generate ETKDG conformers</button>
              </Section>
            )}
          </aside>
        ) : null}

        <section className="v2-workbench-center">
          {mode === "molecule" && !narrow && !isReadOnlyDocument ? (
            <div className="v2-workbench-view-tabs" role="tablist" aria-label="Molecule editor view">
              <button
                type="button"
                role="tab"
                aria-selected={moleculeView === "2d"}
                className={moleculeView === "2d" ? "active" : ""}
                disabled={moleculeSyncStatus === "pending"}
                onClick={async () => {
                  const result = moleculeView === "3d" && currentDocument.moleculeAuthority !== "synchronized"
                    ? await synchronizeMolecule(
                      currentDocument.moleculeAuthority === "molblock" ? "from_2d" : "from_3d",
                    )
                    : { document: currentDocument };
                  if (result) setMoleculeView("2d");
                }}
              >2D connectivity</button>
              <button
                type="button"
                role="tab"
                aria-selected={moleculeView === "3d"}
                className={moleculeView === "3d" ? "active" : ""}
                disabled={moleculeSyncStatus === "pending"}
                onClick={async () => {
                  const result = moleculeView === "2d" && currentDocument.moleculeAuthority !== "synchronized"
                    ? await synchronizeMolecule(
                      currentDocument.moleculeAuthority === "molblock" ? "from_2d" : "from_3d",
                    )
                    : { document: currentDocument };
                  if (result) setMoleculeView("3d");
                }}
              >3D conformer</button>
            </div>
          ) : null}
          {mode !== "molecule" || moleculeView === "3d" || narrow ? (
            <div className="v2-representation-bar" role="toolbar" aria-label="Structure representation">
              {[
                ["ball-stick", "Ball-stick"],
                ["spacefill", "Spacefill"],
                ["wireframe", "Wireframe"],
                ["polyhedra", "Polyhedra"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={representation === value}
                  disabled={largeStructure && value !== "ball-stick"}
                  title={largeStructure && value !== "ball-stick"
                    ? `Unavailable above ${MAX_RENDER_SITES.toLocaleString()} displayed atoms; the complete structure remains available to selection and editing tools.`
                    : undefined}
                  onClick={() => setRepresentation(value)}
                >{label}</button>
              ))}
            </div>
          ) : null}
          {narrow ? (
            <ScientificStructureSummary
              structure={currentViewer}
              summary={currentSummary}
              symmetryBroken={symmetryBroken}
              compact
            />
          ) : null}
          <div className="v2-workbench-canvas">
            {isVibration ? (
              <div className="v2-renderer-fallback" role="region" aria-label="Read-only vibration viewer">
                <strong>OUTCAR vibration data opens in the read-only JSmol compatibility viewer.</strong>
                {fallback}
              </div>
            ) : mode === "molecule" && moleculeView === "2d" && !narrow && !isReadOnlyDocument ? (
              <RendererBoundary fallback={<p>Use the MolBlock source or 3D conformer view.</p>} onError={setRendererError}>
                <Suspense fallback={<div className="v2-workbench-loading">Loading the 2D molecule editor only when needed…</div>}>
                  <KetcherEditor
                    molblock={currentDocument.molblock}
                    onChange={updateMoleculeMolblock}
                    onError={setRendererError}
                  />
                </Suspense>
              </RendererBoundary>
            ) : (
              <RendererBoundary fallback={fallback} onError={setRendererError}>
                <MatterVizBridge
                  structure={activeViewer}
                  bonds={mode === "molecule" ? activeViewer?.properties?.bonds : undefined}
                  selection={selection}
                  representation={representation}
                  selectionGesture={selectionGesture}
                  viewDirection={viewDirection}
                  readOnly={narrow || isReadOnlyDocument || candidatePreview >= 0 || Boolean(trajectoryViewer)}
                  onSelectionChange={(selected, selectedMeasured) => {
                    if (selected?.length) setSelection(selected);
                    setMeasuredSelection(selectedMeasured || []);
                  }}
                  onPointPick={(cartesian) => {
                    if (isReadOnlyDocument) return;
                    const coordinates = defectSettings.coordinate_type === "fractional"
                      ? cartesianToFractional(currentViewer.lattice.matrix, cartesian)
                      : cartesian;
                    setDefectSettings((current) => ({ ...current, coordinates }));
                    setSelectionGesture("");
                  }}
                  onStructureChange={(next, bonds) => {
                    if (candidatePreview >= 0 || trajectoryViewer || narrow || isReadOnlyDocument) return;
                    if (bonds && mode === "molecule") {
                      next.properties = { ...(next.properties || {}), bonds };
                    }
                    applyViewer(next);
                  }}
                  onError={setRendererError}
                />
              </RendererBoundary>
            )}
          </div>
          {isTrajectory ? (
            <TrajectoryPanel
              workspaceName={workspaceName}
              path={path}
              onFrame={(_frame, viewer) => {
                setTrajectoryViewer(viewer);
                setCandidatePreview(-1);
              }}
              onExtract={(frame) => {
                const extractedSummary = {
                  formula: frame.formula,
                  atom_count: frame.atom_count,
                  pbc: frame.pbc,
                  symmetry_groups: [],
                  space_group: null,
                };
                const extractedDocument = createCanonicalDocument(frame, {
                  snapshot: frame.snapshot,
                  viewer: frame.viewer_structure,
                  summary: extractedSummary,
                  symmetryBroken: false,
                  version: frame.source_version || frame.snapshot.source_version,
                  modified: true,
                  moleculeAuthority: frame.snapshot.mode === "molecule" ? "viewer" : "",
                });
                setDocument((current) => ({
                  ...current,
                  capabilities: {
                    ...current.capabilities,
                    trajectory: false,
                    vibration_fallback: false,
                    editable: true,
                  },
                }));
                setHistory(createHistory(extractedDocument));
                setTrajectoryViewer(null);
                setMoleculeSyncStatus("idle");
                setDestination(defaultSavePath(`files/frame_${frame.index}`, frame.snapshot.mode));
              }}
            />
          ) : null}
        </section>

        {!narrow ? (
          <aside className="v2-workbench-right" aria-label="Structure properties">
            <h2>Properties</h2>
            <dl className="v2-workbench-summary">
              <div><dt>Formula</dt><dd>{formulaFromSites(currentViewer)}</dd></div>
              <div><dt>Sites</dt><dd>{currentViewer.sites.length.toLocaleString()}</dd></div>
              <div><dt>PBC</dt><dd>{currentViewer.lattice ? "x · y · z" : "None"}</dd></div>
              {currentViewer.lattice ? (
                <div>
                  <dt>Cell</dt>
                  <dd>{latticeParameters(currentViewer.lattice.matrix).slice(0, 3).map((value) => value.toFixed(3)).join(" × ")} Å · {latticeParameters(currentViewer.lattice.matrix).slice(3).map((value) => `${value.toFixed(2)}°`).join(" / ")}</dd>
                </div>
              ) : null}
              <div><dt>Space group</dt><dd>{symmetryBroken ? "Recalculate before save" : `${currentSummary.space_group?.symbol || "—"} ${currentSummary.space_group?.number || ""}`}</dd></div>
            </dl>
            {compare ? (
              <div className="v2-compare-card">
                <strong>Original vs current</strong>
                <span>{document.viewer_structure?.sites?.length || 0} → {currentViewer.sites.length} sites</span>
                <span>{document.summary?.formula || "Original"} → {formulaFromSites(currentViewer)}</span>
              </div>
            ) : null}
            {rendererError ? <div className="v2-error compact">{rendererError}</div> : null}
            {error ? <div className="v2-error compact">{error}</div> : null}
            {warnings.length ? (
              <div className="v2-workbench-warnings">
                {warnings.map((warning, index) => <p key={`${warning}-${index}`}><AlertTriangle size={14} /> {warning}</p>)}
              </div>
            ) : null}
            <Section title="Coordinates" open>
              {mode === "periodic" ? (
                <div className="v2-preview-tabs" role="tablist" aria-label="Coordinate system">
                  {["fractional", "cartesian"].map((value) => (
                    <button key={value} type="button" role="tab" aria-selected={coordinateMode === value} className={coordinateMode === value ? "active" : ""} onClick={() => setCoordinateMode(value)}>
                      {value}
                    </button>
                  ))}
                </div>
              ) : <small>Molecular coordinates are Cartesian Å.</small>}
              <CoordinateTable
                structure={currentViewer}
                selection={selection}
                coordinateMode={coordinateMode}
                readOnly={isReadOnlyDocument}
                onCoordinate={(index, values) => applyViewer(setSiteCoordinate(currentViewer, index, values, coordinateMode))}
              />
            </Section>
            {mode === "periodic" && !isReadOnlyDocument ? (
              <Section title="Selective dynamics">
                <p><span className="v2-lock-glyph" aria-hidden="true">🔒</span> Set movable directions for {selection.length || 0} selected atoms.</p>
                <div className="v2-button-row">
                  {[
                    ["Lock", [false, false, false]],
                    ["Free", [true, true, true]],
                    ["xy only", [true, true, false]],
                    ["z only", [false, false, true]],
                  ].map(([label, mobility]) => (
                    <button type="button" key={label} disabled={!selection.length} onClick={() => applyViewer(setMobility(currentViewer, selection, mobility), { breaksSymmetry: false })}>{label}</button>
                  ))}
                </div>
              </Section>
            ) : null}
            <Section title="Measurement">
              <label>Measure
                <select value={measurementMode} onChange={(event) => setMeasurementMode(event.target.value)}>
                  <option value="distance">Distance (2 atoms)</option>
                  <option value="angle">Angle (3 atoms)</option>
                  <option value="dihedral">Dihedral (4 atoms)</option>
                  <option value="coordination">Coordination within 3 Å</option>
                  {mode === "periodic" ? <option value="cell_a">Cell vector a</option> : null}
                  {mode === "periodic" ? <option value="cell_b">Cell vector b</option> : null}
                  {mode === "periodic" ? <option value="cell_c">Cell vector c</option> : null}
                </select>
              </label>
              {mode === "periodic" ? <label><input type="checkbox" checked={useMic} onChange={(event) => setUseMic(event.target.checked)} /> Minimum-image convention</label> : null}
              {measured ? <output>{measured.label}: {measured.value.toFixed(5)} {measured.unit}</output> : <small>Select enough base atoms. Leave MIC off for slab z/vacuum distances.</small>}
            </Section>
            {!isReadOnlyDocument ? (
              <Section title="Save As" open>
                <label>Workspace path
                  <input
                    ref={saveInputRef}
                    value={destination}
                    onChange={(event) => {
                      setDestination(event.target.value);
                      setOverwriteConfirmation(null);
                      setFormatLoss([]);
                    }}
                  />
                </label>
                <label><input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(event) => {
                    setOverwrite(event.target.checked);
                    setOverwriteConfirmation(null);
                  }}
                /> Explicitly overwrite an existing file</label>
                <button
                  type="button"
                  disabled={!modified || Boolean(busy) || moleculeSyncStatus !== "idle"}
                  onClick={() => void save(false)}
                ><Save size={15} /> {busy === "save" ? "Saving…" : moleculeSyncStatus === "pending" ? "Synchronizing molecule…" : "Save As"}</button>
                <small>Save As is the default. Existing targets are version-checked, shown for confirmation, and checked again at write time.</small>
                {moleculeSyncStatus === "error" && mode === "molecule" ? (
                  <div className="v2-format-loss" role="alert">
                    <strong>The molecular 2D/3D projections are not synchronized, so saving is locked.</strong>
                    <button type="button" onClick={() => {
                      setMoleculeSyncStatus("idle");
                    }}>Retry molecule synchronization</button>
                  </div>
                ) : null}
                {overwriteConfirmation ? (
                  <div className="v2-workbench-confirm" role="alertdialog" aria-label="Confirm version-checked overwrite">
                    <span>
                      {overwriteConfirmation.path} exists ({Number(overwriteConfirmation.source_version.size).toLocaleString()} bytes).
                      Confirm overwrite of this exact version.
                    </span>
                    <button type="button" onClick={() => setOverwriteConfirmation(null)}>Cancel</button>
                    <button
                      type="button"
                      className="danger"
                      onClick={() => void save(false, overwriteConfirmation.source_version)}
                    >Confirm overwrite</button>
                  </div>
                ) : null}
                {formatLoss.length ? (
                  <div className="v2-format-loss" role="alert">
                    <strong>This format cannot preserve everything:</strong>
                    <ul>{formatLoss.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                    <button type="button" onClick={() => void save(true)}>I understand; save with this loss</button>
                  </div>
                ) : null}
              </Section>
            ) : (
              <p className="v2-muted">
                {isTrajectory
                  ? "Trajectory files are read-only. Extract the displayed frame before editing or saving a structure."
                  : isVibration
                    ? "OUTCAR vibration documents are read-only and use the JSmol compatibility viewer."
                    : "This structure document is read-only."}
              </p>
            )}
          </aside>
        ) : null}
      </main>

      {!narrow && !isReadOnlyDocument ? (
        <CandidateTray
          candidates={candidateState?.candidates || []}
          previewIndex={candidatePreview}
          onPreview={setCandidatePreview}
          onApply={applyCandidate}
          onClear={() => {
            setCandidateState(null);
            setCandidatePreview(-1);
          }}
        />
      ) : null}
      {busy && busy !== "save" ? <div className="v2-workbench-busy"><FlaskConical size={15} /> Running {busy.replaceAll("_", " ")} with the scientific backend…</div> : null}
    </div>
  );
}

export default function StructureWorkbench(props) {
  if (props.preview?.kind === "volume") {
    return <VolumeWorkbench preview={props.preview} onClose={props.onClose} fallback={props.fallback} />;
  }
  return <StructureDocumentWorkbench {...props} />;
}
