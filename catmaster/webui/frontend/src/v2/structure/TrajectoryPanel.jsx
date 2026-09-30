import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import {
  advanceUnwrappedTrajectory,
  createDisplayedTrajectoryFrame,
} from "./history";
import {
  centerSites,
  cloneValue,
} from "./structureModel";

const CACHE_SIZE = 16;

function frameUrl(workspace, path, index) {
  return `/api/trajectories/frame?workspace=${encodeURIComponent(workspace)}&path=${encodeURIComponent(path)}&index=${encodeURIComponent(index)}`;
}

function PropertyPlot({ rows }) {
  const numericKeys = useMemo(() => {
    const keys = new Set();
    for (const row of rows || []) {
      for (const [key, value] of Object.entries(row)) {
        if (key !== "index" && Number.isFinite(Number(value))) keys.add(key);
      }
    }
    return [...keys];
  }, [rows]);
  const [property, setProperty] = useState("");
  useEffect(() => {
    if (!numericKeys.includes(property)) setProperty(numericKeys[0] || "");
  }, [numericKeys.join("|")]);
  if (!numericKeys.length) return <div className="v2-muted">No scalar frame properties were found.</div>;
  const points = (rows || []).filter((row) => Number.isFinite(Number(row[property])));
  const xs = points.map((row) => Number(row.index));
  const ys = points.map((row) => Number(row[property]));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const polyline = points.map((row) => {
    const x = maxX === minX ? 10 : 10 + 280 * (Number(row.index) - minX) / (maxX - minX);
    const y = maxY === minY ? 45 : 85 - 75 * (Number(row[property]) - minY) / (maxY - minY);
    return `${x},${y}`;
  }).join(" ");
  return (
    <div className="v2-trajectory-plot">
      <label>
        Property
        <select value={property} onChange={(event) => setProperty(event.target.value)}>
          {numericKeys.map((key) => <option key={key} value={key}>{key.replaceAll("_", " ")}</option>)}
        </select>
      </label>
      <svg viewBox="0 0 300 96" role="img" aria-label={`${property.replaceAll("_", " ")} across trajectory frames`}>
        <polyline points={polyline} fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <small>{minY.toPrecision(4)} to {maxY.toPrecision(4)}</small>
    </div>
  );
}

export default function TrajectoryPanel({ workspaceName, path, onFrame, onExtract }) {
  const [meta, setMeta] = useState(null);
  const [index, setIndex] = useState(0);
  const [fps, setFps] = useState(8);
  const [stride, setStride] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [unwrap, setUnwrap] = useState(false);
  const [center, setCenter] = useState(false);
  const cacheRef = useRef(new Map());
  const unwrapCacheRef = useRef(new Map());
  const abortRef = useRef(null);
  const controllersRef = useRef(new Set());
  const currentFrameRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    cacheRef.current.clear();
    unwrapCacheRef.current.clear();
    currentFrameRef.current = null;
    setError("");
    apiFetch(`/api/trajectories/meta?workspace=${encodeURIComponent(workspaceName)}&path=${encodeURIComponent(path)}`)
      .then((payload) => {
        if (!cancelled) setMeta(payload);
      })
      .catch((reason) => {
        if (!cancelled) setError(reason.message || String(reason));
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
      for (const controller of controllersRef.current) controller.abort();
      controllersRef.current.clear();
    };
  }, [workspaceName, path]);

  function remember(cache, key, value) {
    cache.delete(key);
    cache.set(key, value);
    while (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value);
  }

  async function loadRawFrame(target, controller) {
    const bounded = Math.max(0, Math.min(Number(meta?.total_frames || 1) - 1, Number(target)));
    let frame = cacheRef.current.get(bounded);
    if (!frame) {
      const activeController = controller || new AbortController();
      controllersRef.current.add(activeController);
      try {
        frame = await apiFetch(
          frameUrl(workspaceName, path, bounded),
          { signal: activeController.signal },
        );
        remember(cacheRef.current, bounded, frame);
      } finally {
        controllersRef.current.delete(activeController);
      }
    }
    return frame;
  }

  async function unwrappedViewerAt(target, targetFrame, controller) {
    const cached = unwrapCacheRef.current.get(target);
    if (cached) {
      remember(unwrapCacheRef.current, target, cached);
      return cloneValue(cached.viewer);
    }

    let startIndex = -1;
    let state = null;
    let viewer = null;
    for (const [candidateIndex, candidate] of unwrapCacheRef.current.entries()) {
      if (candidateIndex <= target && candidateIndex > startIndex) {
        startIndex = candidateIndex;
        state = candidate.state;
        viewer = candidate.viewer;
      }
    }
    if (startIndex < 0) {
      const firstFrame = target === 0 ? targetFrame : await loadRawFrame(0, controller);
      const first = advanceUnwrappedTrajectory(firstFrame.viewer_structure);
      startIndex = 0;
      state = first.state;
      viewer = first.viewer;
      remember(unwrapCacheRef.current, 0, first);
    }
    for (let frameIndex = startIndex + 1; frameIndex <= target; frameIndex += 1) {
      if (controller.signal.aborted) throw new DOMException("Aborted", "AbortError");
      const frame = frameIndex === target
        ? targetFrame
        : await loadRawFrame(frameIndex, controller);
      const advanced = advanceUnwrappedTrajectory(frame.viewer_structure, state);
      state = advanced.state;
      viewer = advanced.viewer;
      remember(unwrapCacheRef.current, frameIndex, advanced);
    }
    return cloneValue(viewer);
  }

  async function fetchFrame(target) {
    const bounded = Math.max(0, Math.min(Number(meta?.total_frames || 1) - 1, Number(target)));
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError("");
    try {
      const frame = await loadRawFrame(bounded, controller);
      let viewer = unwrap
        ? await unwrappedViewerAt(bounded, frame, controller)
        : cloneValue(frame.viewer_structure);
      if (center) viewer = centerSites(viewer);
      if (controller.signal.aborted) return null;
      const displayedFrame = createDisplayedTrajectoryFrame(frame, viewer);
      currentFrameRef.current = displayedFrame;
      setIndex(bounded);
      onFrame?.(displayedFrame, viewer);
      const next = Math.min(Number(meta.total_frames) - 1, bounded + Math.max(1, stride));
      if (next !== bounded && !cacheRef.current.has(next)) {
        void loadRawFrame(next).catch(() => {});
      }
      return displayedFrame;
    } catch (reason) {
      if (reason.name !== "AbortError") setError(reason.message || String(reason));
      return null;
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    if (meta) void fetchFrame(index);
  }, [Boolean(meta), unwrap, center]);

  useEffect(() => {
    if (!playing || !meta) return undefined;
    const timer = window.setInterval(() => {
      const next = index + Math.max(1, stride);
      if (next >= meta.total_frames) {
        setPlaying(false);
        return;
      }
      void fetchFrame(next);
    }, Math.max(25, 1000 / Math.max(1, fps)));
    return () => window.clearInterval(timer);
  }, [playing, index, stride, fps, meta?.total_frames]);

  if (error) return <div className="v2-error compact">{error}</div>;
  if (!meta) return <div className="v2-muted">Reading trajectory metadata without loading every frame into memory…</div>;
  return (
    <section className="v2-trajectory-panel" aria-label="Trajectory controls">
      <div className="v2-trajectory-controls">
        <button type="button" onClick={() => void fetchFrame(0)} aria-label="First frame"><SkipBack size={15} /></button>
        <button type="button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause trajectory" : "Play trajectory"}>
          {playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button type="button" onClick={() => void fetchFrame(meta.total_frames - 1)} aria-label="Last frame"><SkipForward size={15} /></button>
        <label className="v2-frame-slider">
          <span>Frame {index + 1} of {Number(meta.total_frames).toLocaleString()}</span>
          <input
            type="range"
            min="0"
            max={Math.max(0, meta.total_frames - 1)}
            value={index}
            onChange={(event) => void fetchFrame(Number(event.target.value))}
          />
        </label>
        <label>FPS <input type="number" min="1" max="60" value={fps} onChange={(event) => setFps(Number(event.target.value))} /></label>
        <label>Stride <input type="number" min="1" max={Math.max(1, meta.total_frames)} value={stride} onChange={(event) => setStride(Math.max(1, Number(event.target.value)))} /></label>
        <label><input type="checkbox" checked={unwrap} onChange={(event) => setUnwrap(event.target.checked)} /> Unwrap PBC</label>
        <label><input type="checkbox" checked={center} onChange={(event) => setCenter(event.target.checked)} /> Center</label>
        <button type="button" disabled={!currentFrameRef.current} onClick={() => onExtract?.(currentFrameRef.current)}>
          Extract frame
        </button>
        {loading ? <span className="v2-muted">Loading frame…</span> : null}
      </div>
      <PropertyPlot rows={meta.property_series || []} />
    </section>
  );
}
