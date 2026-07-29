import { useCallback, useEffect, useRef, useState } from "react";

export default function useVolumeLoader(sourceUrl, filename) {
  const workerRef = useRef(null);
  const [state, setState] = useState({
    status: sourceUrl ? "idle" : "unavailable",
    progress: 0,
    message: "",
    structure: null,
    volumes: null,
  });

  const cancel = useCallback(() => {
    workerRef.current?.postMessage({ type: "cancel" });
    workerRef.current?.terminate();
    workerRef.current = null;
    setState((current) => ({
      ...current,
      status: current.status === "loading" ? "cancelled" : current.status,
      message: current.status === "loading" ? "Volume loading was cancelled." : current.message,
    }));
  }, []);

  const load = useCallback(() => {
    if (!sourceUrl) return;
    workerRef.current?.terminate();
    const worker = new Worker(new URL("./volumeWorker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setState({
      status: "loading",
      progress: 0,
      message: "Reading the scalar-field file…",
      structure: null,
      volumes: null,
    });
    worker.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.type === "progress") {
        setState((current) => ({
          ...current,
          progress: payload.total > 0 ? Math.min(1, payload.loaded / payload.total) : 0,
          message: payload.total > 0
            ? `Loaded ${(payload.loaded / 1024 / 1024).toFixed(1)} of ${(payload.total / 1024 / 1024).toFixed(1)} MB…`
            : `Loaded ${(payload.loaded / 1024 / 1024).toFixed(1)} MB…`,
        }));
      } else if (payload.type === "stage") {
        setState((current) => ({ ...current, message: payload.message || current.message }));
      } else if (payload.type === "result") {
        setState({
          status: "ready",
          progress: 1,
          message: `${payload.volumes.length} scalar field${payload.volumes.length === 1 ? "" : "s"} ready.`,
          structure: payload.structure,
          volumes: payload.volumes,
        });
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      } else if (payload.type === "error") {
        setState({
          status: "error",
          progress: 0,
          message: payload.message || "The scalar field could not be parsed.",
          structure: null,
          volumes: null,
        });
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      } else if (payload.type === "cancelled") {
        setState({
          status: "cancelled",
          progress: 0,
          message: payload.message || "Volume loading was cancelled.",
          structure: null,
          volumes: null,
        });
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      }
    };
    worker.onerror = (error) => {
      setState({
        status: "error",
        progress: 0,
        message: error.message || "The volume worker stopped unexpectedly.",
        structure: null,
        volumes: null,
      });
    };
    worker.postMessage({ type: "load", url: sourceUrl, filename });
  }, [sourceUrl, filename]);

  useEffect(() => () => workerRef.current?.terminate(), []);
  return { ...state, load, cancel };
}
