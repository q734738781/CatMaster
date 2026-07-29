import { useEffect, useMemo, useRef, useState } from "react";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import "ketcher-react/dist/index.css";

export default function KetcherEditor({ molblock, readOnly = false, onChange, onError }) {
  const provider = useMemo(() => new StandaloneStructServiceProvider(), []);
  const apiRef = useRef(null);
  const subscriptionRef = useRef(null);
  const timerRef = useRef(null);
  const applyingRef = useRef(false);
  const latestRef = useRef(String(molblock || ""));
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const next = String(molblock || "");
    latestRef.current = next;
    if (!apiRef.current || !next.trim()) return;
    applyingRef.current = true;
    apiRef.current.setMolecule(next, { needZoom: true })
      .catch((error) => onError?.(error?.message || String(error)))
      .finally(() => {
        applyingRef.current = false;
      });
  }, [molblock]);

  useEffect(() => () => {
    clearTimeout(timerRef.current);
    if (apiRef.current && subscriptionRef.current) {
      apiRef.current.editor.unsubscribe("change", subscriptionRef.current);
    }
  }, []);

  async function handleInit(ketcher) {
    apiRef.current = ketcher;
    try {
      if (latestRef.current.trim()) {
        applyingRef.current = true;
        await ketcher.setMolecule(latestRef.current, { needZoom: true });
        applyingRef.current = false;
      }
      if (readOnly) ketcher.setSettings({ "general.dearomatize-on-load": false });
      subscriptionRef.current = ketcher.editor.subscribe("change", () => {
        if (applyingRef.current || readOnly) return;
        clearTimeout(timerRef.current);
        timerRef.current = setTimeout(async () => {
          try {
            const next = await ketcher.getMolfile();
            if (next.trim() && next !== latestRef.current) {
              latestRef.current = next;
              onChange?.(next);
            }
          } catch (error) {
            onError?.(error?.message || String(error));
          }
        }, 180);
      });
      setReady(true);
    } catch (error) {
      applyingRef.current = false;
      onError?.(error?.message || String(error));
    }
  }

  return (
    <div className="v2-ketcher-shell" aria-label="Two-dimensional molecule editor">
      {!ready ? <div className="v2-workbench-loading">Starting the 2D molecule editor…</div> : null}
      <Editor
        staticResourcesUrl="/static/"
        structServiceProvider={provider}
        onInit={handleInit}
        errorHandler={(message) => onError?.(String(message || "The molecule editor reported an error."))}
        disableMacromoleculesEditor
        buttons={{
          miew: { hidden: true },
          recognize: { hidden: true },
          fullscreen: { hidden: true },
          settings: { hidden: readOnly },
          clear: { hidden: readOnly },
          open: { hidden: true },
          save: { hidden: true },
        }}
      />
    </div>
  );
}
