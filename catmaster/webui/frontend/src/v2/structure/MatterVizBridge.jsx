import { useEffect, useMemo, useRef } from "react";

import { buildLargeStructureView, displayIndicesToBase } from "./largeStructureView.js";

export default function MatterVizBridge({
  structure,
  bonds,
  selection,
  readOnly = false,
  volumeData,
  slice,
  representation = "ball-stick",
  selectionGesture = "",
  viewDirection,
  onStructureChange,
  onSelectionChange,
  onPointPick,
  onError,
}) {
  const hostRef = useRef(null);
  const componentRef = useRef(null);
  const readOnlyRef = useRef(readOnly);
  readOnlyRef.current = readOnly;
  const callbacksRef = useRef({ onStructureChange, onSelectionChange, onPointPick, onError });
  callbacksRef.current = { onStructureChange, onSelectionChange, onPointPick, onError };
  const projection = useMemo(
    () => buildLargeStructureView(structure, bonds, selection),
    [structure, bonds, selection],
  );
  const projectionRef = useRef(projection);
  projectionRef.current = projection;

  useEffect(() => {
    let disposed = false;
    let unmountComponent = null;
    async function mountRenderer() {
      try {
        const [{ mount, unmount }, { default: MatterVizHost }] = await Promise.all([
          import("svelte"),
          import("./MatterVizHost.svelte"),
        ]);
        if (disposed || !hostRef.current) return;
        const component = mount(MatterVizHost, {
          target: hostRef.current,
          props: {
            structure: projection.structure,
            bonds: projection.bonds,
            read_only: readOnly || projection.isLod,
            large_structure: projection.isLod,
            volumetric_data: volumeData,
            slice_enabled: Boolean(slice?.enabled),
            slice_mode: slice?.mode || "hkl",
            slice_hkl: slice?.hkl || [0, 0, 1],
            slice_distance: Number(slice?.distance ?? 0.5),
            slice_point: slice?.point || [0, 0, 0],
            slice_normal: slice?.normal || [0, 0, 1],
            representation,
            selection_gesture: readOnly ? "" : selectionGesture,
            view_direction: viewDirection,
            on_structure_change: (nextStructure, nextBonds) => {
              if (!readOnlyRef.current && !projectionRef.current.isLod) {
                callbacksRef.current.onStructureChange?.(nextStructure, nextBonds);
              }
            },
            on_selection_change: (selected, measured) => callbacksRef.current.onSelectionChange?.(
              displayIndicesToBase(selected, projectionRef.current.displayToBase),
              displayIndicesToBase(measured, projectionRef.current.displayToBase),
            ),
            on_point_pick: (coordinates) => {
              if (!readOnlyRef.current && !projectionRef.current.isLod) {
                callbacksRef.current.onPointPick?.(coordinates);
              }
            },
            on_error: (message) => callbacksRef.current.onError?.(message),
          },
        });
        componentRef.current = component;
        unmountComponent = () => unmount(component);
        hostRef.current.querySelectorAll("canvas").forEach((canvas) => {
          canvas.setAttribute("role", "img");
          canvas.setAttribute("aria-label", "Interactive three-dimensional structure canvas");
        });
      } catch (error) {
        callbacksRef.current.onError?.(error?.message || String(error));
      }
    }
    mountRenderer();
    return () => {
      disposed = true;
      componentRef.current = null;
      unmountComponent?.();
      if (hostRef.current) hostRef.current.replaceChildren();
    };
  }, []);

  useEffect(() => {
    componentRef.current?.replace_structure?.(projection.structure, projection.bonds);
  }, [projection.structure, projection.bonds]);

  useEffect(() => {
    componentRef.current?.replace_selection?.(projection.selection);
  }, [projection.selection]);

  useEffect(() => {
    componentRef.current?.replace_read_only?.(readOnly || projection.isLod);
  }, [readOnly, projection.isLod]);

  useEffect(() => {
    componentRef.current?.replace_large_structure?.(projection.isLod);
  }, [projection.isLod]);

  useEffect(() => {
    componentRef.current?.replace_volume?.(volumeData);
  }, [volumeData]);

  useEffect(() => {
    componentRef.current?.replace_slice?.(slice || {});
  }, [slice]);

  useEffect(() => {
    componentRef.current?.replace_representation?.(representation);
  }, [representation]);

  useEffect(() => {
    componentRef.current?.replace_selection_gesture?.(readOnly ? "" : selectionGesture || "");
  }, [readOnly, selectionGesture]);

  useEffect(() => {
    componentRef.current?.replace_view_direction?.(viewDirection);
  }, [viewDirection]);

  return (
    <div
      className="v2-matterviz-bridge"
      role="region"
      aria-label={readOnly || projection.isLod ? "Three-dimensional structure preview" : "Three-dimensional structure editor"}
    >
      <div ref={hostRef} className="v2-matterviz-mount" />
      {projection.isLod ? (
        <div className="v2-large-structure-notice" role="status">
          Large-structure view · showing {projection.visible.toLocaleString()} of {projection.total.toLocaleString()} atoms.
          The view uses coarse-to-fine spatial density and keeps selected-atom neighborhoods.
          The full structure remains authoritative; index, species, layer, and coordinate tools still use every atom.
        </div>
      ) : null}
    </div>
  );
}
