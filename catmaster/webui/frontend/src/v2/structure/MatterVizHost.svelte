<script>
  import { onMount } from "svelte";
  import { Structure } from "matterviz/structure";
  import { VolumeSlice, sample_hkl_slice, sample_plane_slice } from "matterviz/isosurface";
  import { scene_registry } from "matterviz/io";

  let {
    structure = $bindable(),
    bonds = $bindable(),
    read_only = false,
    large_structure = false,
    volumetric_data = $bindable(),
    on_structure_change,
    on_selection_change,
    on_point_pick,
    on_error,
    slice_enabled = false,
    slice_mode = "hkl",
    slice_hkl = [0, 0, 1],
    slice_distance = 0.5,
    slice_point = [0, 0, 0],
    slice_normal = [0, 0, 1],
    representation = "ball-stick",
    selection_gesture = "",
    view_direction = undefined,
  } = $props();

  let host_element;
  let overlay_element = $state();
  let selected_sites = $state([]);
  let measured_sites = $state([]);
  let controls_open = $state(false);
  let active_volume_idx = $state(0);
  let isosurface_settings = $state({
    isovalue: 0.01,
    opacity: 0.55,
    positive_color: "#2563eb",
    negative_color: "#dc2626",
    show_negative: true,
    wireframe: false,
    halo: 0,
  });
  let initialized = false;
  let selection_initialized = false;
  let replacing_structure = false;
  let replacing_selection = false;
  let gesture_points = $state([]);
  let gesture_active = $state(false);
  let keyboard_box = $state([0.25, 0.25, 0.75, 0.75]);
  let point_pick_ray = $state(null);
  let point_pick_depth = $state(0.5);
  let point_pick_fractional = $state([0.5, 0.5, 0.5]);
  let scene_props = $state({
    auto_rotate: 0,
    show_site_labels: false,
    show_site_indices: false,
    show_image_atoms: true,
    show_axes: true,
    show_cell: true,
    show_bonds: "always",
    show_polyhedra: "never",
    sphere_segments: 20,
    atom_radius: 0.35,
    bond_thickness: 0.08,
  });

  const control_name = (control, fallback) => {
    const title = control.dataset?.originalTitle || control.getAttribute("title");
    if (title?.trim()) return title.trim();
    const label = control.labels?.[0] || control.closest?.("label");
    const label_text = label?.textContent?.replace(/\s+/g, " ").trim();
    return label_text || fallback;
  };

  const label_matterviz_controls = () => {
    for (const button of host_element?.querySelectorAll("button") || []) {
      if (button.getAttribute("aria-label") || button.textContent?.trim()) continue;
      const inherited_title = control_name(button, "");
      if (inherited_title) button.setAttribute("aria-label", inherited_title);
      if (button.classList.contains("structure-controls-toggle")) {
        button.setAttribute(
          "aria-label",
          button.getAttribute("aria-expanded") === "true"
            ? "Close structure controls"
            : "Open structure controls",
        );
      }
    }
    for (const [index, canvas] of [...(host_element?.querySelectorAll("canvas") || [])].entries()) {
      if (canvas.getAttribute("role") !== "img") {
        canvas.setAttribute("role", "img");
      }
      if (!canvas.getAttribute("aria-label")) {
        canvas.setAttribute(
          "aria-label",
          `Interactive three-dimensional structure canvas ${index + 1}`,
        );
      }
    }
    const controls = host_element?.querySelectorAll(
      'input[type="range"], input[type="number"], [role="slider"], [role="spinbutton"]',
    ) || [];
    for (const [index, control] of [...controls].entries()) {
      if (control.getAttribute("aria-label") || control.getAttribute("aria-labelledby")) continue;
      const kind = control.matches('input[type="range"], [role="slider"]') ? "slider" : "number";
      control.setAttribute(
        "aria-label",
        control_name(control, `Structure ${kind} control ${index + 1}`),
      );
    }
  };

  onMount(() => {
    label_matterviz_controls();
    const observer = new MutationObserver(label_matterviz_controls);
    observer.observe(host_element, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["aria-expanded", "title", "data-original-title", "type", "role"],
    });
    return () => observer.disconnect();
  });

  $effect(() => {
    if (!selection_gesture || read_only || !overlay_element) return;
    queueMicrotask(() => overlay_element?.focus({ preventScroll: true }));
  });

  let custom_views = $derived(
    Array.isArray(view_direction) && view_direction.length === 3
      ? [{ label: "Crystallographic", direction: view_direction, projection: "orthographic" }]
      : undefined,
  );

  const point_in_polygon = (point, polygon) => {
    let inside = false;
    for (let left = 0, right = polygon.length - 1; left < polygon.length; right = left++) {
      const [xi, yi] = polygon[left];
      const [xj, yj] = polygon[right];
      if ((yi > point[1]) !== (yj > point[1])
        && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi || Number.EPSILON) + xi) {
        inside = !inside;
      }
    }
    return inside;
  };

  const event_point = (event) => {
    const rect = host_element?.getBoundingClientRect();
    return rect ? [event.clientX - rect.left, event.clientY - rect.top] : [0, 0];
  };

  const cartesian_to_fractional = (matrix, cartesian) => {
    if (!Array.isArray(matrix) || matrix.length !== 3) return null;
    const [[a, b, c], [d, e, f], [g, h, i]] = matrix.map((row) => row.map(Number));
    const determinant = a * (e * i - f * h)
      - b * (d * i - f * g)
      + c * (d * h - e * g);
    if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null;
    const inverse = [
      [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
      [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
      [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
    ];
    return [
      cartesian.reduce((sum, value, index) => sum + Number(value) * inverse[index][0], 0),
      cartesian.reduce((sum, value, index) => sum + Number(value) * inverse[index][1], 0),
      cartesian.reduce((sum, value, index) => sum + Number(value) * inverse[index][2], 0),
    ];
  };

  const fractional_to_cartesian = (matrix, fractional) => {
    if (!Array.isArray(matrix) || matrix.length !== 3) return null;
    return [0, 1, 2].map((axis) => fractional.reduce(
      (sum, value, vector) => sum + Number(value) * Number(matrix[vector]?.[axis]),
      0,
    ));
  };

  const point_on_ray = (ray, depth) => ray.start.map(
    (value, axis) => value + (ray.end[axis] - value) * depth,
  );

  const set_point_depth = (next_depth) => {
    point_pick_depth = Math.min(1, Math.max(0, Number(next_depth)));
    if (point_pick_ray) {
      point_pick_fractional = point_on_ray(point_pick_ray, point_pick_depth);
    }
  };

  const ray_cell_segment = (ray_origin, ray_direction) => {
    const matrix = structure?.lattice?.matrix;
    const fractional_origin = cartesian_to_fractional(matrix, ray_origin);
    const fractional_tip = cartesian_to_fractional(
      matrix,
      ray_origin.map((value, axis) => value + ray_direction[axis]),
    );
    if (!fractional_origin || !fractional_tip) return null;
    const fractional_direction = fractional_tip.map(
      (value, axis) => value - fractional_origin[axis],
    );
    let entry = 0;
    let exit = Number.POSITIVE_INFINITY;
    for (let axis = 0; axis < 3; axis += 1) {
      const origin = fractional_origin[axis];
      const direction = fractional_direction[axis];
      if (Math.abs(direction) < 1e-12) {
        if (origin < 0 || origin > 1) return null;
        continue;
      }
      const near = Math.min((0 - origin) / direction, (1 - origin) / direction);
      const far = Math.max((0 - origin) / direction, (1 - origin) / direction);
      entry = Math.max(entry, near);
      exit = Math.min(exit, far);
      if (entry > exit) return null;
    }
    if (!Number.isFinite(entry) || !Number.isFinite(exit) || exit < 0) return null;
    const start = fractional_origin.map(
      (value, axis) => value + Math.max(0, entry) * fractional_direction[axis],
    );
    const end = fractional_origin.map(
      (value, axis) => value + exit * fractional_direction[axis],
    );
    return { start, end };
  };

  const establish_point_ray = (client_x, client_y) => {
    const canvas = host_element?.querySelector("canvas");
    const camera = canvas ? scene_registry.get(canvas)?.camera : undefined;
    if (!canvas || !camera || !structure?.lattice?.matrix) {
      on_error?.("The 3D picker needs a periodic structure and an active camera.");
      return;
    }
    const canvas_rect = canvas.getBoundingClientRect();
    const ndc = [
      ((client_x - canvas_rect.left) / canvas_rect.width) * 2 - 1,
      1 - ((client_y - canvas_rect.top) / canvas_rect.height) * 2,
    ];
    camera.updateMatrixWorld?.(true);
    camera.updateProjectionMatrix?.();
    const ray_origin = camera.isOrthographicCamera
      ? camera.position.clone().set(ndc[0], ndc[1], -1).unproject(camera)
      : camera.position.clone();
    const ray_direction = camera.isOrthographicCamera
      ? camera.getWorldDirection(camera.position.clone()).normalize()
      : camera.position.clone().set(ndc[0], ndc[1], 0.5).unproject(camera)
        .sub(camera.position).normalize();
    const ray = ray_cell_segment(
      [ray_origin.x, ray_origin.y, ray_origin.z],
      [ray_direction.x, ray_direction.y, ray_direction.z],
    );
    if (!ray) {
      on_error?.("That sightline misses the unit cell. Click inside the visible cell.");
      return;
    }
    point_pick_ray = ray;
    set_point_depth(0.5);
  };

  const establish_center_ray = () => {
    const canvas = host_element?.querySelector("canvas");
    if (!canvas) {
      on_error?.("The 3D picker could not access the active canvas.");
      return;
    }
    const rect = canvas.getBoundingClientRect();
    establish_point_ray(rect.left + rect.width / 2, rect.top + rect.height / 2);
  };

  function begin_gesture(event) {
    if (!selection_gesture || !host_element) return;
    if (selection_gesture === "point") {
      establish_point_ray(event.clientX, event.clientY);
      return;
    }
    event.currentTarget.setPointerCapture?.(event.pointerId);
    gesture_points = [event_point(event)];
    gesture_active = true;
  }

  function move_gesture(event) {
    if (!gesture_active) return;
    const point = event_point(event);
    if (selection_gesture === "box") {
      gesture_points = [gesture_points[0], point];
      return;
    }
    const previous = gesture_points.at(-1);
    if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 3) {
      gesture_points = [...gesture_points, point];
    }
  }

  function finish_gesture(event, force_box = false) {
    if (!gesture_active || !host_element) return;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    const final_point = event_point(event);
    const points = selection_gesture === "box" || force_box
      ? [gesture_points[0], final_point]
      : [...gesture_points, final_point];
    gesture_active = false;
    gesture_points = [];
    const canvas = host_element.querySelector("canvas");
    const camera = canvas ? scene_registry.get(canvas)?.camera : undefined;
    if (!canvas || !camera || !structure?.sites?.length) {
      on_error?.("The selection tool could not access the active 3D camera.");
      return;
    }
    camera.updateMatrixWorld?.(true);
    camera.updateProjectionMatrix?.();
    const host_rect = host_element.getBoundingClientRect();
    const canvas_rect = canvas.getBoundingClientRect();
    const left = Math.min(points[0][0], points.at(-1)[0]);
    const right = Math.max(points[0][0], points.at(-1)[0]);
    const top = Math.min(points[0][1], points.at(-1)[1]);
    const bottom = Math.max(points[0][1], points.at(-1)[1]);
    const polygon = selection_gesture === "lasso" && !force_box ? points : null;
    const selected = [];
    for (const [site_index, site] of structure.sites.entries()) {
      if (!Array.isArray(site?.xyz) || site.xyz.length !== 3) continue;
      const projected = camera.position.clone().set(...site.xyz).project(camera);
      if (projected.z < -1 || projected.z > 1) continue;
      const screen = [
        canvas_rect.left - host_rect.left + ((projected.x + 1) / 2) * canvas_rect.width,
        canvas_rect.top - host_rect.top + ((1 - projected.y) / 2) * canvas_rect.height,
      ];
      const included = polygon
        ? point_in_polygon(screen, polygon)
        : screen[0] >= left && screen[0] <= right && screen[1] >= top && screen[1] <= bottom;
      if (included) selected.push(site_index);
    }
    selected_sites = event.shiftKey
      ? [...new Set([...selected_sites, ...selected])].sort((a, b) => a - b)
      : selected;
  }

  const apply_keyboard_box = (add = false) => {
    if (!host_element) return;
    const rect = host_element.getBoundingClientRect();
    const [left, top, right, bottom] = keyboard_box;
    const synthetic = {
      clientX: rect.left + right * rect.width,
      clientY: rect.top + bottom * rect.height,
      shiftKey: add,
      currentTarget: { releasePointerCapture: () => {} },
      pointerId: -1,
    };
    gesture_points = [[left * rect.width, top * rect.height]];
    gesture_active = true;
    finish_gesture(synthetic, true);
  };

  const handle_overlay_keydown = (event) => {
    if (selection_gesture === "point") return;
    const [left, top, right, bottom] = keyboard_box;
    const step = event.altKey ? 0.01 : 0.04;
    let next = [left, top, right, bottom];
    if (event.key === "ArrowLeft") next = event.shiftKey
      ? [left, top, Math.max(left + 0.04, right - step), bottom]
      : [Math.max(0, left - step), top, Math.max(right - step, right - left), bottom];
    if (event.key === "ArrowRight") next = event.shiftKey
      ? [left, top, Math.min(1, right + step), bottom]
      : [Math.min(left + step, 1 - (right - left)), top, Math.min(1, right + step), bottom];
    if (event.key === "ArrowUp") next = event.shiftKey
      ? [left, top, right, Math.max(top + 0.04, bottom - step)]
      : [left, Math.max(0, top - step), right, Math.max(bottom - step, bottom - top)];
    if (event.key === "ArrowDown") next = event.shiftKey
      ? [left, top, right, Math.min(1, bottom + step)]
      : [left, Math.min(top + step, 1 - (bottom - top)), right, Math.min(1, bottom + step)];
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      apply_keyboard_box(event.ctrlKey || event.metaKey);
      return;
    }
    if (next.some((value, index) => value !== keyboard_box[index])) {
      event.preventDefault();
      keyboard_box = next;
    }
  };

  let slice = $derived.by(() => {
    const active = volumetric_data?.[active_volume_idx];
    if (!slice_enabled || !active) return null;
    try {
      if (slice_mode === "cartesian") {
        return sample_plane_slice(
          active,
          { point: slice_point, normal: slice_normal },
          { resolution: [240, 240], max_pixels: 240 * 240 },
        );
      }
      return sample_hkl_slice(active, slice_hkl, Number(slice_distance), 240);
    } catch (error) {
      on_error?.(error instanceof Error ? error.message : String(error));
      return null;
    }
  });

  $effect(() => {
    controls_open = !read_only;
  });

  const apply_representation = (next_representation) => {
    const mode = ["ball-stick", "spacefill", "wireframe", "polyhedra"].includes(next_representation)
      ? next_representation
      : "ball-stick";
    if (large_structure) {
      Object.assign(scene_props, {
        show_atoms: true,
        show_image_atoms: false,
        show_bonds: "never",
        show_polyhedra: "never",
        show_site_labels: false,
        show_site_indices: false,
        sphere_segments: 8,
        atom_radius: 0.45,
      });
      return;
    }
    const presets = {
      "ball-stick": {
        show_atoms: true,
        show_bonds: "always",
        show_polyhedra: "never",
        atom_radius: 0.35,
        bond_thickness: 0.08,
      },
      spacefill: {
        show_atoms: true,
        show_bonds: "never",
        show_polyhedra: "never",
        atom_radius: 1,
      },
      wireframe: {
        show_atoms: false,
        show_bonds: "always",
        show_polyhedra: "never",
        bond_thickness: 0.025,
      },
      polyhedra: {
        show_atoms: true,
        show_bonds: "crystals",
        show_polyhedra: "always",
        atom_radius: 0.25,
      },
    };
    Object.assign(scene_props, {
      show_image_atoms: true,
      sphere_segments: 20,
      ...presets[mode],
    });
  };

  $effect(() => {
    apply_representation(representation);
  });

  $effect(() => {
    const current = structure;
    const current_bonds = bonds;
    if (!initialized) {
      initialized = true;
      return;
    }
    if (replacing_structure) {
      replacing_structure = false;
      return;
    }
    if (current && !read_only) {
      on_structure_change?.(
        $state.snapshot(current),
        current_bonds ? $state.snapshot(current_bonds) : undefined,
      );
    }
  });

  $effect(() => {
    const selected = selected_sites;
    const measured = measured_sites;
    if (!selection_initialized) {
      selection_initialized = true;
      return;
    }
    if (replacing_selection) {
      replacing_selection = false;
      return;
    }
    on_selection_change?.($state.snapshot(selected), $state.snapshot(measured));
  });

  export function replace_structure(next_structure, next_bonds = undefined) {
    replacing_structure = true;
    structure = structuredClone(next_structure);
    bonds = next_bonds ? structuredClone(next_bonds) : next_structure?.properties?.bonds;
  }

  export function replace_selection(indices) {
    replacing_selection = true;
    selected_sites = [...indices];
  }

  export function replace_read_only(next_read_only) {
    read_only = Boolean(next_read_only);
    if (read_only) {
      selection_gesture = "";
      gesture_active = false;
      gesture_points = [];
      point_pick_ray = null;
    }
  }

  export function replace_large_structure(next_large_structure) {
    large_structure = Boolean(next_large_structure);
  }

  export function replace_volume(next_volume_data) {
    volumetric_data = next_volume_data ? structuredClone(next_volume_data) : undefined;
    active_volume_idx = 0;
  }

  export function replace_slice(next_slice) {
    slice_enabled = Boolean(next_slice?.enabled);
    slice_mode = next_slice?.mode || "hkl";
    slice_hkl = structuredClone(next_slice?.hkl || [0, 0, 1]);
    slice_distance = Number(next_slice?.distance ?? 0.5);
    slice_point = structuredClone(next_slice?.point || [0, 0, 0]);
    slice_normal = structuredClone(next_slice?.normal || [0, 0, 1]);
  }

  export function replace_representation(next_representation) {
    representation = next_representation;
  }

  export function replace_selection_gesture(next_gesture) {
    selection_gesture = !read_only && ["box", "lasso", "point"].includes(next_gesture)
      ? next_gesture
      : "";
    gesture_active = false;
    gesture_points = [];
    point_pick_ray = null;
  }

  export function replace_view_direction(next_direction) {
    view_direction = Array.isArray(next_direction) && next_direction.length === 3
      ? structuredClone(next_direction)
      : undefined;
  }
</script>

<div
  bind:this={host_element}
  class="catmaster-matterviz-host"
  aria-label={read_only
    ? "Interactive three-dimensional structure preview"
    : "Interactive three-dimensional structure editor"}
>
  <Structure
    bind:structure
    bind:bonds
    bind:selected_sites
    bind:measured_sites
    bind:volumetric_data
    bind:active_volume_idx
    bind:isosurface_settings
    bind:scene_props
    show_controls={read_only
      ? { mode: "always", hidden: ["measure-mode", "controls"] }
      : "always"}
    bind:controls_open
    enable_measure_mode={!read_only}
    enable_info_pane={true}
    views={custom_views}
    allow_file_drop={false}
    performance_mode={large_structure || (structure?.sites?.length ?? 0) > 5000 ? "speed" : "quality"}
    on_error={(event) => on_error?.(event?.error_msg || "The 3D renderer could not display this structure.")}
    on_bonds_change={(next_bonds) => {
      if (!read_only) bonds = next_bonds;
    }}
  />
  {#if selection_gesture && !read_only}
    <div
      bind:this={overlay_element}
      class="catmaster-selection-overlay"
      role="application"
      tabindex="0"
      aria-label={selection_gesture === "point"
        ? "Interstitial position picker. Choose a sightline through the unit cell, then set its depth or exact fractional coordinates."
        : `${selection_gesture === "box" ? "Box" : "Lasso"} atom selection. Drag over atoms, or use the keyboard selection rectangle.`}
      onpointerdown={begin_gesture}
      onpointermove={move_gesture}
      onpointerup={finish_gesture}
      onkeydown={handle_overlay_keydown}
      onpointercancel={() => {
        gesture_active = false;
        gesture_points = [];
      }}
    >
      {#if selection_gesture === "point"}
        <section
          class="catmaster-point-picker"
          aria-label="Exact three-dimensional interstitial coordinates"
          onpointerdown={(event) => event.stopPropagation()}
          onpointermove={(event) => event.stopPropagation()}
          onpointerup={(event) => event.stopPropagation()}
        >
          <strong>Interstitial position in the unit cell</strong>
          <p>
            Click the structure to choose a real sightline through the cell. Adjust depth
            along that segment, or enter exact fractional coordinates.
          </p>
          <label>
            Sightline depth
            <input
              type="range"
              min="0"
              max="1"
              step="0.001"
              value={point_pick_depth}
              disabled={!point_pick_ray}
              aria-label="Interstitial depth along the selected unit-cell sightline"
              oninput={(event) => set_point_depth(event.currentTarget.value)}
            />
            <output>{(point_pick_depth * 100).toFixed(1)}%</output>
          </label>
          <div class="catmaster-point-coordinates" aria-label="Fractional interstitial coordinates">
            {#each ["a", "b", "c"] as axis, index (axis)}
              <label>
                {axis}
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.001"
                  value={point_pick_fractional[index]}
                  aria-label={`Fractional ${axis} coordinate`}
                  oninput={(event) => {
                    const next = [...point_pick_fractional];
                    next[index] = Math.min(1, Math.max(0, Number(event.currentTarget.value)));
                    point_pick_fractional = next;
                    point_pick_ray = null;
                  }}
                />
              </label>
            {/each}
          </div>
          <output class="catmaster-point-output" aria-live="polite">
            Fractional [{point_pick_fractional.map((value) => Number(value).toFixed(4)).join(", ")}]
            {#if fractional_to_cartesian(structure?.lattice?.matrix, point_pick_fractional)}
              · Cartesian [{fractional_to_cartesian(
                structure?.lattice?.matrix,
                point_pick_fractional,
              ).map((value) => Number(value).toFixed(4)).join(", ")}] Å
            {/if}
          </output>
          <div class="catmaster-point-actions">
            <button type="button" onclick={establish_center_ray}>Use centre sightline</button>
            <button
              type="button"
              onclick={() => {
                const cartesian = fractional_to_cartesian(
                  structure?.lattice?.matrix,
                  point_pick_fractional,
                );
                if (cartesian) on_point_pick?.(cartesian);
                else on_error?.("The interstitial coordinates require a non-singular lattice.");
              }}
            >Use this position</button>
          </div>
        </section>
      {:else}
        <section
          class="catmaster-selection-help"
          role="group"
          aria-label="Keyboard and pointer selection instructions"
          onpointerdown={(event) => event.stopPropagation()}
          onpointermove={(event) => event.stopPropagation()}
          onpointerup={(event) => event.stopPropagation()}
        >
          <strong>{selection_gesture === "box" ? "Box" : "Lasso"} atom selection</strong>
          <span>Drag over atoms; hold Shift to add.</span>
          <span>
            Keyboard: arrows move the rectangle, Shift+arrows resize, Alt makes a fine
            adjustment, Enter applies, Ctrl/⌘+Enter adds.
          </span>
          <button type="button" onclick={() => apply_keyboard_box(false)}>
            Apply keyboard rectangle
          </button>
        </section>
        <div
          class="catmaster-keyboard-box"
          aria-hidden="true"
          style={`left:${keyboard_box[0] * 100}%;top:${keyboard_box[1] * 100}%;width:${(keyboard_box[2] - keyboard_box[0]) * 100}%;height:${(keyboard_box[3] - keyboard_box[1]) * 100}%`}
        ></div>
      {/if}
      {#if selection_gesture !== "point" && gesture_active && gesture_points.length > 1}
        <svg aria-hidden="true">
          {#if selection_gesture === "box"}
            {@const start = gesture_points[0]}
            {@const end = gesture_points.at(-1)}
            <rect
              x={Math.min(start[0], end[0])}
              y={Math.min(start[1], end[1])}
              width={Math.abs(end[0] - start[0])}
              height={Math.abs(end[1] - start[1])}
            />
          {:else}
            <polyline points={gesture_points.map((point) => point.join(",")).join(" ")} />
          {/if}
        </svg>
      {/if}
    </div>
  {/if}
  {#if slice}
    <aside class="catmaster-volume-slice" aria-label="Crystallographic volume slice">
      <VolumeSlice
        {slice}
        mode="heatmap-contour"
        colormap="interpolateRdBu"
        symmetric="auto"
        show_colorbar={true}
        colorbar_title="Field value"
      />
    </aside>
  {/if}
</div>

<style>
  .catmaster-matterviz-host {
    position: relative;
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
    background: var(--v2-panel, #ffffff);
  }

  .catmaster-matterviz-host :global(> div) {
    width: 100%;
    height: 100%;
  }

  .catmaster-matterviz-host :global(button[data-original-title]) {
    color: var(--v2-text, #1f2937) !important;
  }

  .catmaster-volume-slice {
    position: absolute;
    right: 0.75rem;
    bottom: 0.75rem;
    width: min(24rem, 42%);
    max-height: 42%;
    padding: 0.5rem;
    border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    border-radius: 0.65rem;
    background: color-mix(in srgb, var(--v2-panel, #fff) 94%, transparent);
    box-shadow: 0 0.4rem 1.2rem rgb(0 0 0 / 16%);
    overflow: auto;
  }

  .catmaster-selection-overlay {
    position: absolute;
    inset: 0;
    z-index: 40;
    cursor: crosshair;
    touch-action: none;
    user-select: none;
  }

  .catmaster-selection-overlay:focus-visible {
    outline: 3px solid #2563eb;
    outline-offset: -3px;
  }

  .catmaster-point-picker,
  .catmaster-selection-help {
    position: absolute;
    left: 50%;
    top: 0.75rem;
    transform: translateX(-50%);
    z-index: 2;
    width: min(38rem, calc(100% - 1.5rem));
    max-height: calc(100% - 1.5rem);
    overflow: auto;
    padding: 0.65rem 0.75rem;
    border: 1px solid rgb(148 163 184 / 75%);
    border-radius: 0.65rem;
    color: #f8fafc;
    background: rgb(15 23 42 / 94%);
    box-shadow: 0 0.5rem 1.5rem rgb(0 0 0 / 30%);
    font: 500 12px/1.35 system-ui, sans-serif;
  }

  .catmaster-point-picker {
    display: grid;
    gap: 0.5rem;
  }

  .catmaster-point-picker p {
    margin: 0;
    color: #dbeafe;
  }

  .catmaster-point-picker > label {
    display: grid;
    grid-template-columns: auto minmax(8rem, 1fr) auto;
    align-items: center;
    gap: 0.5rem;
  }

  .catmaster-point-picker input {
    min-width: 0;
  }

  .catmaster-point-coordinates {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.4rem;
  }

  .catmaster-point-coordinates label {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: 0.3rem;
  }

  .catmaster-point-coordinates input {
    width: 100%;
    border: 1px solid #64748b;
    border-radius: 0.3rem;
    padding: 0.25rem 0.35rem;
    color: #0f172a;
    background: #fff;
  }

  .catmaster-point-output {
    overflow-wrap: anywhere;
    color: #bfdbfe;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .catmaster-point-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.45rem;
    flex-wrap: wrap;
  }

  .catmaster-point-actions button,
  .catmaster-selection-help button {
    min-height: 2rem;
    border: 1px solid #94a3b8;
    border-radius: 0.35rem;
    padding: 0.3rem 0.55rem;
    color: #f8fafc;
    background: #1e3a5f;
  }

  .catmaster-point-actions button:focus-visible,
  .catmaster-selection-help button:focus-visible,
  .catmaster-point-picker input:focus-visible {
    outline: 2px solid #93c5fd;
    outline-offset: 2px;
  }

  .catmaster-selection-help {
    display: flex;
    align-items: center;
    gap: 0.45rem 0.7rem;
    flex-wrap: wrap;
  }

  .catmaster-selection-help span {
    color: #dbeafe;
  }

  .catmaster-keyboard-box {
    position: absolute;
    z-index: 1;
    border: 2px solid #f59e0b;
    background: rgb(245 158 11 / 9%);
    pointer-events: none;
  }

  .catmaster-selection-overlay svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
    pointer-events: none;
  }

  .catmaster-selection-overlay rect,
  .catmaster-selection-overlay polyline {
    fill: rgb(37 99 235 / 12%);
    stroke: #2563eb;
    stroke-width: 1.5;
    stroke-dasharray: 5 3;
  }

  .catmaster-selection-overlay polyline {
    fill: none;
  }

  @media (max-width: 520px) {
    .catmaster-point-picker,
    .catmaster-selection-help {
      top: 0.4rem;
      width: calc(100% - 0.8rem);
      max-height: calc(100% - 0.8rem);
    }

    .catmaster-point-picker > label {
      grid-template-columns: 1fr auto;
    }

    .catmaster-point-picker > label input {
      grid-column: 1 / -1;
    }

    .catmaster-point-coordinates {
      grid-template-columns: 1fr;
    }
  }
</style>
