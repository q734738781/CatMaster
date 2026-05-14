#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, JmolNN, MinimumDistanceNN
from pymatgen.core import Composition, Structure
from pymatgen.io.vasp import Poscar


@dataclass(frozen=True)
class Fragment:
    frag_id: int
    indices: tuple[int, ...]
    natoms: int
    composition: Composition
    formula: str
    reduced_formula: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify connected fragments in a periodic structure and generate a "
            "Selective Dynamics POSCAR that frees only the chosen fragment(s)."
        )
    )
    parser.add_argument("structure", help="Input structure file (.cif, POSCAR/CONTCAR, .vasp).")
    parser.add_argument(
        "--strategy",
        choices=("jmolnn", "crystalnn", "mindistance"),
        default="jmolnn",
        help="Connectivity strategy. jmolnn is the most reliable default for the sample zeolite systems.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Prefix for output files. Default: input file stem in the current directory.",
    )
    parser.add_argument(
        "--select-id",
        nargs="+",
        type=int,
        help="Fragment ID(s) to free. IDs are shown in the fragment table.",
    )
    parser.add_argument(
        "--match-natoms",
        type=int,
        help="Free fragments whose atom count matches this value.",
    )
    parser.add_argument(
        "--match-formula",
        help=(
            "Free fragments whose exact formula matches this value, e.g. C8H10 or H13C9. "
            "This is an exact full-composition match, not a reduced formula match."
        ),
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="Additional 1-based atom indices to free together with any selected fragments.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print fragment summary and exit.",
    )
    parser.add_argument(
        "--write-all-fragments",
        action="store_true",
        help="Write every connected fragment as an individual CIF.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not prompt for a selection when no --select-* option is given.",
    )
    parser.set_defaults(inplace=True)
    parser.add_argument(
        "--inplace",
        dest="inplace",
        action="store_true",
        help=(
            "Overwrite the input VASP structure file with the generated Selective "
            "Dynamics POSCAR and keep a sibling backup named <input>_old. "
            "This is the default behavior."
        ),
    )
    parser.add_argument(
        "--no-inplace",
        dest="inplace",
        action="store_false",
        help="Write a new <stem>_free_selected.vasp instead of overwriting the input file.",
    )
    return parser.parse_args()


def build_structure_graph(structure: Structure, strategy: str) -> StructureGraph:
    if strategy == "jmolnn":
        nn_strategy = JmolNN()
    elif strategy == "crystalnn":
        nn_strategy = CrystalNN()
    elif strategy == "mindistance":
        nn_strategy = MinimumDistanceNN()
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return StructureGraph.from_local_env_strategy(structure, nn_strategy)


def hill_formula(comp: Composition) -> str:
    return comp.hill_formula.replace(" ", "")


def normalize_formula(formula: str) -> tuple[tuple[str, int], ...]:
    parsed = Composition(formula)
    amounts: list[tuple[str, int]] = []
    for el, amt in parsed.get_el_amt_dict().items():
        rounded = round(amt)
        if abs(amt - rounded) > 1e-8:
            raise ValueError(f"Formula contains a non-integer amount for {el}: {amt}")
        amounts.append((el, int(rounded)))
    return tuple(sorted(amounts))


def extract_fragments(structure: Structure, strategy: str) -> list[Fragment]:
    sg = build_structure_graph(structure, strategy)
    graph = sg.graph.to_undirected()
    fragments: list[Fragment] = []

    for frag_id, component in enumerate(
        sorted(nx.connected_components(graph), key=len, reverse=True), start=1
    ):
        indices = tuple(sorted(component))
        counts = Counter(str(structure[idx].specie) for idx in indices)
        comp = Composition(counts)
        fragments.append(
            Fragment(
                frag_id=frag_id,
                indices=indices,
                natoms=len(indices),
                composition=comp,
                formula=hill_formula(comp),
                reduced_formula=comp.reduced_formula.replace(" ", ""),
            )
        )
    return fragments


def format_indices(indices: Iterable[int]) -> str:
    indices = list(indices)
    if not indices:
        return "-"
    if len(indices) <= 8:
        return ",".join(str(i + 1) for i in indices)
    head = ",".join(str(i + 1) for i in indices[:4])
    tail = ",".join(str(i + 1) for i in indices[-4:])
    return f"{head},...,{tail}"


def print_fragment_table(fragments: list[Fragment]) -> None:
    print("Fragments:")
    print("  ID  natoms  formula                 reduced       atom_indices(1-based)")
    for frag in fragments:
        print(
            f"  {frag.frag_id:>2}  {frag.natoms:>6}  {frag.formula:<22}  "
            f"{frag.reduced_formula:<12}  {format_indices(frag.indices)}"
        )


def select_fragments(
    fragments: list[Fragment],
    select_ids: list[int] | None,
    match_natoms: int | None,
    match_formula: str | None,
) -> list[Fragment]:
    formula_key = normalize_formula(match_formula) if match_formula else None
    selected: list[Fragment] = []
    for frag in fragments:
        if select_ids and frag.frag_id not in select_ids:
            continue
        if match_natoms is not None and frag.natoms != match_natoms:
            continue
        if formula_key is not None and normalize_formula(frag.formula) != formula_key:
            continue
        if select_ids or match_natoms is not None or formula_key is not None:
            selected.append(frag)
    return selected


def prompt_for_selection(fragments: list[Fragment]) -> tuple[list[Fragment], list[int]]:
    print()
    print("Selection examples:")
    print("  id=2")
    print("  id=2,3")
    print("  natoms=18")
    print("  formula=C8H10")
    print("  indices=80,81,82")
    print("  natoms=18 formula=C8H10 indices=80,81,82")
    raw = input("Choose fragment(s) to free (empty input exits): ").strip()
    if not raw:
        return [], []

    select_ids: list[int] | None = None
    match_natoms: int | None = None
    match_formula: str | None = None
    manual_indices: list[int] = []

    for token in raw.replace(";", " ").split():
        if "=" not in token:
            raise ValueError(f"Unrecognized token: {token}")
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "id":
            select_ids = [int(item) for item in value.split(",") if item]
        elif key == "natoms":
            match_natoms = int(value)
        elif key == "formula":
            match_formula = value
        elif key == "indices":
            manual_indices = [int(item) for item in value.split(",") if item]
        else:
            raise ValueError(f"Unsupported selector: {key}")

    return select_fragments(fragments, select_ids, match_natoms, match_formula), manual_indices


def collect_free_indices(
    selected: list[Fragment],
    manual_indices_1_based: list[int] | None,
    natoms: int,
) -> set[int]:
    free_indices = {idx for frag in selected for idx in frag.indices}
    for idx_1_based in manual_indices_1_based or []:
        if idx_1_based < 1 or idx_1_based > natoms:
            raise ValueError(f"Atom index out of range: {idx_1_based} (valid range: 1-{natoms})")
        free_indices.add(idx_1_based - 1)
    return free_indices


def build_substructure(structure: Structure, indices: Iterable[int]) -> Structure:
    idx_list = list(indices)
    species = [structure[idx].species for idx in idx_list]
    frac_coords = [structure[idx].frac_coords for idx in idx_list]
    site_properties = {
        name: [values[idx] for idx in idx_list]
        for name, values in structure.site_properties.items()
    }
    return Structure(
        lattice=structure.lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
        site_properties=site_properties,
        charge=structure.charge,
    )


def selected_part_label(structure_path: Path) -> str:
    return f"{structure_path.name}_PART"


def write_fragment_cifs(
    structure: Structure,
    fragments: list[Fragment],
    output_prefix: Path,
) -> list[Path]:
    written: list[Path] = []
    for frag in fragments:
        sub = build_substructure(structure, frag.indices)
        out_path = output_prefix.with_name(
            f"{output_prefix.name}_fragment{frag.frag_id:02d}_{frag.formula}.cif"
        )
        sub.to(filename=str(out_path))
        written.append(out_path)
    return written


def write_selected_part_cif(
    structure: Structure,
    free_indices: set[int],
    structure_path: Path,
) -> tuple[Path, Structure]:
    selected_part = build_substructure(structure, sorted(free_indices))
    out_path = structure_path.with_name(f"{selected_part_label(structure_path)}.cif")
    selected_part.to(filename=str(out_path))
    return out_path, selected_part


def write_selective_dynamics_poscar(
    structure: Structure,
    free_indices: set[int],
    output_prefix: Path,
) -> Path:
    mask = [[idx in free_indices] * 3 for idx in range(len(structure))]
    out_path = output_prefix.with_name(f"{output_prefix.name}_free_selected.vasp")
    poscar = Poscar(structure, selective_dynamics=mask)
    poscar.write_file(str(out_path))
    return out_path


def write_inplace_selective_dynamics_poscar(
    structure: Structure,
    free_indices: set[int],
    structure_path: Path,
) -> tuple[Path, Path]:
    if structure_path.suffix.lower() == ".cif":
        raise ValueError("--inplace can only overwrite VASP structure files, not CIF inputs.")

    backup_path = structure_path.with_name(f"{structure_path.name}_old")
    backup_path.write_text(structure_path.read_text(encoding="utf-8"), encoding="utf-8")

    mask = [[idx in free_indices] * 3 for idx in range(len(structure))]
    poscar = Poscar(structure, selective_dynamics=mask)
    poscar.write_file(str(structure_path))
    return structure_path, backup_path


def main() -> int:
    args = parse_args()
    structure_path = Path(args.structure).resolve()
    structure = Structure.from_file(structure_path)
    fragments = extract_fragments(structure, args.strategy)
    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else structure_path.with_name(structure_path.name)
    )

    print(f"Input: {structure_path}")
    print(f"Strategy: {args.strategy}")
    print(f"Total atoms: {len(structure)}")
    print_fragment_table(fragments)

    if args.write_all_fragments:
        written = write_fragment_cifs(structure, fragments, output_prefix)
        print()
        print("Wrote fragment CIFs:")
        for path in written:
            print(f"  {path}")

    if args.list_only:
        return 0

    selected = select_fragments(
        fragments=fragments,
        select_ids=args.select_id,
        match_natoms=args.match_natoms,
        match_formula=args.match_formula,
    )
    manual_indices = args.indices or []

    should_prompt = (
        not selected
        and not manual_indices
        and not args.no_interactive
        and not any([args.select_id, args.match_natoms is not None, args.match_formula, args.indices])
        and sys.stdin.isatty()
    )
    if should_prompt:
        try:
            selected, manual_indices = prompt_for_selection(fragments)
        except ValueError as exc:
            print(f"Selection error: {exc}", file=sys.stderr)
            return 2

    try:
        free_indices = collect_free_indices(selected, manual_indices, len(structure))
    except ValueError as exc:
        print(f"Selection error: {exc}", file=sys.stderr)
        return 2

    if not free_indices:
        print()
        print("No fragment or atom indices selected. Exiting without writing Selective Dynamics POSCAR.")
        return 0

    if args.inplace:
        try:
            poscar_path, backup_path = write_inplace_selective_dynamics_poscar(
                structure, free_indices, structure_path
            )
        except ValueError as exc:
            print(f"In-place write error: {exc}", file=sys.stderr)
            return 2
    else:
        poscar_path = write_selective_dynamics_poscar(structure, free_indices, output_prefix)
        backup_path = None
    selected_part_path, selected_part = write_selected_part_cif(structure, free_indices, structure_path)

    print()
    print("Selected fragments:")
    if selected:
        for frag in selected:
            print(f"  ID={frag.frag_id} natoms={frag.natoms} formula={frag.formula}")
    else:
        print("  (none)")
    if manual_indices:
        print(f"Additional free atom indices (1-based): {','.join(str(idx) for idx in manual_indices)}")
    print(
        "Selected part:"
        f" natoms={len(selected_part)} formula={hill_formula(selected_part.composition)}"
    )
    print(f"Selective Dynamics POSCAR: {poscar_path}")
    if backup_path is not None:
        print(f"Original structure backup: {backup_path}")
    print(f"Selected part CIF: {selected_part_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
