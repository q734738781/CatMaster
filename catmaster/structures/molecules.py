from __future__ import annotations

import math
from typing import Any

from .serialization import (
    StructureSerializationError,
    derived_summary,
    snapshot_from_molecule,
    snapshot_to_molecule,
    viewer_structure,
)


def _viewer_element(site: dict[str, Any]) -> str:
    species = site.get("species") or []
    element = species[0].get("element") if species and isinstance(species[0], dict) else site.get("label")
    symbol = str(element or "").strip()
    if not symbol:
        raise ValueError("Every molecular site needs an element.")
    return symbol


def _viewer_bond_type(order: Any):
    from rdkit import Chem

    if str(order).lower() == "aromatic":
        return Chem.BondType.AROMATIC
    try:
        numeric = float(order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported molecular bond order: {order}") from exc
    mapping = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
    }
    if numeric not in mapping:
        raise ValueError(f"Unsupported molecular bond order: {order}")
    return mapping[numeric]


def _copy_document_molecule_properties(source: Any, target: Any) -> None:
    """Copy persisted molecule properties while excluding computed RDKit caches."""
    values = source.GetPropsAsDict(
        includePrivate=True,
        includeComputed=False,
        autoConvertStrings=False,
    )
    for name in source.GetPropNames(
        includePrivate=True,
        includeComputed=False,
    ):
        key = str(name)
        value = values[key]
        if isinstance(value, bool):
            target.SetBoolProp(key, value, computed=False)
        elif isinstance(value, int):
            try:
                target.SetIntProp(key, value, computed=False)
            except OverflowError:
                target.SetProp(key, source.GetProp(key), computed=False)
        elif isinstance(value, float):
            target.SetDoubleProp(key, value, computed=False)
        elif isinstance(value, str):
            target.SetProp(key, value, computed=False)
        else:
            # RDKit exposes some vector-valued properties through
            # GetPropsAsDict but SDF data fields are textual.  GetProp is the
            # lossless textual representation exposed by RDKit's public API.
            target.SetProp(key, source.GetProp(key), computed=False)


def rebuild_molecule_from_viewer(snapshot: Any, viewer: dict[str, Any]):
    """Apply the 3D editor view to an RDKit connection table without losing chemistry."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    original = snapshot_to_molecule(snapshot)
    sites = list(viewer.get("sites") or [])
    if not sites:
        raise ValueError("The 3D molecule contains no atoms.")

    editable = Chem.RWMol()
    source_to_new: dict[int, int] = {}
    for site in sites:
        properties = dict(site.get("properties") or {})
        source_index = properties.get("_catmaster_mol_atom_index")
        source_atom = None
        if isinstance(source_index, int) and 0 <= source_index < original.GetNumAtoms():
            candidate = original.GetAtomWithIdx(source_index)
            if candidate.GetSymbol() == _viewer_element(site):
                source_atom = candidate
        if source_atom is not None:
            atom = Chem.Atom(source_atom)
        else:
            atom = Chem.Atom(_viewer_element(site))
            charge = properties.get("formal_charge", 0)
            isotope = properties.get("isotope", 0)
            atom.SetFormalCharge(int(charge or 0))
            atom.SetIsotope(int(isotope or 0))
        new_index = int(editable.AddAtom(atom))
        if source_atom is not None and int(source_index) not in source_to_new:
            source_to_new[int(source_index)] = new_index

    viewer_bonds = list((viewer.get("properties") or {}).get("bonds") or [])
    seen_pairs: set[tuple[int, int]] = set()
    for record in viewer_bonds:
        left = int(record.get("site_idx_1", -1))
        right = int(record.get("site_idx_2", -1))
        if left < 0 or right < 0 or left >= len(sites) or right >= len(sites) or left == right:
            raise ValueError("The 3D molecule contains a bond with invalid atom indices.")
        pair = tuple(sorted((left, right)))
        if pair in seen_pairs:
            raise ValueError(f"The 3D molecule contains a duplicate bond between atoms {left + 1} and {right + 1}.")
        seen_pairs.add(pair)
        bond_type = _viewer_bond_type(record.get("order", 1))
        editable.AddBond(left, right, bond_type)
        new_bond = editable.GetBondBetweenAtoms(left, right)

        source_bond = None
        source_bond_index = record.get("_catmaster_mol_bond_index")
        if isinstance(source_bond_index, int) and 0 <= source_bond_index < original.GetNumBonds():
            source_bond = original.GetBondWithIdx(source_bond_index)
        if source_bond is None:
            left_source = (sites[left].get("properties") or {}).get("_catmaster_mol_atom_index")
            right_source = (sites[right].get("properties") or {}).get("_catmaster_mol_atom_index")
            if isinstance(left_source, int) and isinstance(right_source, int):
                source_bond = original.GetBondBetweenAtoms(left_source, right_source)

        if bond_type == Chem.BondType.AROMATIC:
            new_bond.SetIsAromatic(True)
            editable.GetAtomWithIdx(left).SetIsAromatic(True)
            editable.GetAtomWithIdx(right).SetIsAromatic(True)
        if source_bond is not None and source_bond.GetBondType() == bond_type:
            new_bond.SetBondDir(source_bond.GetBondDir())
            new_bond.SetIsConjugated(source_bond.GetIsConjugated())
            stereo_atoms = list(source_bond.GetStereoAtoms())
            if len(stereo_atoms) == 2 and all(index in source_to_new for index in stereo_atoms):
                new_bond.SetStereoAtoms(source_to_new[stereo_atoms[0]], source_to_new[stereo_atoms[1]])
            new_bond.SetStereo(source_bond.GetStereo())
            for name in source_bond.GetPropNames(includePrivate=True, includeComputed=False):
                new_bond.SetProp(name, source_bond.GetProp(name))

    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(sites))
    has_three_dimensional_coordinates = any(
        abs(float(value)) > 1e-8
        for site in sites
        for value in (site.get("xyz") or [])[2:3]
    )
    conformer.Set3D(has_three_dimensional_coordinates)
    for index, site in enumerate(sites):
        coordinates = list(site.get("xyz") or [])
        if len(coordinates) != 3:
            raise ValueError(f"Molecular atom {index + 1} is missing a three-dimensional coordinate.")
        values = [float(value) for value in coordinates]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Molecular atom {index + 1} has a non-finite coordinate.")
        conformer.SetAtomPosition(index, Point3D(*values))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise ValueError(f"The edited molecular valence or aromaticity is invalid: {exc}") from exc
    Chem.AssignStereochemistry(molecule, cleanIt=False, force=True)
    if has_three_dimensional_coordinates:
        original_centers = dict(
            Chem.FindMolChiralCenters(original, includeUnassigned=False, useLegacyImplementation=False)
        )
        if original_centers:
            geometry = Chem.Mol(molecule)
            for atom in geometry.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            Chem.AssignAtomChiralTagsFromStructure(
                geometry,
                confId=0,
                replaceExistingTags=True,
            )
            Chem.AssignStereochemistry(geometry, cleanIt=True, force=True)
            geometry_centers = dict(
                Chem.FindMolChiralCenters(
                    geometry,
                    includeUnassigned=False,
                    useLegacyImplementation=False,
                )
            )
            for source_index, desired_label in original_centers.items():
                new_index = source_to_new.get(int(source_index))
                if new_index is None:
                    continue
                source_atom = original.GetAtomWithIdx(int(source_index))
                new_atom = molecule.GetAtomWithIdx(new_index)
                mapped_neighbors = {
                    source_to_new.get(neighbor.GetIdx())
                    for neighbor in source_atom.GetNeighbors()
                }
                new_neighbors = {neighbor.GetIdx() for neighbor in new_atom.GetNeighbors()}
                same_local_topology = (
                    None not in mapped_neighbors
                    and mapped_neighbors == new_neighbors
                    and source_atom.GetSymbol() == new_atom.GetSymbol()
                    and source_atom.GetFormalCharge() == new_atom.GetFormalCharge()
                    and all(
                        original.GetBondBetweenAtoms(int(source_index), neighbor.GetIdx()).GetBondType()
                        == molecule.GetBondBetweenAtoms(
                            new_index,
                            source_to_new[neighbor.GetIdx()],
                        ).GetBondType()
                        for neighbor in source_atom.GetNeighbors()
                    )
                )
                if same_local_topology and geometry_centers.get(new_index) != desired_label:
                    raise StructureSerializationError(
                        f"The 3D edit inverted or flattened the configured stereochemistry at atom "
                        f"{new_index + 1}. Adjust the coordinates without crossing the chiral centre, "
                        "or change that stereochemistry explicitly in the 2D editor."
                    )
    _copy_document_molecule_properties(original, molecule)
    return molecule


def snapshot_from_viewer(snapshot: Any, viewer: dict[str, Any]):
    molecule = rebuild_molecule_from_viewer(snapshot, viewer)
    return snapshot_from_molecule(
        molecule,
        fmt=snapshot.format,
        path=snapshot.path,
        version=snapshot.source_version,
    )


def molecule_refresh(request: Any) -> dict[str, Any]:
    # Parsing the MolBlock through RDKit is intentional: Ketcher remains the
    # connection-table editor, while this projection feeds the 3D workbench.
    snapshot_to_molecule(request.input)
    return {
        "kind": "snapshot",
        "snapshot": request.input.model_dump(mode="json"),
        "summary": derived_summary(request.input),
        "viewer_structure": viewer_structure(request.input),
        "atom_mapping": list(range(derived_summary(request.input)["atom_count"])),
        "warnings": [],
        "change": {"source": "molecular connection table"},
    }


def molecule_from_viewer(request: Any) -> dict[str, Any]:
    snapshot = snapshot_from_viewer(request.input, request.params.viewer_structure)
    return {
        "kind": "snapshot",
        "snapshot": snapshot.model_dump(mode="json"),
        "summary": derived_summary(snapshot),
        "viewer_structure": viewer_structure(snapshot),
        "atom_mapping": list(range(derived_summary(snapshot)["atom_count"])),
        "warnings": [],
        "change": {"source": "three-dimensional editor"},
    }


def generate_conformers(
    molecule: Any,
    *,
    count: int,
    random_seed: int,
    optimize: str,
    prune_rms_threshold: float,
) -> list[tuple[Any, float | None]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    work = Chem.AddHs(Chem.Mol(molecule), addCoords=True)
    work.RemoveAllConformers()
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(random_seed)
    parameters.pruneRmsThresh = float(prune_rms_threshold)
    parameters.enforceChirality = True
    conformer_ids = list(AllChem.EmbedMultipleConfs(work, numConfs=int(count), params=parameters))
    if not conformer_ids:
        raise ValueError("RDKit could not generate a 3D conformer for this molecule.")

    candidates: list[tuple[Any, float | None]] = []
    for conformer_id in conformer_ids:
        energy: float | None = None
        if optimize == "mmff" and AllChem.MMFFHasAllMoleculeParams(work):
            properties = AllChem.MMFFGetMoleculeProperties(work)
            force_field = AllChem.MMFFGetMoleculeForceField(work, properties, confId=int(conformer_id))
            if force_field is not None:
                force_field.Minimize(maxIts=500)
                energy = float(force_field.CalcEnergy())
        elif optimize in {"mmff", "uff"} and AllChem.UFFHasAllMoleculeParams(work):
            force_field = AllChem.UFFGetMoleculeForceField(work, confId=int(conformer_id))
            if force_field is not None:
                force_field.Minimize(maxIts=500)
                energy = float(force_field.CalcEnergy())

        candidate = Chem.Mol(work)
        selected = work.GetConformer(int(conformer_id))
        candidate.RemoveAllConformers()
        candidate.AddConformer(Chem.Conformer(selected), assignId=True)
        candidates.append((candidate, energy))
    return sorted(candidates, key=lambda item: float("inf") if item[1] is None else item[1])


def molecule_conformers(request: Any) -> dict[str, Any]:
    molecule = snapshot_to_molecule(request.input)
    params = request.params
    conformers = generate_conformers(
        molecule,
        count=params.count,
        random_seed=params.random_seed,
        optimize=params.optimize,
        prune_rms_threshold=params.prune_rms_threshold,
    )
    candidates: list[dict[str, Any]] = []
    for index, (candidate, energy) in enumerate(conformers):
        snapshot = snapshot_from_molecule(candidate, fmt="sdf")
        candidates.append(
            {
                "candidate_index": index,
                "label": f"Conformer {index + 1}",
                "energy_kcal_mol": energy,
                "snapshot": snapshot.model_dump(mode="json"),
                "summary": derived_summary(snapshot),
                "viewer_structure": viewer_structure(snapshot),
            }
        )
    return {
        "kind": "candidates",
        "candidate_type": "conformer",
        "candidates": candidates,
        "warnings": [],
        "change": {
            "requested_count": int(params.count),
            "candidate_count": len(candidates),
            "optimizer": params.optimize,
        },
    }


__all__ = [
    "generate_conformers",
    "molecule_conformers",
    "molecule_from_viewer",
    "molecule_refresh",
    "rebuild_molecule_from_viewer",
    "snapshot_from_viewer",
]
