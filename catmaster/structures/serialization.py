from __future__ import annotations

import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from monty.json import MontyDecoder, MontyEncoder

from .models import (
    MoleculePayload,
    MoleculeSnapshot,
    PeriodicPayload,
    PeriodicSnapshot,
    SourceVersion,
    StructureSnapshot,
)

MOLECULE_SUFFIXES = {".mol", ".sdf", ".smi", ".smiles"}
PERIODIC_NAMES = {"POSCAR", "CONTCAR"}
PERIODIC_SUFFIXES = {".cif", ".cssr", ".poscar", ".contcar", ".vasp"}
CONSTRAINT_PROPERTY = "selective_dynamics"
_SDF_DATA_HEADER = re.compile(r"^>\s*<([^>]+)>", re.MULTILINE)
_SDF_RECORD_SEPARATOR = re.compile(r"^\$\$\$\$\s*$", re.MULTILINE)


class StructureSerializationError(ValueError):
    pass


class StructureFormatLossError(StructureSerializationError):
    def __init__(self, warnings: list[str]):
        super().__init__("Saving would discard scientific information.")
        self.warnings = warnings


class StructureVersionConflict(StructureSerializationError):
    pass


def source_version(path: Path) -> SourceVersion:
    stat = path.stat()
    return SourceVersion(mtime_ns=int(stat.st_mtime_ns), size=int(stat.st_size))


def _json_safe(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(value, cls=MontyEncoder))


def _plain_viewer_value(value: Any) -> Any:
    if isinstance(value, dict):
        if (
            value.get("@class") == "array"
            and value.get("@module") == "numpy"
            and "data" in value
        ):
            return _plain_viewer_value(value["data"])
        return {
            str(key): _plain_viewer_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_plain_viewer_value(item) for item in value]
    return value


def snapshot_from_structure(
    structure: Any,
    *,
    fmt: str = "",
    path: str = "",
    version: SourceVersion | None = None,
) -> PeriodicSnapshot:
    return PeriodicSnapshot(
        format=fmt,
        path=path,
        source_version=version or SourceVersion(),
        payload=PeriodicPayload(pymatgen=_json_safe(structure.as_dict())),
    )


def snapshot_to_structure(snapshot: PeriodicSnapshot):
    from pymatgen.core import Structure

    if not isinstance(snapshot, PeriodicSnapshot):
        snapshot = PeriodicSnapshot.model_validate(snapshot)
    payload = snapshot.payload.pymatgen
    if not isinstance(payload, dict) or not payload:
        raise StructureSerializationError("Periodic snapshot is missing its pymatgen structure.")
    try:
        decoded = MontyDecoder().process_decoded(payload)
        if isinstance(decoded, Structure):
            return decoded
        return Structure.from_dict(payload)
    except Exception as exc:
        raise StructureSerializationError(f"Periodic snapshot is invalid: {exc}") from exc


def snapshot_from_molecule(
    molecule: Any,
    *,
    fmt: str = "",
    path: str = "",
    version: SourceVersion | None = None,
) -> MoleculeSnapshot:
    from rdkit import Chem

    try:
        property_names = list(
            molecule.GetPropNames(includePrivate=True, includeComputed=False)
        )
        if property_names:
            stream = io.StringIO()
            writer = Chem.SDWriter(stream)
            try:
                writer.write(molecule)
                writer.flush()
                block = stream.getvalue()
            finally:
                writer.close()
        else:
            block = Chem.MolToMolBlock(
                molecule,
                includeStereo=True,
                forceV3000=False,
            )
    except Exception as exc:
        raise StructureSerializationError(f"Molecule could not be serialized: {exc}") from exc
    return MoleculeSnapshot(
        format=fmt,
        path=path,
        source_version=version or SourceVersion(),
        payload=MoleculePayload(molblock=block),
    )


def snapshot_from_atoms(
    atoms: Any,
    *,
    fmt: str = "",
    path: str = "",
    version: SourceVersion | None = None,
) -> StructureSnapshot:
    if any(bool(item) for item in atoms.pbc):
        from pymatgen.io.ase import AseAtomsAdaptor

        return snapshot_from_structure(
            AseAtomsAdaptor.get_structure(atoms),
            fmt=fmt,
            path=path,
            version=version,
        )
    from io import StringIO
    from ase.io import write as ase_write
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    buffer = StringIO()
    ase_write(buffer, atoms, format="xyz")
    molecule = Chem.MolFromXYZBlock(buffer.getvalue())
    if molecule is None:
        raise StructureSerializationError("Could not convert this trajectory frame to a molecule.")
    try:
        rdDetermineBonds.DetermineBonds(molecule, charge=0, allowChargedFragments=True)
    except Exception:
        rdDetermineBonds.DetermineConnectivity(molecule)
    return snapshot_from_molecule(molecule, fmt=fmt, path=path, version=version)


def snapshot_to_molecule(snapshot: MoleculeSnapshot):
    from rdkit import Chem

    if not isinstance(snapshot, MoleculeSnapshot):
        snapshot = MoleculeSnapshot.model_validate(snapshot)
    block = str(snapshot.payload.molblock or "")
    if not block.strip():
        raise StructureSerializationError("Molecule snapshot is missing its MolBlock.")
    if _is_sdf_text(block):
        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(block.encode("utf-8")),
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
        molecule = next((item for item in supplier if item is not None), None)
    else:
        molecule = Chem.MolFromMolBlock(
            block,
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
    if molecule is None:
        raise StructureSerializationError("Molecule MolBlock is invalid.")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def _load_rdkit_molecule(path: Path):
    from rdkit import Chem

    suffix = path.suffix.lower()
    molecule = None
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        molecule = next((item for item in supplier if item is not None), None)
    elif suffix == ".mol":
        molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    elif suffix in {".smi", ".smiles"}:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()
        smiles = first[0].split()[0] if first and first[0].split() else ""
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            molecule = Chem.AddHs(molecule)
    if molecule is None:
        raise StructureSerializationError(f"Could not parse molecule file: {path.name}")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def _read_sdf_text(path: Path) -> str:
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        # CTfiles are historically ASCII-compatible and many legacy data
        # fields use Latin-1.  This fallback is lossless for every input byte.
        return payload.decode("latin-1")


def _is_sdf_text(value: str) -> bool:
    return bool(_SDF_RECORD_SEPARATOR.search(value) or _SDF_DATA_HEADER.search(value))


def _sdf_record_count(value: str) -> int:
    if not value.strip():
        return 0
    separators = len(_SDF_RECORD_SEPARATOR.findall(value))
    if separators:
        tail = _SDF_RECORD_SEPARATOR.split(value)[-1]
        return separators + (1 if tail.strip() else 0)
    return 1


def _sdf_data_property_names(value: str) -> list[str]:
    return sorted({name.strip() for name in _SDF_DATA_HEADER.findall(value) if name.strip()})


def _load_xyz_molecule(path: Path):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    molecule = Chem.MolFromXYZBlock(path.read_text(encoding="utf-8", errors="replace"))
    if molecule is None:
        raise StructureSerializationError(f"Could not parse XYZ molecule: {path.name}")
    try:
        rdDetermineBonds.DetermineBonds(molecule, charge=0, allowChargedFragments=True)
    except Exception:
        try:
            rdDetermineBonds.DetermineConnectivity(molecule)
        except Exception:
            pass
    return molecule


def _load_periodic_structure(path: Path):
    from pymatgen.core import Structure

    try:
        return Structure.from_file(str(path))
    except Exception as primary_exc:
        try:
            from ase.io import read as ase_read
            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = ase_read(str(path), index=0)
            if atoms is None or not any(bool(item) for item in atoms.pbc):
                raise primary_exc
            return AseAtomsAdaptor.get_structure(atoms)
        except Exception as exc:
            raise StructureSerializationError(f"Could not parse periodic structure: {exc}") from exc


def _load_ase_first_frame(path: Path):
    from ase.io import read as ase_read

    try:
        atoms = ase_read(str(path), index=0)
    except Exception as exc:
        raise StructureSerializationError(f"Could not parse structure: {exc}") from exc
    if atoms is None:
        raise StructureSerializationError("Structure file contains no atoms.")
    return atoms


def load_structure_document(path: Path, *, relative_path: str = "") -> tuple[StructureSnapshot, list[str]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise StructureSerializationError("Structure path is not a file.")
    fmt = resolved.name.lower() if resolved.name.upper() in PERIODIC_NAMES else resolved.suffix.lower().lstrip(".")
    version = source_version(resolved)
    display_path = relative_path or resolved.name
    suffix = resolved.suffix.lower()
    warnings: list[str] = []

    if suffix in MOLECULE_SUFFIXES:
        molecule = _load_rdkit_molecule(resolved)
        if suffix == ".sdf":
            sdf_text = _read_sdf_text(resolved)
            record_count = _sdf_record_count(sdf_text)
            if record_count > 1:
                warnings.append(
                    f"{resolved.name} contains {record_count} SDF records. "
                    "Only the first molecule is editable; saving it separately discards the remaining records."
                )
            return MoleculeSnapshot(
                format=fmt,
                path=display_path,
                source_version=version,
                payload=MoleculePayload(molblock=sdf_text),
            ), warnings
        return snapshot_from_molecule(
            molecule,
            fmt=fmt,
            path=display_path,
            version=version,
        ), warnings

    if suffix == ".xyz":
        atoms = _load_ase_first_frame(resolved)
        if any(bool(item) for item in atoms.pbc):
            structure = _load_periodic_structure(resolved)
            return snapshot_from_structure(structure, fmt=fmt, path=display_path, version=version), warnings
        warnings.append(
            "XYZ does not store bond order, stereochemistry, or formal charge; connectivity was inferred for editing."
        )
        return snapshot_from_atoms(
            atoms,
            fmt=fmt,
            path=display_path,
            version=version,
        ), warnings

    if resolved.name.upper() in PERIODIC_NAMES or suffix in PERIODIC_SUFFIXES:
        structure = _load_periodic_structure(resolved)
        return snapshot_from_structure(structure, fmt=fmt, path=display_path, version=version), warnings

    atoms = _load_ase_first_frame(resolved)
    if any(bool(item) for item in atoms.pbc):
        structure = _load_periodic_structure(resolved)
        return snapshot_from_structure(structure, fmt=fmt, path=display_path, version=version), warnings

    from io import StringIO
    from ase.io import write as ase_write

    payload = StringIO()
    ase_write(payload, atoms, format="xyz")
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    molecule = Chem.MolFromXYZBlock(payload.getvalue())
    if molecule is None:
        raise StructureSerializationError("Could not convert the non-periodic structure to a molecule.")
    try:
        rdDetermineBonds.DetermineBonds(molecule, charge=0)
    except Exception:
        rdDetermineBonds.DetermineConnectivity(molecule)
    warnings.append(
        f"{resolved.name} was opened through a coordinate-only compatibility path; review inferred molecular bonds."
    )
    return snapshot_from_molecule(molecule, fmt=fmt, path=display_path, version=version), warnings


def derived_summary(snapshot: StructureSnapshot) -> dict[str, Any]:
    if isinstance(snapshot, PeriodicSnapshot):
        structure = snapshot_to_structure(snapshot)
        summary: dict[str, Any] = {
            "formula": structure.composition.reduced_formula,
            "atom_count": len(structure),
            "pbc": [True, True, True],
            "cell": {
                "matrix": [[float(item) for item in row] for row in structure.lattice.matrix],
                "lengths": [float(item) for item in structure.lattice.abc],
                "angles": [float(item) for item in structure.lattice.angles],
            },
            "has_constraints": CONSTRAINT_PROPERTY in structure.site_properties,
        }
        if len(structure) > 500:
            summary["space_group"] = {"symbol": "Not calculated for large structure", "number": 0}
            summary["symmetry_groups"] = []
            return summary
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            analyzer = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
            summary["space_group"] = {
                "symbol": analyzer.get_space_group_symbol(),
                "number": int(analyzer.get_space_group_number()),
            }
            summary["symmetry_groups"] = [
                [int(index) for index in group]
                for group in analyzer.get_symmetrized_structure().equivalent_indices
            ]
        except Exception:
            summary["space_group"] = {"symbol": "Unresolved", "number": 0}
            summary["symmetry_groups"] = []
        return summary

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecule = snapshot_to_molecule(snapshot)
    return {
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "atom_count": int(molecule.GetNumAtoms()),
        "bond_count": int(molecule.GetNumBonds()),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "pbc": [False, False, False],
    }


def viewer_structure(snapshot: StructureSnapshot) -> dict[str, Any]:
    """Return MatterViz's derived view shape without changing snapshot authority."""
    if isinstance(snapshot, PeriodicSnapshot):
        structure = snapshot_to_structure(snapshot)
        payload = _json_safe(structure.as_dict())
        # MatterViz consumes site properties, but treats arbitrary structure-level
        # calculator metadata as display configuration.  Keep those values in the
        # authoritative snapshot/trajectory metadata and expose a deliberately
        # empty viewer-level property bag.
        payload["properties"] = {}
        for site in payload.get("sites", []):
            if isinstance(site, dict):
                site["properties"] = _plain_viewer_value(site.get("properties") or {})
        return payload

    molecule = snapshot_to_molecule(snapshot)
    conformer = molecule.GetConformer() if molecule.GetNumConformers() else None
    sites: list[dict[str, Any]] = []
    for index, atom in enumerate(molecule.GetAtoms()):
        point = conformer.GetAtomPosition(index) if conformer is not None else None
        xyz = [
            float(point.x) if point is not None else 0.0,
            float(point.y) if point is not None else 0.0,
            float(point.z) if point is not None else 0.0,
        ]
        sites.append(
            {
                "species": [
                    {
                        "element": atom.GetSymbol(),
                        "occu": 1.0,
                        "oxidation_state": float(atom.GetFormalCharge()),
                    }
                ],
                "abc": xyz,
                "xyz": xyz,
                "label": atom.GetSymbol(),
                "properties": {
                    "formal_charge": int(atom.GetFormalCharge()),
                    "isotope": int(atom.GetIsotope()),
                    "_catmaster_mol_atom_index": int(index),
                },
            }
        )
    order_map = {
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "AROMATIC": "aromatic",
    }
    bonds = [
        {
            "site_idx_1": int(bond.GetBeginAtomIdx()),
            "site_idx_2": int(bond.GetEndAtomIdx()),
            "order": order_map.get(str(bond.GetBondType()), float(bond.GetBondTypeAsDouble())),
            "_catmaster_mol_bond_index": int(bond.GetIdx()),
        }
        for bond in molecule.GetBonds()
    ]
    return {
        "sites": sites,
        "charge": int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
        "properties": {"bonds": bonds},
    }


def _has_partial_occupancy(structure: Any) -> bool:
    return any(
        len(site.species) != 1
        or any(abs(float(occupancy) - 1.0) > 1e-12 for occupancy in site.species.values())
        for site in structure
    )


def _has_oxidation_states(structure: Any) -> bool:
    return any(
        getattr(species, "oxi_state", None) is not None
        for site in structure
        for species in site.species
    )


def _ase_array_property_is_lossless(values: Any, atom_count: int) -> bool:
    """Match the primitive rectangular arrays supported by ASE file writers."""
    try:
        import numpy as np

        array = np.asarray(values)
    except Exception:
        return False
    return (
        array.ndim in {1, 2}
        and array.shape[0] == atom_count
        and (
            array.dtype.kind in {"b", "i", "u", "f"}
            or (array.ndim == 1 and array.dtype.kind in {"S", "U"})
        )
    )


def _unpreserved_site_properties(structure: Any, destination: Path) -> list[str]:
    names = set(structure.site_properties)
    if not names:
        return []
    suffix = destination.suffix.lower()
    upper_name = destination.name.upper()
    if upper_name in PERIODIC_NAMES or suffix in {".vasp", ".poscar", ".contcar"}:
        names.discard(CONSTRAINT_PROPERTY)
        return sorted(names)
    if suffix == ".traj":
        # ASE's trajectory writer serializes constraints but drops custom
        # Atoms arrays, even though AseAtomsAdaptor can construct them.
        names.discard(CONSTRAINT_PROPERTY)
        return sorted(names)
    if suffix == ".xyz":
        # This path deliberately writes extended XYZ.  Primitive rectangular
        # arrays and supported ASE constraints survive.
        return sorted(
            name
            for name in names
            if name != CONSTRAINT_PROPERTY
            and not _ase_array_property_is_lossless(
                structure.site_properties[name],
                len(structure),
            )
        )
    # The active CIF reader does not reconstruct arbitrary _atom_site_* fields,
    # and CSSR has no general site-property channel.
    if suffix in {".cif", ".cssr"}:
        return sorted(names)
    return []


def _molecule_data_property_names(molecule: Any, source_text: str) -> list[str]:
    names = set(_sdf_data_property_names(source_text))
    names.update(
        str(name)
        for name in molecule.GetPropNames(
            includePrivate=False,
            includeComputed=False,
        )
    )
    return sorted(names)


def format_loss_warnings(snapshot: StructureSnapshot, destination: Path) -> list[str]:
    suffix = destination.suffix.lower()
    warnings: list[str] = []
    if isinstance(snapshot, PeriodicSnapshot):
        structure = snapshot_to_structure(snapshot)
        has_constraints = CONSTRAINT_PROPERTY in structure.site_properties
        if has_constraints and destination.name.upper() not in PERIODIC_NAMES and suffix not in {
            ".vasp",
            ".poscar",
            ".contcar",
            ".traj",
            ".xyz",
        }:
            warnings.append(
                f"{destination.name} cannot preserve selective-dynamics constraints. "
                "Save as POSCAR/CONTCAR/.vasp, extended XYZ, or .traj."
            )
        unsupported_properties = _unpreserved_site_properties(structure, destination)
        # Selective dynamics already has a specific, actionable warning above.
        if has_constraints and suffix not in {".traj", ".xyz"}:
            unsupported_properties = [
                name for name in unsupported_properties if name != CONSTRAINT_PROPERTY
            ]
        if unsupported_properties:
            warnings.append(
                f"{destination.name} cannot preserve site properties: "
                f"{', '.join(unsupported_properties)}."
            )
        limited_site_format = (
            destination.name.upper() in PERIODIC_NAMES
            or suffix in {".vasp", ".poscar", ".contcar", ".cssr", ".xyz", ".traj"}
        )
        if _has_partial_occupancy(structure) and limited_site_format:
            warnings.append(
                f"{destination.name} cannot represent partial occupancies or disordered sites; "
                "accepted lossy save uses the highest-occupancy species at each site. Use CIF to preserve occupancies."
            )
        oxidation_limited_format = (
            destination.name.upper() in PERIODIC_NAMES
            or suffix in {".vasp", ".poscar", ".contcar", ".cssr", ".traj"}
        )
        if _has_oxidation_states(structure) and oxidation_limited_format:
            warnings.append(
                f"{destination.name} cannot reliably preserve oxidation states. Use CIF or extended XYZ."
            )
        if suffix in MOLECULE_SUFFIXES:
            warnings.append("A periodic crystal cannot be saved as a molecule connection-table format.")
    else:
        molecule = snapshot_to_molecule(snapshot)
        source_text = str(snapshot.payload.molblock or "")
        record_count = _sdf_record_count(source_text) if _is_sdf_text(source_text) else 1
        if record_count > 1:
            warnings.append(
                f"The source contains {record_count} SDF records, but the workbench edits and saves "
                "only the first molecule; the remaining records will be discarded."
            )
        data_properties = _molecule_data_property_names(molecule, source_text)
        if data_properties and suffix != ".sdf":
            warnings.append(
                f"{destination.name} cannot preserve SDF data properties: "
                f"{', '.join(data_properties)}. Save as SDF to retain them."
            )
        if suffix in {".smi", ".smiles"} and molecule.GetNumConformers():
            warnings.append("SMILES preserves molecular topology but cannot preserve the current 3D coordinates.")
        if suffix == ".xyz":
            if molecule.GetNumBonds():
                warnings.append("XYZ cannot preserve molecular bonds, aromaticity, or bond order.")
            if any(atom.GetFormalCharge() for atom in molecule.GetAtoms()):
                warnings.append("XYZ cannot preserve formal charge.")
            if Chem_has_stereo(molecule):
                warnings.append("XYZ cannot preserve molecular stereochemistry.")
        if destination.name.upper() in PERIODIC_NAMES or suffix in PERIODIC_SUFFIXES:
            warnings.append(
                "A molecule saved as a periodic structure loses its connection-table authority; save as SDF or MOL instead."
            )
    return warnings


def Chem_has_stereo(molecule: Any) -> bool:
    from rdkit import Chem

    return bool(Chem.FindMolChiralCenters(molecule, includeUnassigned=True)) or any(
        int(bond.GetStereo()) != int(Chem.rdchem.BondStereo.STEREONONE)
        for bond in molecule.GetBonds()
    )


def _ordered_approximation(structure: Any):
    """Return the explicit lossy representation approved for ordered-only formats."""
    if structure.is_ordered:
        return structure
    from pymatgen.core import Structure

    species = [
        max(
            site.species.items(),
            key=lambda item: (float(item[1]), str(item[0])),
        )[0]
        for site in structure
    ]
    return Structure(
        structure.lattice,
        species,
        structure.frac_coords,
        coords_are_cartesian=False,
        site_properties={
            name: list(values)
            for name, values in structure.site_properties.items()
        },
        labels=[site.label for site in structure],
        properties=dict(structure.properties),
    )


def _extended_xyz_approximation(structure: Any):
    prepared = _ordered_approximation(structure).copy()
    for name in list(prepared.site_properties):
        if name == CONSTRAINT_PROPERTY:
            continue
        if not _ase_array_property_is_lossless(
            prepared.site_properties[name],
            len(prepared),
        ):
            prepared.remove_site_property(name)
    return prepared


def _write_periodic(
    snapshot: PeriodicSnapshot,
    output: Path,
    *,
    format_path: Path,
    cif_symprec: float,
    angle_tolerance: float,
) -> None:
    structure = snapshot_to_structure(snapshot)
    suffix = format_path.suffix.lower()
    upper_name = format_path.name.upper()
    if upper_name in PERIODIC_NAMES or suffix in {".vasp", ".poscar", ".contcar"}:
        from pymatgen.io.vasp import Poscar

        Poscar(_ordered_approximation(structure)).write_file(output)
        return
    if suffix == ".cif":
        from pymatgen.io.cif import CifWriter

        CifWriter(
            structure,
            symprec=float(cif_symprec),
            angle_tolerance=float(angle_tolerance),
            refine_struct=False,
            write_site_properties=True,
        ).write_file(output)
        return
    if suffix == ".cssr":
        _ordered_approximation(structure).to(filename=str(output), fmt="cssr")
        return
    if suffix == ".xyz":
        from ase.io import write as ase_write
        from pymatgen.io.ase import AseAtomsAdaptor

        ase_write(
            str(output),
            AseAtomsAdaptor.get_atoms(_extended_xyz_approximation(structure)),
            format="extxyz",
        )
        return
    if suffix == ".traj":
        from ase.io import write as ase_write
        from pymatgen.io.ase import AseAtomsAdaptor

        ase_write(
            str(output),
            AseAtomsAdaptor.get_atoms(_ordered_approximation(structure)),
            format="traj",
        )
        return
    raise StructureSerializationError(
        "Unsupported periodic save format. Use POSCAR, CONTCAR, .vasp, .cif, .cssr, .xyz, or .traj."
    )


def _write_molecule(snapshot: MoleculeSnapshot, output: Path, *, format_path: Path) -> None:
    from rdkit import Chem

    molecule = snapshot_to_molecule(snapshot)
    suffix = format_path.suffix.lower()
    if suffix == ".sdf":
        writer = Chem.SDWriter(str(output))
        try:
            writer.write(molecule)
        finally:
            writer.close()
        return
    if suffix == ".mol":
        Chem.MolToMolFile(molecule, str(output), includeStereo=True)
        return
    if suffix in {".smi", ".smiles"}:
        output.write_text(
            Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=False) + "\n",
            encoding="utf-8",
        )
        return
    if suffix == ".xyz":
        if molecule.GetNumConformers() == 0:
            raise StructureSerializationError("Molecule has no 3D conformer to save as XYZ.")
        output.write_text(Chem.MolToXYZBlock(molecule), encoding="utf-8")
        return
    raise StructureSerializationError("Unsupported molecule save format. Use .sdf, .mol, .smi, .smiles, or .xyz.")


def save_structure_document(
    snapshot: StructureSnapshot,
    destination: Path,
    *,
    overwrite: bool,
    expected_version: SourceVersion,
    accept_format_loss: bool,
    cif_symprec: float = 0.01,
    cif_angle_tolerance: float = 5.0,
) -> tuple[SourceVersion, list[str]]:
    destination = destination.expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise StructureVersionConflict(
                f"{destination.name} already exists. Choose a new filename or explicitly confirm overwrite."
            )
        actual = source_version(destination)
        if (
            expected_version.mtime_ns <= 0
            or expected_version.size < 0
            or actual.mtime_ns != expected_version.mtime_ns
            or actual.size != expected_version.size
        ):
            raise StructureVersionConflict(
                f"{destination.name} changed on disk after it was opened. Reopen it or save under a new name."
            )

    warnings = format_loss_warnings(snapshot, destination)
    if warnings and not accept_format_loss:
        raise StructureFormatLossError(warnings)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if isinstance(snapshot, PeriodicSnapshot):
            _write_periodic(
                snapshot,
                temporary,
                format_path=destination,
                cif_symprec=cif_symprec,
                angle_tolerance=cif_angle_tolerance,
            )
        else:
            _write_molecule(snapshot, temporary, format_path=destination)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return source_version(destination), warnings
