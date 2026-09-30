from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceVersion(StrictModel):
    mtime_ns: int = Field(0, ge=0)
    size: int = Field(0, ge=0)


class PeriodicPayload(StrictModel):
    pymatgen: dict = Field(default_factory=dict)


class MoleculePayload(StrictModel):
    molblock: str = ""


class PeriodicSnapshot(StrictModel):
    mode: Literal["periodic"] = "periodic"
    format: str = ""
    path: str = ""
    source_version: SourceVersion = Field(default_factory=SourceVersion)
    payload: PeriodicPayload


class MoleculeSnapshot(StrictModel):
    mode: Literal["molecule"] = "molecule"
    format: str = ""
    path: str = ""
    source_version: SourceVersion = Field(default_factory=SourceVersion)
    payload: MoleculePayload


StructureSnapshot: TypeAlias = Annotated[
    PeriodicSnapshot | MoleculeSnapshot,
    Field(discriminator="mode"),
]
STRUCTURE_SNAPSHOT_ADAPTER = TypeAdapter(StructureSnapshot)


class StructureOpenRequest(StrictModel):
    workspace: str = Field(min_length=1, max_length=240)
    path: str = Field(min_length=1, max_length=2048)


class Matrix3x3Params(StrictModel):
    matrix: list[list[float]]

    @field_validator("matrix")
    @classmethod
    def _matrix_is_3x3(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("matrix must contain exactly three rows of three values")
        return [[float(component) for component in row] for row in value]


class SupercellParams(StrictModel):
    matrix: list[list[int]]

    @field_validator("matrix")
    @classmethod
    def _matrix_is_nonsingular_3x3(cls, value: list[list[int]]) -> list[list[int]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("matrix must contain exactly three rows of three integers")
        matrix = [[int(component) for component in row] for row in value]
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        if determinant == 0:
            raise ValueError("supercell matrix must be nonsingular")
        return matrix


class SetCellParams(Matrix3x3Params):
    keep: Literal["fractional", "cartesian"] = "fractional"


class SymmetryParams(StrictModel):
    symprec: float = Field(0.01, gt=0, le=1)
    angle_tolerance: float = Field(5.0, gt=0, le=45)


class SlabCandidatesParams(StrictModel):
    miller_index: list[int] = Field(default_factory=lambda: [1, 1, 1])
    min_slab_size: float = Field(10.0, gt=0)
    min_vacuum_size: float = Field(15.0, ge=0)
    center_slab: bool = True
    symmetrize: bool = False
    orthogonal: bool = False
    lll_reduce: bool = False
    surface_supercell: list[list[int]] = Field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    @field_validator("miller_index")
    @classmethod
    def _miller_is_valid(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("miller_index must contain three integers")
        normalized = [int(item) for item in value]
        if normalized == [0, 0, 0]:
            raise ValueError("miller_index cannot be [0, 0, 0]")
        return normalized

    @field_validator("surface_supercell")
    @classmethod
    def _surface_supercell_is_valid(cls, value: list[list[int]]) -> list[list[int]]:
        return SupercellParams(matrix=value).matrix


class DefectCandidatesParams(StrictModel):
    kind: Literal["vacancy", "substitution", "interstitial"]
    new_species: str = ""
    site_index: int = Field(-1, ge=-1)
    coordinates: list[float] = Field(default_factory=list)
    coordinate_type: Literal["fractional", "cartesian"] = "fractional"
    symprec: float = Field(0.01, gt=0, le=1)
    angle_tolerance: float = Field(5.0, gt=0, le=45)

    @model_validator(mode="after")
    def _validate_kind_inputs(self) -> "DefectCandidatesParams":
        if self.kind == "substitution" and not self.new_species.strip():
            raise ValueError("new_species is required for substitution candidates")
        if self.kind == "interstitial":
            if not self.new_species.strip():
                raise ValueError("new_species is required for an interstitial")
            if len(self.coordinates) != 3:
                raise ValueError("coordinates must contain three values for an interstitial")
        return self


class AdsorptionCandidatesParams(StrictModel):
    adsorbate_molblock: str = Field(min_length=1)
    distance: float = Field(2.0, gt=0)
    site_kinds: list[Literal["ontop", "bridge", "hollow"]] = Field(
        default_factory=lambda: ["ontop", "bridge", "hollow"]
    )
    reorient: bool = False
    orientation_euler_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @field_validator("orientation_euler_deg")
    @classmethod
    def _orientation_is_xyz(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("orientation_euler_deg must contain x, y, and z rotation angles")
        return [float(item) for item in value]


class MoleculeConformersParams(StrictModel):
    count: int = Field(10, ge=1, le=100)
    random_seed: int = 42
    optimize: Literal["mmff", "uff", "none"] = "mmff"
    prune_rms_threshold: float = Field(0.35, ge=0)


class MoleculeRefreshParams(StrictModel):
    pass


class MoleculeFromViewerParams(StrictModel):
    viewer_structure: dict = Field(default_factory=dict)


class MakeSupercellRequest(StrictModel):
    operation: Literal["make_supercell"]
    input: PeriodicSnapshot
    params: SupercellParams


class SetCellRequest(StrictModel):
    operation: Literal["set_cell"]
    input: PeriodicSnapshot
    params: SetCellParams


class PrimitiveRequest(StrictModel):
    operation: Literal["primitive"]
    input: PeriodicSnapshot
    params: SymmetryParams = Field(default_factory=SymmetryParams)


class ConventionalRequest(StrictModel):
    operation: Literal["conventional"]
    input: PeriodicSnapshot
    params: SymmetryParams = Field(default_factory=SymmetryParams)


class StandardizeRequest(StrictModel):
    operation: Literal["standardize"]
    input: PeriodicSnapshot
    params: SymmetryParams = Field(default_factory=SymmetryParams)


class SymmetrizeRequest(StrictModel):
    operation: Literal["symmetrize"]
    input: PeriodicSnapshot
    params: SymmetryParams = Field(default_factory=SymmetryParams)


class SlabCandidatesRequest(StrictModel):
    operation: Literal["slab_candidates"]
    input: PeriodicSnapshot
    params: SlabCandidatesParams


class DefectCandidatesRequest(StrictModel):
    operation: Literal["defect_candidates"]
    input: PeriodicSnapshot
    params: DefectCandidatesParams


class AdsorptionCandidatesRequest(StrictModel):
    operation: Literal["adsorption_candidates"]
    input: PeriodicSnapshot
    params: AdsorptionCandidatesParams


class MoleculeConformersRequest(StrictModel):
    operation: Literal["molecule_conformers"]
    input: MoleculeSnapshot
    params: MoleculeConformersParams = Field(default_factory=MoleculeConformersParams)


class MoleculeRefreshRequest(StrictModel):
    operation: Literal["molecule_refresh"]
    input: MoleculeSnapshot
    params: MoleculeRefreshParams = Field(default_factory=MoleculeRefreshParams)


class MoleculeFromViewerRequest(StrictModel):
    operation: Literal["molecule_from_viewer"]
    input: MoleculeSnapshot
    params: MoleculeFromViewerParams


TransformRequest: TypeAlias = Annotated[
    MakeSupercellRequest
    | SetCellRequest
    | PrimitiveRequest
    | ConventionalRequest
    | StandardizeRequest
    | SymmetrizeRequest
    | SlabCandidatesRequest
    | DefectCandidatesRequest
    | AdsorptionCandidatesRequest
    | MoleculeConformersRequest
    | MoleculeRefreshRequest
    | MoleculeFromViewerRequest,
    Field(discriminator="operation"),
]
TRANSFORM_REQUEST_ADAPTER = TypeAdapter(TransformRequest)


class SaveStructureRequest(StrictModel):
    workspace: str = Field(min_length=1, max_length=240)
    destination_path: str = Field(min_length=1, max_length=2048)
    snapshot: StructureSnapshot
    viewer_structure: dict = Field(default_factory=dict)
    overwrite: bool = False
    expected_source_version: SourceVersion = Field(default_factory=SourceVersion)
    accept_format_loss: bool = False
    cif_symprec: float = Field(0.01, gt=0, le=1)
    cif_angle_tolerance: float = Field(5.0, gt=0, le=45)
