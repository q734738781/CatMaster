#!/usr/bin/env python3
"""Create the deterministic, authenticated-workspace fixture used by WebUI QA.

The caller provisions the user account through the real WebUI first, then
passes that user's workspace path here. The fixture is local-only, idempotent,
and never touches paths outside that exact workspace.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from catmaster.storage import connect_workspace_db
from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    HypothesisCreateRequest,
    ResultCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.runtime.self_evolution import (
    LearningCandidate,
    Observation,
    SelfEvolutionStore,
)
from catmaster.runtime.self_evolution.storage import hash_tree
from catmaster.tools.base import ensure_project_space_layout
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_models import (
    ArtifactPart,
    MessagePart,
    ThreadMessage,
    ThreadStatus,
    ToolCallPart,
)
from catmaster.webui.thread_store import ThreadStore


THREAD_ID = "thread_release_browser_fixture"
LONG_FILENAME = (
    "surface-screening-results-with-a-deliberately-long-name-for-overflow-and-"
    "accessible-label-verification.csv"
)
LONG_TOKEN = "CANDIDATE_" + ("FeCoNiMo" * 30)
LONG_URL = (
    "https://example.org/research/catalyst-screening/"
    + "retained-candidate-evidence/" * 18
    + "?view=full-record"
)
PRIMARY_GRAPH_ID = "graph_browser_fixture"
SECONDARY_GRAPH_ID = "graph_browser_fixture_secondary"
LARGE_GRAPH_ID = "graph_browser_fixture_large"
SELF_EVOLUTION_OBSERVATION_IDS = (
    "obs_browser_bounded_integrity",
    "obs_browser_overgeneral_integrity",
    "obs_browser_stale_integrity",
    "obs_browser_conflicting_integrity",
)
SELF_EVOLUTION_CANDIDATE_IDS = (
    "sec_browser_bounded",
    "sec_browser_overgeneral",
    "sec_browser_stale",
    "sec_browser_conflict",
)


def _write_files(workspace: Path) -> None:
    files = workspace / "files"
    files.mkdir(parents=True, exist_ok=True)
    (files / "research-summary.md").write_text(
        "# Catalyst screening summary\n\n"
        "The current comparison retains three candidates for follow-up. "
        "Open the source table for exact energies and uncertainty notes.\n",
        encoding="utf-8",
    )
    csv_rows = ["candidate,energy_ev,status"]
    for index in range(250):
        status = ("retained", "recheck", "excluded")[index % 3]
        csv_rows.append(f"candidate-{index + 1:03d},{-1.5 + index * 0.004:.3f},{status}")
    csv_text = "\n".join(csv_rows) + "\n"
    (files / "candidate-energies.csv").write_text(csv_text, encoding="utf-8")
    (files / LONG_FILENAME).write_text(csv_text, encoding="utf-8")
    (files / "experiment-plan.json").write_text(
        json.dumps(
            {
                "question": "Which retained surface binds CO most selectively?",
                "experiments": [
                    "relax clean surfaces",
                    "sample adsorption sites",
                    "compare corrected adsorption energies",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (files / "Fe.cif").write_text(
        """data_Fe
_symmetry_space_group_name_H-M   'I m -3 m'
_cell_length_a   2.8665
_cell_length_b   2.8665
_cell_length_c   2.8665
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0 0 0
Fe2 Fe 0.5 0.5 0.5
""",
        encoding="utf-8",
    )
    # Structure-workbench fixtures exercise scientific round-trips, lazy
    # trajectories, molecule semantics, and the volume worker in the same
    # authenticated workspace used by browser QA.
    from ase import Atoms
    from ase.io import write as ase_write
    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    from pymatgen.io.vasp.outputs import Chgcar
    from pymatgen.io.vasp import Poscar
    from rdkit import Chem
    from rdkit.Chem import AllChem

    constrained = Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 83, 72),
        ["Si", "O"],
        [[0.0, 0.0, 0.0], [0.23, 0.37, 0.41]],
        site_properties={
            "selective_dynamics": [[False, True, False], [True, True, True]],
        },
    )
    Poscar(constrained).write_file(files / "selective-dynamics.vasp")
    CifWriter(constrained, symprec=None).write_file(files / "triclinic.cif")
    slab = Structure(
        Lattice.from_parameters(3.61, 3.61, 24.0, 90, 90, 120),
        ["Cu", "Cu", "Cu", "Cu"],
        [[0, 0, 0.44], [0.5, 0.5, 0.44], [0, 0.5, 0.52], [0.5, 0, 0.52]],
    )
    Poscar(slab).write_file(files / "slab-large-vacuum.vasp")
    ase_write(
        files / "water.xyz",
        Atoms("OH2", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]),
        format="xyz",
    )
    (files / "partial-occupancy.cif").write_text(
        """data_partial_occupancy
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5
_cell_length_b   5
_cell_length_c   5
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0 0 0 0.5
K1  K  0 0 0 0.5
Cl1 Cl 0.5 0.5 0.5 1.0
""",
        encoding="utf-8",
    )

    molecule = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)c1ccccc1[NH3+]"))
    embed = AllChem.ETKDGv3()
    embed.randomSeed = 7
    if AllChem.EmbedMolecule(molecule, embed) != 0:
        raise RuntimeError("Could not create the molecule browser fixture.")
    writer = Chem.SDWriter(str(files / "charged-chiral-aromatic.sdf"))
    writer.write(molecule)
    writer.close()

    trajectory = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.72 + frame_index / 10_000]],
            cell=[10, 10, 10],
            pbc=True,
            info={"energy": -1.0 + frame_index / 1_000},
        )
        for frame_index in range(301)
    ]
    ase_write(files / "trajectory-301.traj", trajectory, format="traj")
    ase_write(files / "trajectory-301.extxyz", trajectory, format="extxyz")
    ase_write(files / "XDATCAR", trajectory, format="vasp-xdatcar")
    for atom_count in (1_000, 10_000, 50_000):
        side = int(np.ceil(atom_count ** (1 / 3)))
        indices = np.arange(atom_count, dtype=float)
        positions = np.column_stack(
            (
                indices % side,
                np.floor(indices / side) % side,
                np.floor(indices / (side * side)),
            )
        ) * 1.8
        ase_write(
            files / f"atoms-{atom_count}.extxyz",
            Atoms(
                numbers=np.full(atom_count, 14, dtype=int),
                positions=positions,
                cell=[side * 1.8, side * 1.8, side * 1.8],
                pbc=True,
            ),
            format="extxyz",
        )

    cube_atoms = Atoms("He", positions=[[2, 2, 2]], cell=[4, 4, 4], pbc=True)
    grid_axis = np.linspace(-1, 1, 12)
    gx, gy, gz = np.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij")
    signed_grid = np.exp(-4 * (gx**2 + gy**2 + gz**2)) - 0.18
    ase_write(files / "signed-density.cube", cube_atoms, data=signed_grid, format="cube")
    vasp_field = Chgcar(
        Structure(Lattice.cubic(4), ["He"], [[0.5, 0.5, 0.5]]),
        {"total": signed_grid},
    )
    for field_name in ("CHGCAR", "LOCPOT", "ELFCAR"):
        vasp_field.write_file(files / field_name)

    xsf_values = " ".join(f"{value:.6f}" for value in signed_grid.ravel(order="F"))
    (files / "signed-density.xsf").write_text(
        "CRYSTAL\n"
        "PRIMVEC\n4 0 0\n0 4 0\n0 0 4\n"
        "PRIMCOORD\n1 1\n2 2 2 2\n"
        "BEGIN_BLOCK_DATAGRID_3D\nfield\n"
        "BEGIN_DATAGRID_3D_signed\n"
        "12 12 12\n0 0 0\n4 0 0\n0 4 0\n0 0 4\n"
        f"{xsf_values}\n"
        "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n",
        encoding="utf-8",
    )
    (files / "broken.cif").write_text(
        "data_broken\n_cell_length_a this-is-not-a-number\n",
        encoding="utf-8",
    )
    # A small valid single-page PDF is sufficient for the preview/download
    # contract; the fixture does not need a PDF authoring dependency.
    (files / "source-note.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    internal = workspace / "metadata" / "browser-fixture-private.json"
    internal.parent.mkdir(parents=True, exist_ok=True)
    internal.write_text('{"provider_token":"must-never-appear"}\n', encoding="utf-8")


def _assistant_parts(
    pair_index: int,
    *,
    workspace: Path,
    artifact_id: str,
) -> tuple[list[MessagePart], str, dict[str, object]]:
    base_text = (
        f"Cycle {pair_index + 1}: the calculation record is complete. "
        "The evidence table remains available in Files, and no conclusion is "
        "silently inferred beyond the displayed values. The geometry, convergence "
        "criteria, and uncertainty note were checked independently before this "
        "status was recorded.\n\n"
        "The retained observation is deliberately separated from interpretation: "
        "the displayed energy is evidence, while the proposed mechanism remains a "
        "testable hypothesis. Open the linked artifact to inspect the source rows "
        "used for the comparison."
    )
    status = "completed"
    meta: dict[str, object] = {}

    if pair_index == 25:
        return [
            MessagePart(
                id="part_fixture_reasoning",
                type="reasoning",
                status="completed",
                text="Comparing convergence, geometry changes, and the retained scientific constraints.",
            ),
            MessagePart(
                id="part_fixture_long_text",
                type="text",
                status="completed",
                text=(base_text + "\n\n") * 260,
            ),
        ], status, meta
    if pair_index == 26:
        return [
            MessagePart(
                id="part_fixture_markdown_stress",
                type="text",
                status="completed",
                text=(
                    f"{base_text}\n\n"
                    "| Candidate | Energy / eV | Decision |\n"
                    "| --- | ---: | --- |\n"
                    "| A | -1.24 | retain |\n"
                    "| B | -1.08 | recheck |\n\n"
                    "```python\n"
                    "retained = [row for row in candidates if row.energy_ev < -1.0]\n"
                    "```\n\n"
                    f"Complete record: {LONG_URL}\n\n"
                    f"Stable identifier used for overflow QA: {LONG_TOKEN}"
                ),
            )
        ], status, meta
    if pair_index == 27:
        return [
            ToolCallPart(
                id="part_fixture_tool",
                tool_call_id="call_fixture_tool",
                tool="remote_submission",
                status="completed",
                input={
                    "task": "Relax retained catalyst candidate",
                    "engine_path": str(workspace / "metadata" / "runs" / "private"),
                    "api_key": "must-never-appear",
                },
                output={
                    "status": "completed",
                    "summary": "Geometry relaxation finished.",
                    "provider_payload": {"raw": "must-never-appear"},
                },
                meta={"agent_name": "materials_worker"},
            )
        ], status, meta
    if pair_index == 28:
        return [
            MessagePart(
                id="part_fixture_receipt",
                type="receipt",
                status="completed",
                meta={
                    "task_name": "Surface relaxation",
                    "status": "completed",
                    "machine": "cpu_server_2",
                    "elapsed_seconds": 184,
                    "run_dir": str(workspace / "metadata" / "runs" / "private"),
                },
            )
        ], status, meta
    if pair_index == 29:
        return [
            ArtifactPart(
                id="part_fixture_artifact",
                artifact_id=artifact_id,
                renderer="markdown",
                title="Catalyst screening summary",
                summary="A concise comparison with a recoverable source file.",
                path="files/research-summary.md",
                status="completed",
            )
        ], status, meta
    if pair_index == 30:
        return [
            MessagePart(
                id="part_fixture_interrupt",
                type="interrupt",
                status="pending",
                text="Review the bounded calculation before it is submitted.",
                meta={
                    "title": "Review the proposed relaxation",
                    "payload": {
                        "interrupts": [
                            {
                                "value": {
                                    "action_requests": [
                                        {
                                            "name": "submit relaxation",
                                            "args": {
                                                "candidate": "A",
                                                "force_threshold": 0.05,
                                            },
                                        }
                                    ],
                                    "review_configs": [
                                        {
                                            "action_name": "submit relaxation",
                                            "allowed_decisions": [
                                                "approve",
                                                "edit",
                                                "reject",
                                            ],
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                },
            )
        ], "interrupted", meta
    if pair_index == 31:
        meta = {
            "error": (
                "The calculation stopped before producing a result at "
                f"{workspace / 'metadata' / 'runs' / 'private'}."
            ),
            "error_code": "REMOTE_EXIT",
            "retry_safe": True,
        }
        return [], "failed", meta
    if pair_index == 32:
        return [
            MessagePart(
                id="part_fixture_unknown",
                type="provider-future-payload",
                status="completed",
                text="LLM_RAW_RESPONSE",
                meta={
                    "engine_path": str(workspace / "metadata" / "runs" / "private"),
                    "provider_token": "must-never-appear",
                },
            )
        ], status, meta
    if pair_index == 33:
        return [
            MessagePart(
                id="part_fixture_trace",
                type="trace",
                status="completed",
                text=(
                    "RUN_STATE_CHANGE: internal step completed in "
                    f"{workspace / 'metadata' / 'runs' / 'private'}"
                ),
            )
        ], status, meta
    if pair_index == 34:
        return [], "failed", {
            "error": {
                "status_code": 422,
                "detail": [
                    {
                        "loc": ["body", "force_threshold"],
                        "msg": "Input should be greater than zero",
                        "type": "greater_than",
                    },
                    {
                        "loc": ["body", "structure"],
                        "msg": "Field required",
                        "type": "missing",
                    },
                ],
            },
            "error_code": "REQUEST_VALIDATION",
            "retry_safe": False,
        }
    if pair_index == 35:
        return [], "failed", {
            "error": (
                "<!doctype html><html><head><title>502 Bad Gateway</title></head>"
                "<body><h1>Bad Gateway</h1><p>cloud proxy request id: private</p>"
                "</body></html>"
            ),
            "error_code": "HTTP_502",
            "retry_safe": True,
        }
    return [
        MessagePart(
            id=f"part_fixture_answer_{pair_index:02d}",
            type="text",
            status="completed",
            text=base_text,
        )
    ], status, meta


def _write_research_graph(workspace: Path, thread_id: str) -> None:
    service = ResearchGraphService(
        workspace=workspace,
        workspace_id=workspace.name,
    )
    with connect_workspace_db(workspace) as connection:
        connection.execute(
            "DELETE FROM ui_events WHERE graph_id IN (?, ?, ?)",
            (PRIMARY_GRAPH_ID, SECONDARY_GRAPH_ID, LARGE_GRAPH_ID),
        )
        connection.execute(
            "DELETE FROM research_graphs WHERE graph_id IN (?, ?, ?)",
            (PRIMARY_GRAPH_ID, SECONDARY_GRAPH_ID, LARGE_GRAPH_ID),
        )
    service.store.create_graph(
        graph_id=PRIMARY_GRAPH_ID,
        title="Catalyst selectivity mechanism",
        question="Which surface mechanism controls the observed CO selectivity?",
        orchestration_mode="manual",
        initial_hypotheses=[
            {
                "title": "Site ensemble mechanism",
                "claim": "A compact metal ensemble controls the observed CO selectivity.",
                "rationale": "The retained structures expose different contiguous metal ensembles.",
                "predictions": [
                    "Selectivity follows ensemble size under matched coverage.",
                    "Isolated sites do not reproduce the retained trend.",
                ],
                "importance": "high",
            }
        ],
    )
    view = service.presentation(PRIMARY_GRAPH_ID)
    hypotheses = [
        node for node in view["nodes"] if node["kind"] == "hypothesis"
    ]
    experiment = service.add_experiment(
        PRIMARY_GRAPH_ID,
        ExperimentCreateRequest(
            expected_revision=view["graph"]["revision"],
            title="Matched adsorption comparison",
            objective="Compare CO adsorption on matched ensemble and support environments.",
            plan_summary=(
                "Relax the retained structures with identical settings, sample "
                "the same coverage, and compare corrected adsorption energies."
            ),
            decision_rule=(
                "An ensemble-size trend at matched support coordination supports "
                "the site-ensemble hypothesis; failure to preserve the ordering "
                "opposes it."
            ),
            execution_lane="experiment",
            expected_value="high",
            estimated_compute_cost="medium",
            state="ready",
            tests_hypothesis_ids=[node["node_id"] for node in hypotheses],
        ),
    )
    first_result = service.record_result(
        PRIMARY_GRAPH_ID,
        ResultCreateRequest(
            expected_revision=experiment["graph"]["revision"],
            title="Matched comparison result",
            summary=(
                "The matched comparison follows ensemble size, while the "
                "charge-transfer descriptor does not preserve the ordering."
            ),
            experiment_node_id=experiment["node"]["node_id"],
            judgments=[
                {
                    "hypothesis_node_id": hypotheses[0]["node_id"],
                    "relation": "supports",
                }
            ],
            refs=[
                {
                    "ref_kind": "note",
                    "ref_id": "files/research-summary.md",
                }
            ],
        ),
    )
    next_hypothesis = service.add_hypothesis(
        PRIMARY_GRAPH_ID,
        HypothesisCreateRequest(
            expected_revision=first_result["graph"]["revision"],
            title="Coverage-dependent ensemble mechanism",
            claim="The ensemble effect changes when CO coverage reorganizes the active site.",
            rationale="The first result was obtained at only one matched coverage.",
            predictions=[
                "The ensemble ordering weakens or reverses across a coverage series."
            ],
            importance="medium",
            suggested_by_result_ids=[first_result["node"]["node_id"]],
        ),
    )
    service.add_experiment(
        PRIMARY_GRAPH_ID,
        ExperimentCreateRequest(
            expected_revision=next_hypothesis["graph"]["revision"],
            title="Coverage-series follow-up",
            objective="Test whether coverage changes the ensemble ordering.",
            plan_summary="Repeat the matched comparison at three explicit CO coverages.",
            decision_rule=(
                "A reproducible coverage-dependent ordering supports the new "
                "hypothesis; an unchanged ordering opposes it."
            ),
            execution_lane="experiment",
            expected_value="high",
            estimated_compute_cost="high",
            state="draft",
            tests_hypothesis_ids=[next_hypothesis["node"]["node_id"]],
            depends_on_experiment_ids=[experiment["node"]["node_id"]],
        ),
    )
    service.store.create_graph(
        graph_id=SECONDARY_GRAPH_ID,
        title="Twenty-five-node rendering acceptance",
        question="Can the Research Graph remain usable at the documented 25-node density?",
        orchestration_mode="manual",
        initial_hypotheses=[
            {
                "title": f"Historical correction branch {index + 1:02d}",
                "claim": (
                    f"Correction branch {index + 1:02d} preserves the observed "
                    "ordering under its stated boundary."
                ),
                "rationale": "This is a deterministic medium-density browser fixture.",
                "predictions": [
                    f"Matched comparison {index + 1:02d} preserves the ordering."
                ],
            }
            for index in range(10)
        ],
    )
    medium = service.presentation(SECONDARY_GRAPH_ID)
    medium_hypotheses = [
        node for node in medium["nodes"] if node["kind"] == "hypothesis"
    ]
    revision = medium["graph"]["revision"]
    for index in range(5):
        medium_experiment = service.add_experiment(
            SECONDARY_GRAPH_ID,
            ExperimentCreateRequest(
                expected_revision=revision,
                title=f"Historical comparison {index + 1:02d}",
                objective=f"Test correction branch {index + 1:02d}.",
                plan_summary="Run one matched deterministic comparison.",
                decision_rule="The predefined ordering either persists or does not.",
                execution_lane="experiment",
                state="ready",
                tests_hypothesis_ids=[medium_hypotheses[index]["node_id"]],
            ),
        )
        medium_result = service.record_result(
            SECONDARY_GRAPH_ID,
            ResultCreateRequest(
                expected_revision=medium_experiment["graph"]["revision"],
                title=f"Historical result {index + 1:02d}",
                summary="The matched correction comparison was recorded.",
                experiment_node_id=medium_experiment["node"]["node_id"],
                judgments=[
                    {
                        "hypothesis_node_id": medium_hypotheses[index]["node_id"],
                        "relation": "supports" if index % 2 == 0 else "inconclusive",
                    }
                ],
            ),
        )
        revision = medium_result["graph"]["revision"]
    for index in range(5):
        added = service.add_hypothesis(
            SECONDARY_GRAPH_ID,
            HypothesisCreateRequest(
                expected_revision=revision,
                title=f"Follow-up correction branch {index + 11:02d}",
                claim=(
                    f"A bounded follow-up for branch {index + 1:02d} explains "
                    "the unresolved comparison."
                ),
                rationale="The matched result leaves one explicit alternative open.",
                predictions=[
                    f"Follow-up observable {index + 1:02d} separates the alternatives."
                ],
            ),
        )
        revision = added["graph"]["revision"]
    service.store.create_graph(
        graph_id=LARGE_GRAPH_ID,
        title="One-hundred-node rendering acceptance",
        question="Can the Research Graph remain usable at the documented 100-node density?",
        orchestration_mode="manual",
        initial_hypotheses=[
            {
                "title": f"Mechanistic branch {index + 1:02d}",
                "claim": f"Mechanistic branch {index + 1:02d} explains the retained observation.",
                "rationale": "This is a deterministic browser-performance fixture.",
                "predictions": [f"Observable series {index + 1:02d} changes reproducibly."],
            }
            for index in range(40)
        ],
    )
    large = service.presentation(LARGE_GRAPH_ID)
    large_hypotheses = [
        node for node in large["nodes"] if node["kind"] == "hypothesis"
    ]
    revision = large["graph"]["revision"]
    for index in range(30):
        experiment = service.add_experiment(
            LARGE_GRAPH_ID,
            ExperimentCreateRequest(
                expected_revision=revision,
                title=f"Discrimination experiment {index + 1:02d}",
                objective=f"Test mechanistic branch {index + 1:02d}.",
                plan_summary="Run the matched deterministic comparison.",
                decision_rule="The predefined observable separates the alternatives.",
                execution_lane="experiment",
                state="ready",
                tests_hypothesis_ids=[large_hypotheses[index]["node_id"]],
            ),
        )
        large_result = service.record_result(
            LARGE_GRAPH_ID,
            ResultCreateRequest(
                expected_revision=experiment["graph"]["revision"],
                title=f"Discrimination result {index + 1:02d}",
                summary="The predefined observable was recorded for this branch.",
                experiment_node_id=experiment["node"]["node_id"],
                judgments=[
                    {
                        "hypothesis_node_id": large_hypotheses[index]["node_id"],
                        "relation": "supports" if index % 2 == 0 else "inconclusive",
                    }
                ],
            ),
        )
        revision = large_result["graph"]["revision"]
    service.bind_thread(
        thread_id,
        graph_id=PRIMARY_GRAPH_ID,
        focus_node_id=first_result["node"]["node_id"],
    )


def _write_self_evolution_fixture(workspace: Path, thread_id: str) -> None:
    store = SelfEvolutionStore(workspace, project_id=workspace.name)
    placeholders = ",".join("?" for _ in SELF_EVOLUTION_CANDIDATE_IDS)
    observation_placeholders = ",".join("?" for _ in SELF_EVOLUTION_OBSERVATION_IDS)
    with store._connect() as connection:
        connection.execute(
            f"DELETE FROM candidates WHERE candidate_id IN ({placeholders})",
            SELF_EVOLUTION_CANDIDATE_IDS,
        )
        connection.execute(
            f"DELETE FROM observations WHERE observation_id IN ({observation_placeholders})",
            SELF_EVOLUTION_OBSERVATION_IDS,
        )
    for candidate_id in SELF_EVOLUTION_CANDIDATE_IDS:
        candidate_dir = store.candidate_dir(candidate_id)
        if candidate_dir.is_dir():
            shutil.rmtree(candidate_dir)

    observation_specs = [
        (
            "run_browser_qc_correction",
            "materials_worker/bounded-integrity-check",
            (
                "Ordinary scientific QC must not grow an unsolicited checksum "
                "step, while explicit transfer-integrity requests remain in scope."
            ),
            "files/research-summary.md",
            "explicit correction",
            (
                "Use scientific convergence and geometry evidence here; do not "
                "invent a transfer-integrity requirement."
            ),
        ),
        (
            "run_browser_qc_repeat",
            "materials_worker/checksum-every-qc",
            (
                "The complete episode exposes an unresolved integrity boundary "
                "that a proposer must not generalize to every QC review."
            ),
            "files/candidate-energies.csv",
            "scope boundary",
            (
                "The scientific result was already complete; the extra integrity "
                "step did not change the decision."
            ),
        ),
        (
            "run_browser_stale_integrity",
            "materials_worker/stale-integrity-boundary",
            (
                "The narrow integrity boundary remains useful, but its owner skill "
                "changed before this revision could be reviewed."
            ),
            "files/experiment-plan.json",
            "target changed",
            (
                "The exact target must be rebased without changing the "
                "episode-grounded activation boundary."
            ),
        ),
        (
            "run_browser_conflicting_integrity",
            "materials_worker/conflicting-integrity-boundary",
            (
                "The complete episode contains conflicting activation evidence "
                "that requires a narrower revision before canary use."
            ),
            "files/experiment-plan.json",
            "conflicting boundary",
            (
                "The proposed rule cannot yet distinguish ordinary QC from an "
                "explicit transfer-integrity request."
            ),
        ),
    ]
    observations = [
        Observation(
            observation_id=SELF_EVOLUTION_OBSERVATION_IDS[index],
            run_id=run_id,
            thread_id=thread_id,
            signal_kind="skill_revision",
            target=target,
            claim=claim,
            evidence_refs=[
                {
                    "source_ref": source_ref,
                    "reason": reason,
                    "excerpt": excerpt,
                }
            ],
            outcome_ref=source_ref,
            status="consolidated",
            created_at=f"2026-07-28T08:{index:02d}:00+00:00",
        )
        for index, (run_id, target, claim, source_ref, reason, excerpt) in enumerate(
            observation_specs
        )
    ]
    for observation in observations:
        store.write_observation(observation)

    candidate_specs = [
        {
            "candidate_id": SELF_EVOLUTION_CANDIDATE_IDS[0],
            "status": "review",
            "name": "bounded-integrity-check",
            "rationale": (
                "The complete episode shows unnecessary QC overreach and also "
                "defines when integrity verification is genuinely useful."
            ),
            "recommendation": "approve",
            "overbroad": False,
            "proposed_rule": (
                "Use an integrity check only for an explicit transfer mismatch "
                "or a direct user request; never add it to ordinary scientific QC."
            ),
        },
        {
            "candidate_id": SELF_EVOLUTION_CANDIDATE_IDS[1],
            "status": "review",
            "name": "checksum-every-qc",
            "rationale": (
                "The proposer generalized beyond the complete episode even though "
                "its boundary evidence keeps unrelated QC outside scope."
            ),
            "recommendation": "reject",
            "overbroad": True,
            "proposed_rule": "Run a checksum verification during every QC review.",
        },
        {
            "candidate_id": SELF_EVOLUTION_CANDIDATE_IDS[2],
            "status": "revision",
            "name": "stale-integrity-boundary",
            "rationale": (
                "The evidence remains relevant, but the target skill changed after "
                "this revision was prepared and the exact diff must be rebased."
            ),
            "recommendation": "needs_revision",
            "overbroad": False,
            "proposed_rule": "Apply the bounded integrity rule after rebasing.",
        },
        {
            "candidate_id": SELF_EVOLUTION_CANDIDATE_IDS[3],
            "status": "revision",
            "name": "conflicting-integrity-boundary",
            "rationale": (
                "Supporting corrections and a valid counterexample currently imply "
                "incompatible activation boundaries that require human resolution."
            ),
            "recommendation": "needs_revision",
            "overbroad": False,
            "proposed_rule": "Resolve the contradictory activation boundary first.",
        },
    ]
    for index, spec in enumerate(candidate_specs):
        candidate = LearningCandidate(
            candidate_id=str(spec["candidate_id"]),
            project_id=workspace.name,
            run_id=f"run_browser_self_evolution_{index + 1}",
            thread_id=thread_id,
            action="skill",
            status=str(spec["status"]),
            route="new_skill",
            group="materials_worker",
            name=str(spec["name"]),
            rationale=str(spec["rationale"]),
            evidence_ids=[SELF_EVOLUTION_OBSERVATION_IDS[index]],
            review={
                "recommendation": str(spec["recommendation"]),
                "summary": (
                    "The proposed behavior is assessed against the complete episode, "
                    "the current skill, and its stated activation boundary."
                ),
                "change_points": [
                    {
                        "title": "Narrow the QC activation boundary",
                        "before": "Integrity verification could appear during ordinary QC.",
                        "after": str(spec["proposed_rule"]),
                        "evidence": "The complete episode and result define the boundary.",
                        "evidence_source": "exact-target episode",
                        "impact": "Avoids unrelated work while preserving the valid narrow case.",
                    }
                ],
                "evidence_sufficiency": (
                    "The complete exact-target episode and result are linked to "
                    "the proposal; no recurrence quota is assumed."
                ),
                "scope_assessment": "One bounded materials-worker QC decision.",
                "proportionality_assessment": {
                    "status": "fail" if spec["overbroad"] else "pass",
                    "explanation": (
                        "The proposed scope still includes unrelated ordinary QC."
                        if spec["overbroad"]
                        else "The change is smaller than the repeated burden it removes."
                    ),
                },
                "counterexamples": [
                    "Retain integrity verification for explicit transfer mismatches."
                ],
                "concerns": (
                    ["The scope conflicts with the ordinary-QC counterexample."]
                    if spec["overbroad"]
                    else []
                ),
                "human_checks": [
                    "Confirm that ordinary convergence and geometry QC remain unaffected."
                ],
            },
            created_at=f"2026-07-28T08:{10 + index:02d}:00+00:00",
        )
        revision_root = store.reset_candidate_dir(candidate.candidate_id)
        proposed = (
            revision_root
            / "proposed"
            / candidate.group
            / candidate.name
        )
        proposed.mkdir(parents=True)
        (proposed / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    f"name: {candidate.name}",
                    "description: Apply one bounded materials QC activation rule.",
                    "license: project-local",
                    "compatibility: local",
                    "---",
                    f"# {candidate.name}",
                    "",
                    "## Overview",
                    str(spec["proposed_rule"]),
                    "",
                    "## Quick Start",
                    "Use this guidance only when the stated integrity boundary matches.",
                    "",
                    "## Workflow",
                    "Check the explicit request or transfer evidence, then apply the bounded rule.",
                    "",
                    "## Method-critical defaults",
                    "Do not add integrity work to ordinary convergence, geometry, energy, or mechanism QC.",
                    "",
                    "## Output Contract",
                    "Report whether the narrow boundary matched and cite the deciding evidence.",
                    "",
                    "## References",
                    "This workspace candidate is grounded in its linked observations.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        candidate.bundle_hash = hash_tree(proposed)
        store.write_candidate(candidate)
        store.write_revision_json(
            candidate.candidate_id,
            candidate.revision,
            "proposal.json",
            {
                "delta_operation": "replace",
                "evidence_ids": [SELF_EVOLUTION_OBSERVATION_IDS[index]],
                "applicability_boundary": [
                    "Explicit user request for integrity verification.",
                    "A transfer receipt reports a concrete mismatch.",
                ],
                "non_applicability": [
                    "Ordinary convergence, geometry, energy, or mechanism QC."
                ],
                "expected_step_change": str(spec["proposed_rule"]),
            },
        )
        store.write_revision_json(
            candidate.candidate_id,
            candidate.revision,
            "validation.json",
            {
                "candidate_id": candidate.candidate_id,
                "valid": True,
                "checks": ["exact target", "bounded diff", "non-applicability boundary"],
                "errors": [],
            },
        )


def create_fixture(workspace: Path) -> str:
    workspace = workspace.expanduser().resolve()
    ensure_project_space_layout(workspace, create=True)
    _write_files(workspace)
    store = ThreadStore(workspace=workspace, workspace_id=workspace.name)
    thread = store.create_thread(
        thread_id=THREAD_ID,
        title="WebUI release acceptance",
        entrypoint="research",
        meta={"permission_mode": "hitl"},
    )
    with connect_workspace_db(workspace) as connection:
        connection.execute(
            "DELETE FROM thread_messages WHERE thread_id = ?",
            (THREAD_ID,),
        )

    registry = ArtifactRegistry(workspace=workspace, workspace_id=workspace.name)
    artifact = registry.register_path(
        "files/research-summary.md",
        thread_id=THREAD_ID,
        message_id="msg_fixture_assistant_29",
        title="Catalyst screening summary",
        summary="Human-readable release fixture.",
    )
    start = time.time() - 4_800
    for pair_index in range(40):
        user_id = f"msg_fixture_user_{pair_index:02d}"
        assistant_id = f"msg_fixture_assistant_{pair_index:02d}"
        user_time = start + pair_index * 120
        assistant_time = user_time + 1
        store.append_message(
            ThreadMessage(
                id=user_id,
                thread_id=THREAD_ID,
                role="user",
                status="completed",
                created_at=user_time,
                updated_at=user_time,
                parts=[
                    MessagePart(
                        id=f"part_fixture_question_{pair_index:02d}",
                        type="text",
                        status="completed",
                        text=(
                            f"Please inspect scientific cycle {pair_index + 1} and preserve "
                            "the exact evidence. Distinguish the measured or calculated "
                            "observation from any mechanistic interpretation, report the "
                            "convergence state, and keep the source artifact recoverable.\n\n"
                            "Do not collapse the worker trace into an unsupported conclusion. "
                            "If the evidence is incomplete, say what experiment or calculation "
                            "would resolve it and why."
                        ),
                    )
                ],
            )
        )
        parts, status, meta = _assistant_parts(
            pair_index,
            workspace=workspace,
            artifact_id=artifact.artifact_id,
        )
        store.append_message(
            ThreadMessage(
                id=assistant_id,
                thread_id=THREAD_ID,
                role="assistant",
                status=status,
                created_at=assistant_time,
                updated_at=assistant_time,
                parts=parts,
                meta=meta,
            )
        )
    store.update_thread(
        thread.thread_id,
        title="WebUI release acceptance",
        status=ThreadStatus.IDLE,
        active_message_id="",
        active_run_id="",
    )
    _write_research_graph(workspace, thread.thread_id)
    _write_self_evolution_fixture(workspace, thread.thread_id)
    return thread.thread_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    print(create_fixture(args.workspace))


if __name__ == "__main__":
    main()
