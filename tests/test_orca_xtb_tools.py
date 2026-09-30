from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.remote.cpu.native_argv_runtime import load_manifest
from catmaster.remote.cpu import orca_boot
from catmaster.remote.cpu.orca_boot import _runtime_input_text
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.analysis import analyze_orca_results, analyze_xtb_results
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.geometry_inputs import crest_prepare, orca_nebts_prepare, orca_prepare, xtb_prepare
from catmaster.tools.registry import ToolRegistry


def _project_space(tmp_path: Path) -> Path:
    project = tmp_path / "project_space"
    ensure_project_space_layout(project, create=True)
    return project


def _write_xyz(path: Path, symbols: tuple[str, ...] = ("O", "H", "H")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = {
        "O": "0.000000 0.000000 0.000000",
        "H": "0.750000 0.000000 0.500000",
        "C": "0.000000 0.000000 1.200000",
    }
    rows = [f"{symbol} {coordinates[symbol]}" for symbol in symbols]
    path.write_text(f"{len(symbols)}\nfixture\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _schema_pair(name: str) -> tuple[dict, dict]:
    registry = ToolRegistry()
    openai_schema = next(item for item in registry.as_openai_tools() if item["name"] == name)["parameters"]
    langchain_schema = registry.as_langchain_tools(allowlist=[name])[0].args_schema
    return openai_schema, langchain_schema


def test_native_quantum_prepare_schemas_are_fully_reachable_and_non_nullable() -> None:
    for name in ("xtb_prepare", "crest_prepare"):
        for schema in _schema_pair(name):
            assert set(schema["properties"]) == {"output_root", "argv", "asset_mappings"}
            assert schema["properties"]["argv"]["type"] == "array"
            assert schema["properties"]["asset_mappings"]["type"] == "array"
            assert "anyOf" not in schema["properties"]["argv"]
            assert "anyOf" not in schema["properties"]["asset_mappings"]

    for schema in _schema_pair("orca_prepare"):
        assert set(schema["properties"]) == {
            "input_path",
            "output_root",
            "simple_keywords",
            "input_blocks",
            "charge",
            "multiplicity",
        }
        assert {"input_path", "output_root", "simple_keywords", "charge", "multiplicity"} <= set(schema["required"])
        assert schema["properties"]["input_blocks"]["type"] == "array"

    names = {item["name"] for item in ToolRegistry().as_openai_tools()}
    assert "orca_nebts_prepare" in names
    assert "orca_scan_prepare" not in names
    assert "orca_optts_prepare" not in names
    assert "orca_irc_prepare" not in names
    assert "crest_conformer_search" not in names


@pytest.mark.parametrize("prepare", [xtb_prepare, crest_prepare])
def test_native_argv_prepare_preserves_exact_tokens_assets_and_dotfiles(tmp_path: Path, prepare) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "inputs" / "coord.xyz")
        (files / "inputs" / ".xcontrol").write_text("$constrain\n atoms: 1\n$end\n", encoding="utf-8")
        argv = ["inputs/coord.xyz", "--input", ".xcontrol", "literal;token", "$(not-shell)", "*.xyz"]
        content, artifact = prepare(
            {
                "output_root": "prepared/native",
                "argv": argv,
                "asset_mappings": [
                    {"source_path": "inputs/coord.xyz", "stage_path": "inputs/coord.xyz"},
                    {"source_path": "inputs/.xcontrol", "stage_path": ".xcontrol"},
                ],
            }
        )
        stage = files / artifact["data"]["stage_path"]
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        assert manifest == {"argv": argv, "input_assets": ["inputs/coord.xyz", ".xcontrol"]}
        assert (stage / ".xcontrol").read_text(encoding="utf-8").startswith("$constrain")
        assert load_manifest(stage / "manifest.json") == (argv, ["inputs/coord.xyz", ".xcontrol"])
        assert "completed" in content


def test_crest_prepare_accepts_toml_first_and_does_not_inject_subrmsd(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "inputs" / "coord.xyz")
        (files / "inputs" / "crest.toml").write_text("input = 'coord.xyz'\n", encoding="utf-8")
        _, artifact = crest_prepare(
            {
                "output_root": "prepared/crest",
                "argv": ["crest.toml"],
                "asset_mappings": [
                    {"source_path": "inputs/crest.toml", "stage_path": "crest.toml"},
                    {"source_path": "inputs/coord.xyz", "stage_path": "coord.xyz"},
                ],
            }
        )
        manifest = json.loads((files / artifact["data"]["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["argv"] == ["crest.toml"]
        assert "--subrmsd" not in manifest["argv"]


def test_xtb_and_crest_uncommon_native_options_round_trip_without_host_parsing(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "inputs" / "coord.xyz")
        (files / "inputs" / "constraints.inp").write_text("$constrain\n atoms: 1\n$end\n", encoding="utf-8")

        xtb_argv = ["coord.xyz", "--grad", "--input", "constraints.inp"]
        _, xtb_artifact = xtb_prepare(
            {
                "output_root": "prepared/xtb-uncommon",
                "argv": xtb_argv,
                "asset_mappings": [
                    {"source_path": "inputs/coord.xyz", "stage_path": "coord.xyz"},
                    {"source_path": "inputs/constraints.inp", "stage_path": "constraints.inp"},
                ],
            }
        )
        assert json.loads((files / xtb_artifact["data"]["manifest_path"]).read_text(encoding="utf-8"))["argv"] == xtb_argv

        crest_argv = ["coord.xyz", "--cinp", "constraints.inp", "-xnam", "custom-backend", "--scratch", "scratch"]
        _, crest_artifact = crest_prepare(
            {
                "output_root": "prepared/crest-uncommon",
                "argv": crest_argv,
                "asset_mappings": [
                    {"source_path": "inputs/coord.xyz", "stage_path": "coord.xyz"},
                    {"source_path": "inputs/constraints.inp", "stage_path": "constraints.inp"},
                ],
            }
        )
        staged_argv = json.loads((files / crest_artifact["data"]["manifest_path"]).read_text(encoding="utf-8"))["argv"]
        assert staged_argv == crest_argv
        assert "--subrmsd" not in staged_argv


def test_native_prepare_rejects_existing_stage_parent_traversal_and_duplicate_targets(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "inputs" / "coord.xyz")
        (files / "prepared" / "occupied").mkdir(parents=True)
        (files / "prepared" / "occupied" / "old.txt").write_text("old", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError):
            xtb_prepare({"output_root": "prepared/occupied", "argv": [], "asset_mappings": []})
        with pytest.raises(CatMasterToolExecutionError):
            xtb_prepare(
                {
                    "output_root": "prepared/traversal",
                    "argv": [],
                    "asset_mappings": [{"source_path": "inputs/coord.xyz", "stage_path": "../coord.xyz"}],
                }
            )
        with pytest.raises(CatMasterToolExecutionError):
            xtb_prepare(
                {
                    "output_root": "prepared/duplicate",
                    "argv": [],
                    "asset_mappings": [
                        {"source_path": "inputs/coord.xyz", "stage_path": "coord.xyz"},
                        {"source_path": "inputs/coord.xyz", "stage_path": "coord.xyz"},
                    ],
                }
            )


def test_orca_prepare_writes_only_authored_keywords_blocks_and_explicit_state(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "molecules" / "water.xyz")
        _, artifact = orca_prepare(
            {
                "input_path": "molecules/water.xyz",
                "output_root": "prepared/orca",
                "simple_keywords": ["r2SCAN-3c", "Opt", "Freq"],
                "input_blocks": ["%scf\n  MaxIter 400\nend"],
                "charge": -1,
                "multiplicity": 2,
            }
        )
        record = artifact["data"]["records"][0]
        text = (files / record["input_path"]).read_text(encoding="utf-8")
        assert text == "! r2SCAN-3c Opt Freq\n%scf\n  MaxIter 400\nend\n* xyzfile -1 2 input.xyz\n"
        assert "TightSCF" not in text
        assert "VeryTightSCF" not in text
        assert "TightOpt" not in text
        assert "%pal" not in text


def test_orca_prepare_requires_charge_and_multiplicity_and_rejects_duplicate_owned_blocks(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "molecules" / "water.xyz")
        with pytest.raises(CatMasterToolExecutionError):
            orca_prepare(
                {
                    "input_path": "molecules/water.xyz",
                    "output_root": "prepared/missing-state",
                    "simple_keywords": ["HF", "def2-SVP"],
                }
            )
        with pytest.raises(CatMasterToolExecutionError):
            orca_prepare(
                {
                    "input_path": "molecules/water.xyz",
                    "output_root": "prepared/duplicate-block",
                    "simple_keywords": ["HF", "def2-SVP"],
                    "input_blocks": ["%geom\nend", "%geom\nend"],
                    "charge": 0,
                    "multiplicity": 1,
                }
            )


def test_orca_prepare_rejects_periodic_inputs(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        periodic = files / "molecules" / "periodic.xyz"
        periodic.parent.mkdir(parents=True, exist_ok=True)
        periodic.write_text(
            "1\nLattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\"\nH 0 0 0\n",
            encoding="utf-8",
        )
        with pytest.raises(CatMasterToolExecutionError, match="periodic"):
            orca_prepare(
                {
                    "input_path": "molecules/periodic.xyz",
                    "output_root": "prepared/periodic",
                    "simple_keywords": ["HF", "def2-SVP"],
                    "charge": 0,
                    "multiplicity": 2,
                }
            )


def test_orca_neb_requires_mapped_endpoints_and_native_neb_block(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        _write_xyz(files / "molecules" / "reactant.xyz", ("C", "H"))
        _write_xyz(files / "molecules" / "product.xyz", ("C", "H"))
        _, artifact = orca_nebts_prepare(
            {
                "reactant_path": "molecules/reactant.xyz",
                "product_path": "molecules/product.xyz",
                "output_root": "prepared/neb",
                "simple_keywords": ["r2SCAN-3c", "NEB-TS"],
                "neb_block": "%neb\n  Product \"product.xyz\"\n  NImages 8\nend",
                "charge": 0,
                "multiplicity": 1,
            }
        )
        text = (files / artifact["data"]["input_path"]).read_text(encoding="utf-8")
        assert "NImages 8" in text
        assert text.endswith("* xyzfile 0 1 reactant.xyz\n")

        _write_xyz(files / "molecules" / "bad.xyz", ("H", "C"))
        with pytest.raises(CatMasterToolExecutionError, match="element ordering"):
            orca_nebts_prepare(
                {
                    "reactant_path": "molecules/reactant.xyz",
                    "product_path": "molecules/bad.xyz",
                    "output_root": "prepared/bad-neb",
                    "simple_keywords": ["r2SCAN-3c", "NEB-TS"],
                    "neb_block": "%neb\n Product \"product.xyz\"\nend",
                    "charge": 0,
                    "multiplicity": 1,
                }
            )


@pytest.mark.parametrize(
    ("canonical", "expected_fragment"),
    [
        ("! HF def2-SVP PAL8\n* xyzfile 0 1 input.xyz\n", "%pal\n  nprocs 4\nend"),
        ("! HF def2-SVP\n%pal nprocs 8 end\n* xyzfile 0 1 input.xyz\n", "%pal nprocs 4 end"),
        ("! HF def2-SVP\n%pal\n  nprocs 8\n  bind true\nend\n* xyzfile 0 1 input.xyz\n", "nprocs 4\n  bind true"),
        (
            "! HF def2-SVP\n%pal nprocs 8\n  nprocs_group 2\nend\n* xyzfile 0 1 input.xyz\n",
            "%pal nprocs 4\n  nprocs_group 2",
        ),
        (
            "! HF def2-SVP PAL8(4x2)\n* xyzfile 0 1 input.xyz\n",
            "%pal\n  nprocs 4\n  nprocs_group 2\nend",
        ),
        (
            "! HF def2-SVP\n%pal nprocs_world 8 end\n* xyzfile 0 1 input.xyz\n",
            "%pal nprocs 4 end",
        ),
    ],
)
def test_orca_runtime_reconciles_pal_without_mutating_science(canonical: str, expected_fragment: str) -> None:
    runtime = _runtime_input_text(canonical, 4)
    assert expected_fragment in runtime
    assert "HF def2-SVP" in runtime
    assert "* xyzfile 0 1 input.xyz" in runtime
    assert "PAL8" not in runtime


def test_orca_runtime_rejects_conflicting_parallel_authorship() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        _runtime_input_text("! HF PAL4\n%pal nprocs 8 end\n* xyzfile 0 1 input.xyz\n", 2)
    with pytest.raises(ValueError, match="Multiple nprocs"):
        _runtime_input_text(
            "! HF\n%pal\n  nprocs 4\n  nprocs 4\nend\n* xyzfile 0 1 input.xyz\n",
            2,
        )
    with pytest.raises(ValueError, match="not divisible"):
        _runtime_input_text("! HF PAL8(4x2)\n* xyzfile 0 1 input.xyz\n", 3)
    with pytest.raises(ValueError, match="Conflicting"):
        _runtime_input_text("! HF PAL4\n! def2-SVP PAL8\n* xyzfile 0 1 input.xyz\n", 4)


def test_orca_runtime_reconciles_pal_token_on_continuation_simple_line() -> None:
    runtime = _runtime_input_text(
        "! HF\n! def2-SVP PAL8\n* xyzfile 0 1 input.xyz\n",
        4,
    )

    assert "PAL8" not in runtime
    assert "%pal\n  nprocs 4\nend" in runtime


def test_orca_boot_keeps_canonical_input_byte_identical(tmp_path: Path, monkeypatch) -> None:
    canonical = b"! HF def2-SVP PAL8\n* xyzfile 0 1 input.xyz\n"
    (tmp_path / "job.inp").write_bytes(canonical)

    def _fake_run(command, **kwargs):
        kwargs["stdout"].write("*** ORCA TERMINATED NORMALLY ***\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setattr(orca_boot.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", [orca_boot.__file__, "--input", "job.inp", "--orca_bin", "/bin/true"])

    assert orca_boot.main() == 0
    assert (tmp_path / "job.inp").read_bytes() == canonical
    assert "%pal\n  nprocs 4\nend" in (tmp_path / "job.runtime.inp").read_text(encoding="utf-8")


def test_orca_property_conversion_uses_documented_property_invocation(tmp_path: Path, monkeypatch) -> None:
    captured: list[list[str]] = []
    (tmp_path / "job.runtime.property.txt").write_text("property data\n", encoding="utf-8")

    def _fake_run(command, **kwargs):
        captured.append(command)
        assert kwargs["shell"] is False
        (tmp_path / "job.runtime.property.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orca_boot.shutil, "which", lambda name: "/opt/orca_2json" if name == "orca_2json" else None)
    monkeypatch.setattr(orca_boot.subprocess, "run", _fake_run)

    orca_boot._try_orca_2json("job.runtime")

    assert captured == [["/opt/orca_2json", "job.runtime", "-property"]]


def test_xtb_analysis_uses_last_energy_and_preserves_missing_frequency_semantics(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "xtb"
        run.mkdir(parents=True)
        (run / "xtb_summary.json").write_text(
            json.dumps({"returncode": 0, "normal_termination": True, "log_file": "xtb_stdout.out"}),
            encoding="utf-8",
        )
        (run / "xtb_stdout.out").write_text(
            "TOTAL ENERGY -1.0\nTOTAL ENERGY -2.5\nnormal termination of xtb\n",
            encoding="utf-8",
        )
        _, artifact = analyze_xtb_results({"result_root": "results/xtb"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json"]).read_text(encoding="utf-8"))
        record = payload["records"][0]
        assert record["energy_hartree"] == -2.5
        assert record["task_state"] == "not_applicable"
        assert record["frequency_state"] == "not_calculated"
        assert record["imaginary_frequency_count"] is None
        assert "relative_energy" not in record


def test_crest_analysis_maps_energy_rows_to_ensemble_frames(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "crest"
        run.mkdir(parents=True)
        (run / "crest_summary.json").write_text(
            json.dumps({"returncode": 0, "normal_termination": True, "log_file": "crest_stdout.out"}),
            encoding="utf-8",
        )
        (run / "crest_stdout.out").write_text("CREST terminated normally\n", encoding="utf-8")
        (run / "xtbopt.xyz").write_text("1\npreopt\nH 9 0 0\n", encoding="utf-8")
        (run / "crest_best.xyz").write_text("1\nbest\nH 0 0 0\n", encoding="utf-8")
        (run / "crest.energies").write_text("1 -10.0\n2 -9.5\n", encoding="utf-8")
        (run / "crest_conformers.xyz").write_text(
            "1\na\nH 0 0 0\n1\nb\nH 0 0 0.1\n",
            encoding="utf-8",
        )
        _, artifact = analyze_xtb_results({"result_root": "results/crest"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json"]).read_text(encoding="utf-8"))
        record = payload["records"][0]
        table = record["conformer_energies"]
        assert table["property_state"] == "calculated"
        assert table["ensemble_frames"] == 2
        assert [row["energy"] for row in table["rows"]] == [-10.0, -9.5]
        assert record["task_state"] == "not_applicable"
        assert record["final_structure"].endswith("crest_best.last.xyz")


def test_crest_analysis_maps_documented_crest_ensemble_file(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "crest_sort"
        run.mkdir(parents=True)
        (run / "crest_summary.json").write_text(
            json.dumps({"returncode": 0, "normal_termination": True, "log_file": "crest_stdout.out"}),
            encoding="utf-8",
        )
        (run / "crest_stdout.out").write_text("CREST terminated normally\n", encoding="utf-8")
        (run / "crest.energies").write_text("0.000\n0.550\n", encoding="utf-8")
        (run / "crest_ensemble.xyz").write_text(
            "1\na\nH 0 0 0\n1\nb\nH 0 0 0.1\n",
            encoding="utf-8",
        )
        _, artifact = analyze_xtb_results({"result_root": "results/crest_sort"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json"]).read_text(encoding="utf-8"))
        table = payload["records"][0]["conformer_energies"]

    assert table["property_state"] == "calculated"
    assert table["ensemble_path"].endswith("crest_ensemble.xyz")
    assert [row["conformer_index"] for row in table["rows"]] == [1, 2]


def test_orca_analysis_separates_execution_optimization_frequency_and_shielding(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "orca"
        run.mkdir(parents=True)
        (run / "job.inp").write_text("! PBE0 def2-TZVP NMR\n! Opt\n* xyzfile 0 1 input.xyz\n", encoding="utf-8")
        (run / "orca_summary.json").write_text(
            json.dumps({"returncode": 0, "normal_termination": True}), encoding="utf-8"
        )
        (run / "job.out").write_text(
            "FINAL SINGLE POINT ENERGY -5.0\n"
            "FINAL SINGLE POINT ENERGY -5.5\n"
            "CHEMICAL SHIELDING SUMMARY\n"
            " 0 C 123.4\n\n"
            "*** ORCA TERMINATED NORMALLY ***\n",
            encoding="utf-8",
        )
        _, artifact = analyze_orca_results({"result_root": "results/orca"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json"]).read_text(encoding="utf-8"))
        record = payload["records"][0]
        assert record["execution_state"] == "completed"
        assert record["task_state"] == "not_converged"
        assert record["final_energy_hartree"] == -5.5
        assert record["frequency_state"] == "not_calculated"
        assert record["imaginary_frequency_count"] is None
        assert record["nmr_isotropic_shieldings"][0]["isotropic_shielding_ppm"] == 123.4
        assert "chemical_shift" not in json.dumps(record)


def test_orca_analysis_uses_documented_final_property_json_geometry(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "orca_json"
        run.mkdir(parents=True)
        (run / "job.inp").write_text("! PBE0 def2-TZVP\n* xyzfile 0 1 input.xyz\n", encoding="utf-8")
        (run / "orca_summary.json").write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "normal_termination": True,
                    "runtime_input": "job.runtime.inp",
                }
            ),
            encoding="utf-8",
        )
        (run / "job.out").write_text(
            "FINAL SINGLE POINT ENERGY -4.0\n*** ORCA TERMINATED NORMALLY ***\n",
            encoding="utf-8",
        )
        (run / "job.runtime.property.json").write_text(
            json.dumps(
                {
                    "Calculation_Status": {"Status": "NORMAL TERMINATION"},
                    "Geometries": [
                        {"Single_Point_Data": {"finalenergy": -5.0, "converged": True}},
                        {"Single_Point_Data": {"finalenergy": -5.75, "converged": True}},
                    ],
                }
            ),
            encoding="utf-8",
        )

        _, artifact = analyze_orca_results({"result_root": "results/orca_json"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json"]).read_text(encoding="utf-8"))
        record = payload["records"][0]

    assert record["final_energy_hartree"] == -5.75
    assert record["energy_source"] == "property_json"
    assert record["scf_state"] == "converged"
    assert record["task_state"] == "not_applicable"


def test_orca_analysis_does_not_infer_neb_convergence_from_normal_exit(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "orca_neb"
        run.mkdir(parents=True)
        (run / "job.inp").write_text(
            "! r2SCAN-3c NEB-TS\n* xyzfile 0 1 reactant.xyz\n",
            encoding="utf-8",
        )
        (run / "orca_summary.json").write_text(
            json.dumps({"returncode": 0, "normal_termination": True}),
            encoding="utf-8",
        )
        (run / "job.out").write_text(
            "FINAL SINGLE POINT ENERGY -5.0\n*** ORCA TERMINATED NORMALLY ***\n",
            encoding="utf-8",
        )

        _, artifact = analyze_orca_results({"result_root": "results/orca_neb"})
        payload = json.loads(
            (project / "files" / artifact["data"]["summary_json"]).read_text(encoding="utf-8")
        )
        record = payload["records"][0]

    assert record["execution_state"] == "completed"
    assert record["scf_state"] == "converged"
    assert record["task_state"] == "unknown"


def test_orca_task_keyword_reference_does_not_anchor_an_electronic_method() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    reference = (
        repo_root
        / "skills"
        / "orca_xtb_worker"
        / "orca-optfreq-thermochemistry"
        / "references"
        / "orca_native_examples.md"
    ).read_text(encoding="utf-8")

    forbidden_method_tokens = ("pbe0", "b3lyp", "wb97", "r2scan", "def2-", "d3", "d4")
    lowered = reference.lower()
    assert not any(token in lowered for token in forbidden_method_tokens)
    assert 'simple_keywords += ["Opt"]' in reference
    assert 'simple_keywords += ["Freq"]' in reference


def test_orca_skill_exposes_task_dependent_method_selection() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skill = (
        repo_root
        / "skills"
        / "orca_xtb_worker"
        / "orca-optfreq-thermochemistry"
        / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "### Electronic method selection" in skill
    assert "r2SCAN-3c" in skill
    assert "WB97M-V/def2-TZVPP" in skill
    assert "Do not choose B3LYP as the unprescribed routine default" in skill
    assert "A method recommendation does not authorize an optimization or frequency stage" in skill
