from __future__ import annotations

import importlib.util
import json
import csv
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_staged_ris_and_enw_converters_preserve_complete_abstracts() -> None:
    abstract = "complete abstract segment " * 40 + "END-OF-ABSTRACT"
    for specialist, skill in (("litreview_agent", "literature-evidence-use"), ("writing_specialist", "citation-management")):
        module = _load_module(
            REPO_ROOT
            / "skills"
            / specialist
            / skill
            / "scripts"
            / "converters.py",
            f"{specialist}_academic_converters",
        )
        fields = {"TI": ["Complete metadata"], "AB": [abstract]}

        ris = module.medline_to_ris(fields)
        enw = module.medline_to_enw(fields)

        assert f"N2  - {abstract}" in ris
        assert f"%X {abstract}" in enw
        assert "END-OF-ABSTRACT" in ris
        assert "END-OF-ABSTRACT" in enw


def test_staged_openalex_fallback_preserves_all_authors_and_abstract(monkeypatch) -> None:
    module = _load_module(
        REPO_ROOT
        / "skills"
        / "writing_specialist"
        / "citation-management"
        / "scripts"
        / "academic_search.py",
        "writing_academic_search",
    )
    authors = [f"Author {index}" for index in range(12)]
    tokens = [f"token{index}" for index in range(100)]
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Complete scholarly metadata",
                "publication_year": 2026,
                "publication_date": "2026-01-01",
                "cited_by_count": 1,
                "authorships": [
                    {"author": {"display_name": author}} for author in authors
                ],
                "abstract_inverted_index": {
                    token: [index] for index, token in enumerate(tokens)
                },
                "primary_location": {"source": {"display_name": "Journal"}},
                "relevance_score": 1.0,
            }
        ]
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: _Response())

    result = module.search("catalysis", limit=1)[0]

    assert result["authors"] == authors
    assert result["abstract"] == " ".join(tokens)
    assert "score" not in result
    assert "relevance_score" not in result


@pytest.mark.parametrize("role,skill", [
    ("litreview_agent", "literature-evidence-use"),
    ("writing_specialist", "citation-management"),
])
def test_all_conversion_sources_keep_long_abstracts_and_authors(role, skill) -> None:
    root = REPO_ROOT / "skills" / role / skill / "scripts"
    module = _load_module(root / "converters.py", f"{role}_converters_all_sources")
    abstract = "Complete abstract content " * 60 + "FINAL-SENTENCE"
    authors = [{"given": "Researcher", "family": f"Family{i}"} for i in range(14)]
    work = {"title": ["Selected source"], "author": authors, "abstract": abstract}
    atom = ET.fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Source</title>'
        + f"<summary>{abstract}</summary>"
        + "".join(f'<author><name>Researcher {a["family"]}</name></author>' for a in authors)
        + "</entry></feed>"
    )
    outputs = (
        module.crossref_to_ris(work), module.crossref_to_enw(work),
        module.arxiv_to_ris(atom), module.arxiv_to_enw(atom),
        module.medline_to_ris({"AU": [a["family"] for a in authors], "AB": [abstract]}),
        module.medline_to_enw({"AU": [a["family"] for a in authors], "AB": [abstract]}),
    )
    for output in outputs:
        assert abstract in output
        for author in authors:
            assert author["family"] in output
    completed = subprocess.run(
        [sys.executable, str(root / "format-converter.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--format" in completed.stdout


@pytest.mark.parametrize("output_format", ["default", "bib", "json", "tsv", "ris", "enw", "zotero-rdf"])
def test_metadata_export_does_not_assign_scientific_support(tmp_path, output_format) -> None:
    root = REPO_ROOT / "skills/writing_specialist/citation-management/scripts"
    module = _load_module(root / "citation_records.py", "citation_records")
    abstract = "Uninspected provider abstract " * 60 + "ABSTRACT-END"
    work = {
        "title": ["A metadata candidate"], "container-title": ["Independent Journal"],
        "DOI": "10.1234/example", "published": {"date-parts": [[2026, 1, 2]]},
        "author": [{"family": f"Author{i}", "given": "First"} for i in range(14)],
        "abstract": abstract, "score": 9999,
    }
    candidate = module.candidate_from_crossref(work, "selected source")
    assert candidate.retrieval_score == 9999
    visible = candidate.as_dict()
    assert visible["claim_relation"] == "unassessed"
    assert visible["access_depth"] == "metadata"
    assert not {"score", "retrieval_score", "support_grade", "confidence"} & visible.keys()
    assert len(visible["authors"]) == 14
    assert visible["abstract"] == abstract

    source = tmp_path / "selected.json"
    source.write_text(json.dumps({"message": {"items": [work]}}), encoding="utf-8")
    output = tmp_path / "export"
    command = [
        sys.executable, str(root / "citation_records.py"), str(source), str(output),
    ]
    if output_format != "default":
        command.extend(["--format", output_format])
    subprocess.run(command, capture_output=True, text=True, check=True)
    exported = output.read_text(encoding="utf-8")
    assert "Author13" in exported
    assert "9999" not in exported
    if output_format == "json":
        assert json.loads(exported)[0]["claim_relation"] == "unassessed"
        assert abstract in exported
    elif output_format == "tsv":
        row = next(csv.DictReader(exported.splitlines(), delimiter="\t"))
        assert row["claim_relation"] == "unassessed"
        assert row["access_depth"] == "metadata"
        assert "score" not in row
    elif output_format == "zotero-rdf":
        ET.fromstring(exported)
    else:
        assert abstract in exported
    if output_format in {"default", "bib"}:
        assert exported.startswith("@article{")
        assert "access_depth" not in exported
        assert "claim_relation" not in exported
    assert {path.name for path in tmp_path.iterdir()} == {"selected.json", "export"}


@pytest.mark.parametrize("role,skill", [
    ("litreview_agent", "literature-evidence-use"),
    ("writing_specialist", "citation-management"),
])
@pytest.mark.parametrize("verbose", [False, True])
def test_format_converter_cli_defaults_to_only_bib(tmp_path, monkeypatch, capsys, role, skill, verbose) -> None:
    root = REPO_ROOT / "skills" / role / skill / "scripts"
    monkeypatch.syspath_prepend(str(root))
    module = _load_module(root / "format-converter.py", f"{role}_format_converter")
    payload = {"message": {
        "title": ["Selected source"], "author": [{"family": "Author", "given": "First"}],
        "DOI": "10.1234/example", "issued": {"date-parts": [[2026]]},
    }}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    output = tmp_path / "references"
    argv = ["format-converter.py", "--doi", "10.1234/example", "--output", str(output)]
    if verbose:
        argv.append("--verbose")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    generated = list(output.iterdir())
    assert len(generated) == 1
    assert generated[0].suffix == ".bib"
    assert generated[0].read_text().startswith("@article{")
    console = capsys.readouterr().out
    assert "Success: 1" in console
    assert str(output) in console
    assert ("Downloading DOI:" in console) == verbose
    assert ("Selected source" in console) == verbose


@pytest.mark.parametrize("verbose", [False, True])
def test_citation_validation_progress_is_opt_in(tmp_path, capsys, verbose) -> None:
    module = _load_module(
        REPO_ROOT / "skills/writing_specialist/citation-management/scripts/validate_citations.py",
        "validation_progress",
    )
    bibliography = tmp_path / "references.bib"
    bibliography.write_text('\n'.join(
        f'@article{{paper{i},\n author={{Author {i}}},\n title={{Title {i}}},\n'
        f'journal={{Journal}},\n year={{2026}},\n doi={{10.1234/example{i}}}\n}}' for i in range(100)
    ))
    report = module.CitationValidator().validate_file(str(bibliography), verbose=verbose)
    assert report["total_entries"] == 100
    console = capsys.readouterr().err
    assert ("Validating entry" in console) == verbose
    assert "Found 100 entries" in console
