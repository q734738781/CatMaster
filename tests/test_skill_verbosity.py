from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "skills/writing_specialist/citation-management/scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"verbosity_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("script", ["doi_to_bibtex", "extract_metadata", "search_pubmed", "search_google_scholar"])
def test_progress_toggle_preserves_full_cli_exports(script, verbose, tmp_path, monkeypatch, capsys):
    module = load_script(script)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    output = tmp_path / "export.txt"
    count = 3
    if script == "doi_to_bibtex":
        identifiers = [f"10.1234/source{i}" for i in range(count)]
        monkeypatch.setattr(module.DOIConverter, "doi_to_bibtex", lambda self, doi: f"@misc{{{doi}, title={{Complete title}}}}")
        args = identifiers
        progress = "Converting DOI"
    elif script == "extract_metadata":
        identifiers = tmp_path / "identifiers.txt"
        identifiers.write_text("\n".join(f"10.1234/source{i}" for i in range(count)))
        monkeypatch.setattr(module.MetadataExtractor, "extract_from_doi", lambda self, doi: {
            "doi": doi, "title": "Complete title", "authors": "First Author and Last Author", "year": "2026",
        })
        args = ["--input", str(identifiers)]
        progress = "Identified as doi"
    elif script == "search_pubmed":
        # Exercise multiple fetch batches without any external requests.
        count = 205
        pmids = [str(i) for i in range(count)]

        def get(_session, url, **kwargs):
            if "esearch" in url:
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                    "esearchresult": {"idlist": pmids, "count": str(count)},
                })
            rows = "".join(f"<PubmedArticle><PMID>{i}</PMID></PubmedArticle>" for i in kwargs["params"]["id"].split(","))
            return SimpleNamespace(raise_for_status=lambda: None, content=f"<PubmedArticleSet>{rows}</PubmedArticleSet>".encode())

        monkeypatch.setattr(module.requests.Session, "get", get)
        monkeypatch.setattr(module.PubMedSearcher, "_extract_metadata_from_xml", lambda self, article: {
            "pmid": article.findtext("PMID"), "title": "Complete title", "abstract": "Complete abstract",
        })
        args = ["example query", "--limit", str(count), "--format", "json"]
        progress = "Fetching metadata for PMIDs"
    else:
        monkeypatch.setattr(module, "SCHOLARLY_AVAILABLE", True)
        monkeypatch.setattr(module, "scholarly", SimpleNamespace(search_pubs=lambda _: iter([
            {"bib": {"title": f"Complete title {i}", "author": ["First Author", "Last Author"], "abstract": "Complete abstract"}}
            for i in range(count)
        ])), raising=False)
        args = ["example query", "--limit", str(count), "--format", "json"]
        progress = "Retrieved 1/"
    argv = [script, *args, "--output", str(output)]
    if verbose:
        argv.append("--verbose")
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()  # Discard any optional-dependency import notice.
    module.main()
    console = capsys.readouterr()
    assert (progress in console.err) == verbose
    assert str(output) in console.err
    assert console.out == ""
    exported = output.read_text()
    if script.startswith("search_"):
        rows = json.loads(exported)["results"]
        assert len(rows) == count
        assert rows[-1]["abstract"] == "Complete abstract"
    else:
        assert exported.count("@") == count
        assert "10.1234/source2" in exported


def test_nonverbose_conversion_still_reports_failure(monkeypatch, capsys):
    module = load_script("doi_to_bibtex")
    monkeypatch.setattr(module.requests.Session, "get", lambda *args, **kwargs: SimpleNamespace(status_code=404))
    assert module.DOIConverter().doi_to_bibtex("10.1234/missing") is None
    assert "DOI not found: 10.1234/missing" in capsys.readouterr().err


@pytest.mark.parametrize("saved_report,verbose", [(False, False), (True, False), (True, True)])
def test_saved_validation_report_avoids_duplicate_error_dump(saved_report, verbose, tmp_path, monkeypatch, capsys):
    module = load_script("validate_citations")
    bibliography = tmp_path / "broken.bib"
    bibliography.write_text('@article{broken,\n title={Only a title}\n}\n')
    report_path = tmp_path / "diagnostics.json"
    argv = ["validate_citations", str(bibliography)]
    if saved_report:
        argv += ["--report", str(report_path)]
    if verbose:
        argv += ["--verbose"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
    console = capsys.readouterr().out
    assert "Errors: 3" in console
    assert ("Missing required field" in console) == (verbose or not saved_report)
    if saved_report:
        assert str(report_path) in console
        report = json.loads(report_path.read_text())
        assert {row["field"] for row in report["errors"]} == {"author", "journal", "year"}
