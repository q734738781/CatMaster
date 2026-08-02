from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfWriter

from catmaster.runtime.literature import acquisition
from catmaster.runtime.self_evolution.gate import CandidateGate
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.registry import get_tool_registry


def _write_test_pdf(path: Path, *, title: str) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": title, "/Subject": "x" * 120_000})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer.write(handle)


def test_acquire_literature_source_downloads_and_verifies_selected_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)

    def _source(identifier: str, output_path: Path, config: dict):
        assert identifier == "10.1234/example"
        assert config["download_strategy"] == "legal_only"
        assert config["scihub_enabled"] is False
        _write_test_pdf(output_path, title="Operando reconstruction of catalyst A")
        return {"success": True, "file": str(output_path), "source": "test"}

    monkeypatch.setattr(acquisition, "_legal_pdf_sources", lambda kind, config: [("unpaywall", _source)])
    monkeypatch.setattr(acquisition, "_metadata_title", lambda kind, identifier: "")

    with workspace_scope(project):
        content, artifact = acquisition.acquire_literature_source(
            {
                "identifier": "https://doi.org/10.1234/example",
                "expected_title": "Operando reconstruction of catalyst A",
            }
        )

    data = artifact["data"]
    assert data["status"] == "downloaded_pdf"
    assert data["source"] == "unpaywall"
    assert data["identity_check"] == "expected_title_present_in_pdf"
    assert data["page_count"] == 2
    assert data["path"] == "literature/sources/10.1234_example.pdf"
    assert (layout["files_root"] / data["path"]).is_file()
    assert "Verified scholarly PDF" in content


def test_acquire_literature_source_rejects_mismatched_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)

    def _wrong_source(identifier: str, output_path: Path, config: dict):
        _write_test_pdf(output_path, title="An unrelated clinical trial")
        return {"success": True, "file": str(output_path), "source": "test"}

    monkeypatch.setattr(acquisition, "_legal_pdf_sources", lambda kind, config: [("unpaywall", _wrong_source)])
    monkeypatch.setattr(acquisition, "_metadata_title", lambda kind, identifier: "")
    monkeypatch.setattr(
        acquisition,
        "_save_static_page",
        lambda url, output_path: {"status": "not_found", "source": "static_http"},
    )

    with workspace_scope(project):
        _, artifact = acquisition.acquire_literature_source(
            {
                "identifier": "10.1234/example",
                "expected_title": "Operando reconstruction of catalyst A",
            }
        )

    data = artifact["data"]
    assert data["status"] == "not_found"
    assert data["attempts"][0]["source"] == "unpaywall"
    assert data["attempts"][0]["status"].startswith("title_mismatch_")
    assert not (layout["files_root"] / "literature/sources/10.1234_example.pdf").exists()


def test_acquire_literature_source_fetches_static_page_once_and_reuses_local_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)
    calls: list[str] = []

    def _open(self, url: str, max_chars: int):
        calls.append(url)
        return SimpleNamespace(
            requested_url=url,
            final_url=url,
            title="Catalyst landing page",
            description="",
            text="A detailed abstract describing operando catalyst reconstruction. " * 10,
        )

    monkeypatch.setattr(acquisition, "_legal_pdf_sources", lambda kind, config: [])
    monkeypatch.setattr(acquisition.OnlineSearchAdapter, "open_public_page", _open)

    with workspace_scope(project):
        first_content, first_artifact = acquisition.acquire_literature_source(
            {"identifier": "https://example.org/article/42"}
        )
        second_content, second_artifact = acquisition.acquire_literature_source(
            {"identifier": "https://example.org/article/42"}
        )

    assert calls == ["https://example.org/article/42"]
    assert first_artifact["data"]["status"] == "saved_text"
    assert second_artifact["data"]["status"] == "cached_text"
    assert first_artifact["data"]["path"] == second_artifact["data"]["path"]
    assert "one static public-page fetch" in first_content
    assert "do not repeatedly reopen" in second_content


def test_literature_acquisition_tool_schema_is_small_and_nonnullable() -> None:
    registry = get_tool_registry()
    openai_tool = next(
        item for item in registry.as_openai_tools() if item["name"] == "acquire_literature_source"
    )
    schema = openai_tool["parameters"]
    assert set(schema["properties"]) == {"identifier", "expected_title"}
    assert schema["properties"]["expected_title"]["type"] == "string"
    assert schema["properties"]["expected_title"]["default"] == ""
    assert "expected_title" not in schema.get("required", [])

    langchain_tool = next(
        item for item in registry.as_langchain_tools() if item.name == "acquire_literature_source"
    )
    langchain_schema = langchain_tool.args_schema
    if hasattr(langchain_schema, "model_json_schema"):
        langchain_schema = langchain_schema.model_json_schema()
    assert langchain_schema["properties"]["expected_title"]["type"] == "string"
    assert "anyOf" not in langchain_schema["properties"]["expected_title"]


def test_literature_acquisition_legal_source_inventory_excludes_browser_and_grey_routes() -> None:
    config = acquisition._scansci_config()
    config["public_url"] = "https://doi.org/10.1234/example"
    config["core_api_key"] = ""
    labels = [name for name, _ in acquisition._legal_pdf_sources("doi", config)]

    assert labels == [
        "unpaywall",
        "openalex_oa",
        "crossref_pdf",
        "semantic_scholar_oa",
        "europe_pmc",
        "pubmed_central",
        "doaj",
    ]
    assert not any(
        token in label
        for label in labels
        for token in ("browser", "publisher", "scihub", "libgen", "tor")
    )


def test_expected_title_conflict_is_not_hidden_by_a_matching_doi() -> None:
    matched, reason = acquisition._identity_match(
        kind="doi",
        identifier="10.1234/example",
        expected_title="Operando reconstruction of catalyst A",
        pdf_title="An unrelated clinical trial",
        first_pages_text="This article has DOI 10.1234/example.",
    )
    assert matched is False
    assert reason.startswith("doi_present_but_title_mismatch_")


def test_scansci_browser_is_internal_and_after_direct_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)
    order: list[str] = []

    def _direct(identifier: str, output_path: Path, config: dict):
        order.append("direct")
        return None

    def _browser(identifier: str, output_path: Path, config: dict):
        order.append("browser")
        assert config["browser_headless"] is True
        _write_test_pdf(output_path, title="Operando reconstruction of catalyst A")
        return {"success": True, "file": str(output_path), "source": "browser"}

    monkeypatch.setattr(acquisition, "_legal_pdf_sources", lambda kind, config: [("unpaywall", _direct)])
    monkeypatch.setattr(
        acquisition,
        "_browser_pdf_sources",
        lambda kind: [("scansci_browser", _browser)],
    )
    monkeypatch.setattr(acquisition, "_metadata_title", lambda kind, identifier: "")

    with workspace_scope(project):
        _, artifact = acquisition.acquire_literature_source(
            {
                "identifier": "10.1038/example",
                "expected_title": "Operando reconstruction of catalyst A",
            }
        )

    assert order == ["direct", "browser"]
    assert artifact["data"]["status"] == "downloaded_pdf"
    assert artifact["data"]["source"] == "scansci_browser"


def test_browser_fallback_inventory_is_doi_only_and_has_no_grey_sources() -> None:
    labels = [name for name, _ in acquisition._browser_pdf_sources("doi")]
    assert labels == ["scansci_browser"]
    assert acquisition._browser_pdf_sources("arxiv") == []
    assert acquisition._browser_pdf_sources("url") == []
    assert not any(token in label for label in labels for token in ("scihub", "libgen", "tor"))


def test_self_evolution_reads_the_current_litreview_tool_surface() -> None:
    surfaces = CandidateGate._runtime_tool_surfaces()["litreview_agent"]
    assert len(surfaces) == 1
    assert "acquire_literature_source" in surfaces[0]
    assert not any(name.startswith("agent_browser_") for name in surfaces[0])
