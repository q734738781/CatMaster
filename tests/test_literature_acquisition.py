from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfWriter

from catmaster.runtime.literature import acquisition
from catmaster.runtime.tool_runtime import toolcall_context
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
    assert set(schema["properties"]) == {
        "identifier",
        "expected_title",
        "include_supplementary",
    }
    assert schema["properties"]["expected_title"]["type"] == "string"
    assert schema["properties"]["expected_title"]["default"] == ""
    assert schema["properties"]["include_supplementary"]["type"] == "boolean"
    assert schema["properties"]["include_supplementary"]["default"] is False
    assert "expected_title" not in schema.get("required", [])
    assert "include_supplementary" not in schema.get("required", [])

    langchain_tool = next(
        item for item in registry.as_langchain_tools() if item.name == "acquire_literature_source"
    )
    langchain_schema = langchain_tool.args_schema
    if hasattr(langchain_schema, "model_json_schema"):
        langchain_schema = langchain_schema.model_json_schema()
    assert langchain_schema["properties"]["expected_title"]["type"] == "string"
    assert "anyOf" not in langchain_schema["properties"]["expected_title"]
    assert langchain_schema["properties"]["include_supplementary"]["type"] == "boolean"
    assert "anyOf" not in langchain_schema["properties"]["include_supplementary"]


def test_batch_literature_acquisition_schema_has_two_nonnullable_inputs_and_hard_limit() -> None:
    registry = get_tool_registry()
    openai_tool = next(
        item
        for item in registry.as_openai_tools()
        if item["name"] == "batch_acquire_literature_sources"
    )
    schema = openai_tool["parameters"]

    assert set(schema["properties"]) == {
        "identifiers",
        "input_path",
        "include_supplementary",
        "expected_titles",
    }
    assert schema["properties"]["identifiers"]["type"] == "array"
    assert schema["properties"]["identifiers"]["maxItems"] == 50
    assert schema["properties"]["input_path"]["type"] == "string"
    assert schema["properties"]["input_path"]["default"] == ""
    assert schema["properties"]["include_supplementary"]["type"] == "boolean"
    assert schema["properties"]["include_supplementary"]["default"] is False
    assert "anyOf" not in schema["properties"]["identifiers"]
    assert "anyOf" not in schema["properties"]["input_path"]

    langchain_tool = next(
        item
        for item in registry.as_langchain_tools()
        if item.name == "batch_acquire_literature_sources"
    )
    langchain_schema = langchain_tool.args_schema
    if hasattr(langchain_schema, "model_json_schema"):
        langchain_schema = langchain_schema.model_json_schema()
    assert langchain_schema["properties"]["identifiers"]["maxItems"] == 50
    assert "anyOf" not in langchain_schema["properties"]["identifiers"]
    assert "anyOf" not in langchain_schema["properties"]["input_path"]


def test_batch_literature_acquisition_reads_file_normalizes_and_deduplicates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    input_path = layout["files_root"] / "selected.tsv"
    input_path.write_text(
        "doi\ttitle\n"
        "DOI:10.1234/ONE\tFirst\n"
        "https://doi.org/10.1234/one\tDuplicate\n"
        "10.1234/two\tSecond\n",
        encoding="utf-8",
    )
    calls: list[dict] = []

    def _acquire(payload: dict):
        calls.append(dict(payload))
        identifier = payload["identifier"]
        status = "downloaded_pdf" if identifier.endswith("/one") else "not_found"
        data = {
            "status": status,
            "source": "test",
            "identifier": identifier,
        }
        if status == "downloaded_pdf":
            data["path"] = f"literature/sources/{identifier.replace('/', '_')}.pdf"
        else:
            data["reason"] = "not available"
        return "done", {"tool_name": "acquire_literature_source", "data": data}

    monkeypatch.setattr(acquisition, "acquire_literature_source", _acquire)

    with workspace_scope(project):
        content, artifact = acquisition.batch_acquire_literature_sources(
            {"input_path": "selected.tsv"}
        )

    assert calls == [
        {"identifier": "10.1234/one", "expected_title": "", "include_supplementary": False},
        {"identifier": "10.1234/two", "expected_title": "", "include_supplementary": False},
    ]
    data = artifact["data"]
    assert data["status"] == "partial"
    assert data["input_source"] == "selected.tsv"
    assert data["requested_count"] == 3
    assert data["unique_count"] == 2
    assert data["duplicate_count"] == 1
    assert data["status_counts"] == {"downloaded_pdf": 1, "not_found": 1}
    assert "Normalized duplicates skipped: 1" in content
    assert "10.1234/two: not_found — not available" in content


def test_batch_literature_acquisition_accepts_direct_list_and_forwards_si(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    def _acquire(payload: dict):
        calls.append(dict(payload))
        return "done", {
            "tool_name": "acquire_literature_source",
            "data": {
                "status": "cached_pdf",
                "source": "workspace_cache",
                "path": "literature/sources/example.pdf",
            },
        }

    monkeypatch.setattr(acquisition, "acquire_literature_source", _acquire)

    _, artifact = acquisition.batch_acquire_literature_sources(
        {
            "identifiers": ["https://doi.org/10.1234/EXAMPLE"],
            "include_supplementary": True,
        }
    )

    assert calls == [
        {"identifier": "10.1234/example", "expected_title": "", "include_supplementary": True}
    ]
    assert artifact["data"]["status"] == "ok"
    assert artifact["data"]["input_source"] == "direct"


def test_batch_literature_acquisition_rejects_more_than_fifty_file_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    input_path = layout["files_root"] / "too-many.txt"
    input_path.write_text(
        "\n".join(f"10.1234/item-{index}" for index in range(51)) + "\n",
        encoding="utf-8",
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        acquisition,
        "acquire_literature_source",
        lambda payload: calls.append(dict(payload)),
    )

    with workspace_scope(project):
        content, artifact = acquisition.batch_acquire_literature_sources(
            {"input_path": "too-many.txt"}
        )

    assert artifact["data"]["status"] == "invalid_request"
    assert "hard limit of 50" in content
    assert calls == []


def test_batch_literature_acquisition_requires_exactly_one_input_source() -> None:
    _, empty = acquisition.batch_acquire_literature_sources({})
    _, both = acquisition.batch_acquire_literature_sources(
        {"identifiers": ["10.1234/example"], "input_path": "selected.txt"}
    )

    assert empty["data"]["status"] == "invalid_request"
    assert both["data"]["status"] == "invalid_request"
    assert "exactly one" in empty["data"]["message"]
    assert "exactly one" in both["data"]["message"]


def test_literature_acquisition_legal_source_inventory_excludes_browser_and_grey_routes() -> None:
    config = acquisition._scansci_config()
    config["public_url"] = "https://doi.org/10.1234/example"
    config["identifier"] = "10.1234/example"
    config["core_api_key"] = ""
    config["elsevier_api_key"] = ""
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


def test_configured_elsevier_api_is_a_direct_legal_source() -> None:
    config = acquisition._scansci_config()
    config["public_url"] = "https://doi.org/10.1016/j.neunet.2026.108582"
    config["identifier"] = "10.1016/j.neunet.2026.108582"
    config["core_api_key"] = ""
    config["elsevier_api_key"] = "configured-for-test"

    labels = [name for name, _ in acquisition._legal_pdf_sources("doi", config)]

    assert labels[-1] == "elsevier_api"
    assert labels[-2] == "doaj"
    assert not any(
        token in label
        for label in labels
        for token in ("browser", "scihub", "libgen", "tor")
    )


def test_scansci_config_forwards_elsevier_environment_without_nullable_fields(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ELSEVIER_API_KEY", "  test-key  ")
    monkeypatch.setenv("ELSEVIER_INSTTOKEN", "  test-token  ")

    config = acquisition._scansci_config()

    assert config["elsevier_api_key"] == "test-key"
    assert config["elsevier_insttoken"] == "test-token"


def test_acquire_literature_source_can_add_supplementary_files_after_main_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    supplementary_calls: list[str] = []

    def _source(identifier: str, output_path: Path, config: dict):
        _write_test_pdf(output_path, title="Operando reconstruction of catalyst A")
        return {"success": True, "file": str(output_path), "source": "test"}

    def _fetch_supplementary(doi: str, output_dir: Path, config: dict):
        supplementary_calls.append(doi)
        path = Path(output_dir) / "dataset.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"supplementary-data")
        return [str(path)]

    monkeypatch.setattr(
        acquisition,
        "_legal_pdf_sources",
        lambda kind, config: [("unpaywall", _source)],
    )
    monkeypatch.setattr(acquisition, "_metadata_title", lambda kind, identifier: "")
    import scansci_pdf.supplementary

    monkeypatch.setattr(
        scansci_pdf.supplementary,
        "fetch_supplementary",
        _fetch_supplementary,
    )

    payload = {
        "identifier": "10.1234/example",
        "expected_title": "Operando reconstruction of catalyst A",
    }
    with workspace_scope(project):
        _, main_artifact = acquisition.acquire_literature_source(payload)
        content, complete_artifact = acquisition.acquire_literature_source(
            {**payload, "include_supplementary": True}
        )

    assert main_artifact["data"]["status"] == "downloaded_pdf"
    assert complete_artifact["data"]["status"] == "cached_pdf"
    assert complete_artifact["data"]["supplementary_status"] == "downloaded"
    assert complete_artifact["data"]["supplementary_paths"] == [
        "literature/sources/10.1234_example_supplementary/dataset.xlsx"
    ]
    assert supplementary_calls == ["10.1234/example"]
    supplementary_path = complete_artifact["data"]["supplementary_paths"][0]
    assert (layout["files_root"] / supplementary_path).is_file()
    assert "Supplementary Information: downloaded (1 file(s))" in content


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


def test_scansci_runtime_contract_uses_the_pinned_browser_backend() -> None:
    acquisition._require_scansci_version()
    assert acquisition._scansci_config()["browser_backend"] == "patchright"
    assert [name for name, _ in acquisition._browser_pdf_sources("doi")] == [
        "scansci_browser"
    ]


def test_identifier_normalization_uses_one_canonical_acquisition_identity() -> None:
    assert acquisition._normalize_identifier("doi:10.1234/ABC") == (
        "doi",
        "10.1234/abc",
        "https://doi.org/10.1234/abc",
    )
    assert acquisition._normalize_identifier("https://doi.org/10.1234/ABC") == (
        "doi",
        "10.1234/abc",
        "https://doi.org/10.1234/abc",
    )
    assert acquisition._normalize_identifier("arXiv:2401.01234v2") == (
        "arxiv",
        "2401.01234v2",
        "https://arxiv.org/abs/2401.01234v2",
    )
    assert acquisition._normalize_identifier("arXiv:2401.01234v1")[1] != (
        acquisition._normalize_identifier("arXiv:2401.01234v2")[1]
    )
    assert acquisition._normalize_identifier("PMID: 123456") == (
        "pmid",
        "123456",
        "https://pubmed.ncbi.nlm.nih.gov/123456/",
    )
    assert acquisition._normalize_identifier(
        "https://Example.ORG:443/a%20b?q=1#fragment"
    ) == (
        "url",
        "https://example.org/a%20b?q=1",
        "https://example.org/a%20b?q=1",
    )


def test_run_scoped_acquisition_cache_is_shared_across_parent_and_worker_contexts(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)
    acquisition._reset_acquisition_cache_for_tests()
    calls: list[str] = []

    def _operation() -> tuple[str, dict]:
        calls.append("fetch")
        return acquisition._tool_result(
            {
                "status": "saved_text",
                "source": "static_http",
                "path": "literature/sources/paper.md",
            }
        )

    with workspace_scope(project), toolcall_context(
        "parent_call",
        context={"run_id": "parent_run", "search_scope": "research_run_42"},
    ):
        first = acquisition._run_cached_acquisition(
            kind="doi",
            identifier="10.1234/example",
            operation=_operation,
        )
    with workspace_scope(project), toolcall_context(
        "worker_call",
        context={"run_id": "worker_run", "search_scope": "research_run_42"},
    ):
        second = acquisition._run_cached_acquisition(
            kind="doi",
            identifier="10.1234/example",
            operation=_operation,
        )

    assert calls == ["fetch"]
    assert first[1]["data"]["status"] == "saved_text"
    assert second[1]["data"]["status"] == "cached_text"
    assert second[1]["data"]["run_cache_hit"] is True


def test_concurrent_acquisition_reuses_one_inflight_operation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)
    acquisition._reset_acquisition_cache_for_tests()
    operation_started = threading.Event()
    release_operation = threading.Event()
    calls: list[str] = []
    results: list[tuple[str, dict]] = []

    def _operation() -> tuple[str, dict]:
        calls.append("fetch")
        operation_started.set()
        assert release_operation.wait(timeout=2)
        return acquisition._tool_result(
            {
                "status": "downloaded_pdf",
                "source": "unpaywall",
                "path": "literature/sources/paper.pdf",
                "page_count": 2,
                "identity_check": "expected_title_present_in_pdf",
            }
        )

    def _worker(call_id: str) -> None:
        with workspace_scope(project), toolcall_context(
            call_id,
            context={"run_id": call_id, "search_scope": "research_run_99"},
        ):
            results.append(
                acquisition._run_cached_acquisition(
                    kind="doi",
                    identifier="10.1234/concurrent",
                    operation=_operation,
                )
            )

    first = threading.Thread(target=_worker, args=("parent",))
    second = threading.Thread(target=_worker, args=("worker",))
    first.start()
    assert operation_started.wait(timeout=2)
    second.start()
    second.join(timeout=0.1)
    assert second.is_alive()
    release_operation.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert calls == ["fetch"]
    assert sorted(item[1]["data"]["status"] for item in results) == [
        "cached_pdf",
        "downloaded_pdf",
    ]


def test_acquisition_retries_only_transient_or_materially_changed_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)
    acquisition._reset_acquisition_cache_for_tests()
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    calls: list[str] = []

    def _not_found() -> tuple[str, dict]:
        calls.append("deterministic")
        return acquisition._tool_result(
            {
                "status": "not_found",
                "attempts": [{"source": "unpaywall", "status": "not_found"}],
            }
        )

    def _transient() -> tuple[str, dict]:
        calls.append("transient")
        return acquisition._tool_result(
            {
                "status": "not_found",
                "attempts": [
                    {"source": "unpaywall", "status": "source_error:TimeoutError"}
                ],
            }
        )

    with workspace_scope(project), toolcall_context(
        "call",
        context={"search_scope": "research_run_retry"},
    ):
        acquisition._run_cached_acquisition(
            kind="doi", identifier="10.1234/stable", operation=_not_found
        )
        cached = acquisition._run_cached_acquisition(
            kind="doi", identifier="10.1234/stable", operation=_not_found
        )
        acquisition._run_cached_acquisition(
            kind="doi", identifier="10.1234/transient", operation=_transient
        )
        acquisition._run_cached_acquisition(
            kind="doi", identifier="10.1234/transient", operation=_transient
        )
        monkeypatch.setenv("CORE_API_KEY", "test-route-enabled")
        acquisition._run_cached_acquisition(
            kind="doi",
            identifier="10.1234/stable",
            operation=_not_found,
        )

    assert cached[1]["data"]["run_cache_hit"] is True
    assert calls == [
        "deterministic",
        "transient",
        "transient",
        "deterministic",
    ]
