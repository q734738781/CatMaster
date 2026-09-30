import json

from catmaster.research.knowledge_graph.context import ResearchGraphContextBuilder
from catmaster.research.knowledge_graph.models import GraphCreateRequest, ResultCreateRequest
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.storage import connect_workspace_db


def test_methods_result_conclusion_roundtrip_update_and_old_record(tmp_path):
    service = ResearchGraphService(workspace=tmp_path)
    graph = service.create_graph(GraphCreateRequest(question="Which evidence discriminates Cu mechanisms?"))["graph"]
    gid = graph["graph_id"]
    result = service.record_result(gid, ResultCreateRequest(expected_revision=graph["revision"],
        summary="Only the tested representation has poor held-out fit.",
        methods="Scaffold-held-out comparison of descriptors and Morgan fingerprints.",
        conclusion="The small descriptor model does not rule out ee predictability."))["node"]
    rid = result["node_id"]
    # A summary-only correction must not erase the method and interpretation.
    updated = service.update_bound_result(graph_id=gid, experiment_node_id="", result_node_id=rid,
        summary="Corrected held-out outcome.", title="", refs=[])["node"]
    assert updated["body"]["methods"] == result["body"]["methods"]
    assert updated["body"]["conclusion"] == result["body"]["conclusion"]
    corrected = service.update_bound_result(graph_id=gid, experiment_node_id="", result_node_id=rid,
        summary=updated["body"]["summary"], title="", refs=[], conclusion="More data are needed.")["node"]
    assert corrected["body"]["methods"] == result["body"]["methods"]
    assert corrected["body"]["conclusion"] == "More data are needed."
    context = ResearchGraphContextBuilder(workspace=tmp_path, store=service.store).build(gid, focus_node_id=rid)
    assert "Scaffold-held-out" in json.dumps(context)
    # Simulate an old JSON record; the current typed read uses an explicit gap.
    with connect_workspace_db(tmp_path) as conn:
        conn.execute("UPDATE research_nodes SET body_json=? WHERE node_id=?", (json.dumps({"summary": "Old result"}), rid))
    old = service.store.get_node(gid, rid)
    assert old["body"]["methods"] == old["body"]["conclusion"] == "missing due to old record"
