// agent_task is a legacy persisted role. Native async specialists come from
// durable execution state and are rendered in Task Context, not as navigation rows.
const HIDDEN_NAVIGATION_ROLES = new Set([
  "research_planning",
  "research_comparison",
  "agent_task",
]);
const CHILD_ROLES = new Set(["research_execution"]);

export const RELATED_RESEARCH_ACTIVITY_ID = "related-research-activity";

export const ACTIVE_RESEARCH_STATES = new Set([
  "planning",
  "comparing",
  "reviewing",
  "validating",
  "running",
  "waiting",
  "waiting_continue",
  "waiting_review",
  "operationally_incomplete",
  "blocked",
]);

export const RESEARCH_GRAPH_ACTIVITY_EVENTS = Object.freeze([
  "research_graph.updated",
  "research_graph.planning_started",
  "research_graph.planning_attached",
  "research_graph.planning_preview",
  "research_graph.planning_finished",
  "research_graph.planning_no_change",
  "research_graph.planning_stale",
  "graph.created",
  "graph.updated",
  "hypothesis.added",
  "experiment.added",
  "experiment.adopted",
  "experiment.resumed",
  "node.updated",
  "planning.branches_admitted",
  "experiment.launch_claimed",
  "experiment.blocked",
  "launch.claimed",
  "launch.submitting",
  "launch.running",
  "launch.completed",
  "launch.blocked",
  "launch.unknown",
  "launch.incomplete",
  "result.recorded",
  "result.updated",
  "result.retracted",
  "result.judgment_updated",
  "claim.revised",
  "research.disposition",
  "research.reconsidered",
  "edge.added",
  "ref.added",
]);

export function researchStateLabel(value) {
  const state = String(value || "idle").toLowerCase();
  return {
    idle: "Ready",
    planning: "Planning",
    comparing: "Comparing experiments",
    reviewing: "Independent scientific reconsideration",
    validating: "Bounded validation",
    parked: "Open premises saved",
    running: "Running",
    waiting: "Waiting — research unfinished",
    waiting_continue: "Waiting to continue",
    waiting_review: "Waiting for review",
    operationally_incomplete: "Needs attention",
    paused: "Research paused",
    blocked: "Blocked",
    completed: "Completed",
    related: "Older activity",
  }[state] || state.replace(/[_-]+/g, " ");
}

function titleMatches(thread, query) {
  if (!query) return true;
  return String(thread?.title || "").toLowerCase().includes(query);
}

export function buildResearchSessionRows(threads, query = "") {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  const visible = (Array.isArray(threads) ? threads : []).filter(
    (thread) => !HIDDEN_NAVIGATION_ROLES.has(String(thread?.thread_role || "primary")),
  );
  const byId = new Map(visible.map((thread) => [thread.thread_id, thread]));
  const childrenByParent = new Map();
  visible.forEach((thread) => {
    if (
      !CHILD_ROLES.has(String(thread?.thread_role || ""))
      || !thread?.parent_thread_id
      || !byId.has(thread.parent_thread_id)
    ) return;
    const rows = childrenByParent.get(thread.parent_thread_id) || [];
    rows.push(thread);
    childrenByParent.set(thread.parent_thread_id, rows);
  });

  const orphanExecutions = visible.filter((thread) => (
    CHILD_ROLES.has(String(thread?.thread_role || ""))
    && (!thread.parent_thread_id || !byId.has(thread.parent_thread_id))
  ));
  const topLevel = visible.filter(
    (thread) => !CHILD_ROLES.has(String(thread?.thread_role || "")),
  );

  const rows = topLevel.flatMap((root) => {
    const children = [...(childrenByParent.get(root.thread_id) || [])].sort(
      (left, right) => Number(right.updated_at || 0) - Number(left.updated_at || 0),
    );
    const rootMatch = titleMatches(root, normalizedQuery);
    const matchingChildren = children.filter((child) => titleMatches(child, normalizedQuery));
    if (normalizedQuery && !rootMatch && !matchingChildren.length) return [];
    return [{
      root,
      children: normalizedQuery && !rootMatch ? matchingChildren : children,
      matchingChildIds: new Set(matchingChildren.map((child) => child.thread_id)),
      rootMatch,
      isResearchSession: String(root?.thread_role || "") === "research_root",
      isRelatedGroup: false,
    }];
  });
  const relatedTitle = "Related research activity";
  const relatedRootMatch = !normalizedQuery || relatedTitle.toLowerCase().includes(normalizedQuery);
  const relatedChildren = orphanExecutions.filter(
    (child) => relatedRootMatch || titleMatches(child, normalizedQuery),
  );
  if (relatedChildren.length) {
    const newest = Math.max(
      ...relatedChildren.map((thread) => Number(thread.updated_at || 0)),
      0,
    );
    rows.push({
      root: {
        thread_id: RELATED_RESEARCH_ACTIVITY_ID,
        title: relatedTitle,
        updated_at: newest,
        thread_role: "research_root",
        research_activity: {
          state: "related",
          current_title: "Older execution threads with no unambiguous session owner",
        },
      },
      children: [...relatedChildren].sort(
        (left, right) => Number(right.updated_at || 0) - Number(left.updated_at || 0),
      ),
      matchingChildIds: new Set(relatedChildren.map((child) => child.thread_id)),
      rootMatch: relatedRootMatch,
      isResearchSession: true,
      isRelatedGroup: true,
    });
  }
  return rows.sort(
    (left, right) => Number(right.root?.updated_at || 0) - Number(left.root?.updated_at || 0),
  );
}

export function preferredWorkspaceThreadId(threads, rememberedThreadId = "") {
  const rows = Array.isArray(threads) ? threads : [];
  const remembered = String(rememberedThreadId || "").trim();
  if (remembered && rows.some((thread) => thread?.thread_id === remembered)) {
    return remembered;
  }
  const visible = rows.filter(
    (thread) => !HIDDEN_NAVIGATION_ROLES.has(String(thread?.thread_role || "primary")),
  );
  return String(
    visible.find((thread) => !String(thread?.parent_thread_id || "").trim())?.thread_id
      || visible[0]?.thread_id
      || rows[0]?.thread_id
      || "",
  );
}

export function researchSessionDefaultOpen(thread) {
  const state = String(thread?.research_activity?.state || "idle").toLowerCase();
  return ACTIVE_RESEARCH_STATES.has(state);
}
