export function hasArtifactPreviewFields(artifact) {
  return Boolean(
    artifact
    && typeof artifact === "object"
    && (
      artifact.artifact_id
      || artifact.path
      || artifact.title
      || artifact.renderer
      || artifact.download_url
      || artifact.preview_url
    )
  );
}

export function artifactForSelection(selection, artifacts = []) {
  if (selection?.type !== "artifact" || !selection.artifact_id) return null;
  const listedArtifact = (Array.isArray(artifacts) ? artifacts : [])
    .find((item) => item?.artifact_id === selection.artifact_id) || null;
  if (!hasArtifactPreviewFields(selection.artifact)) return listedArtifact;
  return { ...(listedArtifact || {}), ...selection.artifact };
}
