export function interruptActions(part) {
  const raw = part?.meta?.payload?.interrupts || part?.payload?.interrupts;
  const rows = Array.isArray(raw) ? raw : [raw];
  const out = [];
  for (const row of rows) {
    const value = row?.value || row;
    const requests = value?.action_requests || value?.actionRequests || [];
    if (Array.isArray(requests) && requests.length) {
      requests.forEach((request) => {
        out.push({
          name: request?.name || "",
          args: request?.args || {},
        });
      });
    }
  }
  if (!out.length) return [{ name: "", args: {} }];
  return out;
}

export function repeatInterruptDecision(part, decision) {
  return interruptActions(part).map(() => ({ ...decision }));
}
