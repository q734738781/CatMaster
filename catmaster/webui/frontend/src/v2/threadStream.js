// The transcript is durable. After an absence, read its current snapshot and
// cursor instead of playing old Activity events through the visible cards.
export function subscribeThreadStream({ url, eventNames, loadSnapshot, onEvent, onError,
  visibility = document, EventSourceClass = EventSource }) {
  let source = null;
  let request = null;
  let disposed = false;

  function close() {
    request?.abort();
    source?.close();
    source = null;
  }

  async function refresh() {
    close();
    if (disposed || visibility.hidden) return;
    const controller = new AbortController();
    request = controller;
    try {
      const cursor = await loadSnapshot(controller.signal);
      if (controller.signal.aborted || disposed) return;
      const connection = new EventSourceClass(`${url}?last_seq=${cursor}`);
      source = connection;
      let disconnected = false;
      for (const name of eventNames) {
        connection.addEventListener(name, (event) => {
          if (source === connection && !disconnected && !visibility.hidden) onEvent(event);
        });
      }
      connection.onerror = () => {
        // Preserve the browser's native retry delay. On reconnection, replace
        // the old cursor before accepting any buffered events.
        if (connection.readyState !== 1) disconnected = true;
      };
      connection.onopen = () => {
        if (source === connection && disconnected) refresh();
      };
    } catch (error) {
      if (!controller.signal.aborted && !disposed) onError(error);
    }
  }

  visibility.addEventListener("visibilitychange", refresh);
  refresh();
  return () => {
    disposed = true;
    visibility.removeEventListener("visibilitychange", refresh);
    close();
  };
}
