import { useRef, useState } from "react";
import { AttachmentPrimitive, ComposerPrimitive, useComposer, useComposerRuntime } from "@assistant-ui/react";
import { Activity, ChevronDown, Compass, ListPlus, Paperclip, RotateCcw, Send, Shield, Square, X } from "lucide-react";

function ComposerAttachment() {
  return (
    <AttachmentPrimitive.Root className="v2-composer-attachment">
      <AttachmentPrimitive.unstable_Thumb className="v2-composer-attachment-thumb" />
      <AttachmentPrimitive.Name />
      <AttachmentPrimitive.Remove className="v2-icon-btn compact" title="Remove attachment">
        <X size={13} />
      </AttachmentPrimitive.Remove>
    </AttachmentPrimitive.Root>
  );
}

const RUNNING_STRATEGIES = [
  { value: "interrupt", label: "Steer", icon: Compass, title: "Interrupt the active run at a safe checkpoint, then apply this message." },
  { value: "enqueue", label: "Queue", icon: ListPlus, title: "Keep the active run and process this message afterward." },
  { value: "rollback", label: "Replace", icon: RotateCcw, title: "Discard this turn's checkpoint changes, then apply this message." },
  { value: "reject", label: "Reject if busy", icon: Shield, title: "Submit only if the thread is no longer busy." },
];

function CatMasterSubmitButton({ thread, isRunning, hasInterrupt, strategy, onSubmit, buttonRef }) {
  const text = useComposer((state) => state.text);
  const composer = useComposerRuntime();
  const strategyLabel = RUNNING_STRATEGIES.find((item) => item.value === strategy)?.label || "Steer";
  const label = hasInterrupt && !isRunning ? "Send follow-up" : isRunning ? strategyLabel : "Send";
  const canSubmit = String(text || "").trim().length > 0 && thread?.thread_id;
  if (!isRunning && !hasInterrupt) {
    return (
      <ComposerPrimitive.Send className="v2-primary-btn" aria-label={label} title={label}>
        <Send size={15} />
        <span className="v2-send-label">{label}</span>
      </ComposerPrimitive.Send>
    );
  }
  return (
    <button
      type="button"
      ref={buttonRef}
      className="v2-primary-btn"
      aria-label={label}
      title={label}
      disabled={!canSubmit}
      onClick={async () => {
        if (!canSubmit) return;
        await onSubmit(text, [], { strategy });
        await composer.reset();
      }}
    >
      <Send size={15} />
      <span className="v2-send-label">{label}</span>
    </button>
  );
}

export default function ThreadComposer({ thread, isRunning, hasInterrupt, onSubmit, onStop, controls }) {
  const [strategy, setStrategy] = useState("interrupt");
  const [stopAction, setStopAction] = useState("interrupt");
  const submitButtonRef = useRef(null);
  const runningChoice = RUNNING_STRATEGIES.find((item) => item.value === strategy) || RUNNING_STRATEGIES[0];
  const StrategyIcon = runningChoice.icon;
  return (
    <ComposerPrimitive.Root
      className="v2-composer"
    >
      <div className="v2-composer-main">
        <ComposerPrimitive.Attachments components={{ Attachment: ComposerAttachment }} />
        <ComposerPrimitive.Input
          aria-label={isRunning ? `${runningChoice.label} CatMaster` : "Message CatMaster"}
          placeholder={isRunning ? runningChoice.title : "Ask a research question, or describe a task…"}
          submitMode="ctrlEnter"
          onKeyDown={(event) => {
            if ((isRunning || hasInterrupt) && event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submitButtonRef.current?.click();
            }
          }}
          disabled={!thread?.thread_id}
        />
      </div>
      <div className="v2-composer-settings">{controls}</div>
      {isRunning ? <div className="v2-composer-run-bar">
        <span className="v2-composer-run-label"><Activity size={14} />Working{Number(thread?.pending_run_count || 0) > 0 ? <small>{thread.pending_run_count} queued</small> : null}</span>
        <div className="v2-run-control-group">
          <label className="v2-composer-select">
            <select aria-label="How to stop the active run" value={stopAction} onChange={(event) => setStopAction(event.target.value)}>
              <option value="interrupt">Keep progress</option>
              <option value="rollback">Discard this turn</option>
            </select>
            <ChevronDown size={13} aria-hidden="true" />
          </label>
          <button type="button" className="v2-ghost-btn" onClick={() => onStop?.(stopAction)}><Square size={13} />Stop</button>
        </div>
      </div> : null}
      <div className="v2-composer-actions">
        <ComposerPrimitive.AddAttachment className="v2-ghost-btn" aria-label="Attach files" title="Attach files" multiple disabled={!thread?.thread_id || isRunning}>
          <Paperclip size={15} />
          <span className="v2-attach-label">Attach</span>
        </ComposerPrimitive.AddAttachment>
        <div className="v2-composer-send-group">
        {isRunning ? (
          <label className="v2-run-strategy v2-composer-select" title={runningChoice.title}>
            <StrategyIcon size={15} aria-hidden="true" />
            <select
              aria-label="Running thread message strategy"
              value={strategy}
              onChange={(event) => setStrategy(event.target.value)}
            >
              {RUNNING_STRATEGIES.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
            <ChevronDown size={13} aria-hidden="true" />
          </label>
        ) : null}
        {!isRunning ? <span className="v2-composer-shortcut"><kbd>Ctrl</kbd> <kbd>↵</kbd> to send</span> : null}
        <CatMasterSubmitButton
          thread={thread}
          isRunning={isRunning}
          hasInterrupt={hasInterrupt}
          strategy={strategy}
          onSubmit={onSubmit}
          buttonRef={submitButtonRef}
        />
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
}
