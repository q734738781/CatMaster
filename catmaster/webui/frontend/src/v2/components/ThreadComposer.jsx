import { AttachmentPrimitive, ComposerPrimitive, useComposer, useComposerRuntime } from "@assistant-ui/react";
import { Paperclip, Send, Square, X } from "lucide-react";

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

function CatMasterSubmitButton({ thread, isRunning, hasInterrupt, onSubmit }) {
  const text = useComposer((state) => state.text);
  const composer = useComposerRuntime();
  const label = hasInterrupt ? "Respond" : isRunning ? "Steer" : "Send";
  const canSubmit = String(text || "").trim().length > 0 && thread?.thread_id;
  if (!isRunning && !hasInterrupt) {
    return (
      <ComposerPrimitive.Send className="v2-primary-btn">
        <Send size={15} />
        {label}
      </ComposerPrimitive.Send>
    );
  }
  return (
    <button
      type="button"
      className="v2-primary-btn"
      disabled={!canSubmit}
      onClick={async () => {
        if (!canSubmit) return;
        await onSubmit(text);
        await composer.reset();
      }}
    >
      <Send size={15} />
      {label}
    </button>
  );
}

export default function ThreadComposer({ thread, isRunning, hasInterrupt, onSubmit }) {
  return (
    <ComposerPrimitive.Root
      className="v2-composer"
    >
      <div className="v2-composer-main">
        <ComposerPrimitive.Attachments components={{ Attachment: ComposerAttachment }} />
        <ComposerPrimitive.Input
          aria-label={isRunning ? "Steer CatMaster" : "Message CatMaster"}
          placeholder={isRunning ? "Steer the next safe boundary..." : "Ask CatMaster..."}
          submitMode="ctrlEnter"
          disabled={!thread?.thread_id}
        />
      </div>
      <div className="v2-composer-actions">
        <ComposerPrimitive.AddAttachment className="v2-ghost-btn" multiple disabled={!thread?.thread_id || isRunning}>
          <Paperclip size={15} />
          Attach
        </ComposerPrimitive.AddAttachment>
        {isRunning ? (
          <ComposerPrimitive.Cancel className="v2-ghost-btn">
            <Square size={15} />
            Stop
          </ComposerPrimitive.Cancel>
        ) : null}
        <CatMasterSubmitButton thread={thread} isRunning={isRunning} hasInterrupt={hasInterrupt} onSubmit={onSubmit} />
      </div>
    </ComposerPrimitive.Root>
  );
}
