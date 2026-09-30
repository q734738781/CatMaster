import { useComposerRuntime } from "@assistant-ui/react";
import { ArrowUpRight, BookOpen, FlaskConical, Network, Orbit } from "lucide-react";

const STARTERS = [
  { icon: BookOpen, title: "Explore the literature", detail: "Find sources. Connect the evidence.", prompt: "Help me review the literature on [topic]. Start with the key questions and primary sources, then summarize what is known and what remains uncertain." },
  { icon: FlaskConical, title: "Run a calculation", detail: "From a structure to a useful result.", prompt: "I would like to calculate [property] for [system]. Help me choose an appropriate method and use the files I attach as the starting point." },
  { icon: Network, title: "Plan a study", detail: "Turn a question into a research path.", prompt: "Help me investigate [research question]. Develop competing hypotheses and propose experiments that can distinguish them." },
];

export default function ConversationWelcome() {
  const composer = useComposerRuntime();
  function useStarter(prompt) {
    // Prefill only; the user edits and sends through the native composer.
    composer.setText(prompt);
    document.querySelector(".v2-composer textarea")?.focus();
  }
  return (
    <section className="v2-welcome" aria-label="Start a conversation">
      <div className="v2-welcome-content">
        <span className="v2-welcome-mark"><Orbit size={40} strokeWidth={1.1} /></span>
        <div className="v2-eyebrow">A SPACE FOR SCIENTIFIC WORK</div>
        <h2>Where should we begin?</h2>
        <p>Work through a question, run a calculation,<br className="v2-welcome-break" /> or follow the evidence somewhere new.</p>
        <div className="v2-starter-grid">
          {STARTERS.map(({ icon: Icon, title, detail, prompt }) => (
            <button key={title} type="button" onClick={() => useStarter(prompt)}>
              <span className="v2-starter-icon"><Icon size={20} strokeWidth={1.5} /><ArrowUpRight size={15} /></span>
              <strong>{title}</strong><span>{detail}</span>
            </button>
          ))}
        </div>
        <small className="v2-welcome-hint">Choose a starting point, or write your own brief below.</small>
      </div>
    </section>
  );
}
