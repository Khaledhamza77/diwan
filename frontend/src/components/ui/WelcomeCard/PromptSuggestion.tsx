type PromptSuggestionProps = {
  text: string; // text to display
  category?: string; // optional small label
  disabled?: boolean;
  onPick: (text: string) => void; // callback receives the text
};

export function PromptSuggestion({
  text,
  category,
  disabled = false,
  onPick,
}: PromptSuggestionProps) {
  return (
    <button
      disabled={disabled}
      type="button"
      onClick={() => onPick(text)}
      aria-label={text}
      className="
        w-full text-right rounded-xl border border-white/10
        bg-white/5 backdrop-blur-sm
        px-4 py-3
        text-gray-200
        shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_6px_18px_rgba(0,0,0,0.25)]
        transition hover:bg-white/7 hover:-translate-y-0.5 focus:outline-none
        focus-visible:ring-2 focus-visible:ring-sky-400/40
      "
    >
      {category && (
        <span className="block text-[11px] tracking-[.08em] uppercase font-semibold text-sky-400" dir="rtl">
          {category}
        </span>
      )}
      <span className="block text-[15px] font-semibold text-slate-100" dir="rtl">
        {text}
      </span>
    </button>
  );
}
``;
