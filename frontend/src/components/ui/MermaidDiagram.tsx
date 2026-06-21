import * as React from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  themeVariables: {
    background: "#1e1e2e",
    primaryColor: "#a855f7",
    primaryTextColor: "#e2e8f0",
    primaryBorderColor: "#6d28d9",
    lineColor: "#94a3b8",
    secondaryColor: "#2d2d3f",
    tertiaryColor: "#1e1e2e",
  },
  flowchart: { curve: "basis" },
});

let _counter = 0;

interface MermaidDiagramProps {
  code: string;
}

export const MermaidDiagram: React.FC<MermaidDiagramProps> = ({ code }) => {
  const [svg, setSvg] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const id = `mermaid-${++_counter}`;

    setSvg(null);
    setError(null);

    mermaid
      .render(id, code)
      .then(({ svg: rendered }) => {
        if (!cancelled) setSvg(rendered);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message ?? err));
      });

    return () => {
      cancelled = true;
    };
  }, [code]);

  // Must be declared before any conditional returns — Rules of Hooks
  const handleDownload = React.useCallback(() => {
    if (!svg) return;
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "diagram.svg";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [svg]);

  if (error) {
    return (
      <pre className="text-xs text-red-400 bg-[#1e1e2e] p-3 rounded overflow-x-auto">
        {`Diagram render error: ${error}\n\n${code}`}
      </pre>
    );
  }

  if (!svg) {
    return (
      <div className="my-4 h-24 flex items-center justify-center rounded-lg bg-[#1e1e2e] text-gray-500 text-sm">
        Rendering diagram…
      </div>
    );
  }

  return (
    <div className="relative my-4 overflow-x-auto rounded-lg bg-[#1e1e2e] p-4">
      {/* Download button — semi-transparent, top-right corner */}
      <button
        onClick={handleDownload}
        title="Download diagram"
        className="absolute top-2 right-2 z-10 flex items-center gap-1 rounded-md bg-black/40 px-2 py-1 text-[11px] text-gray-300 backdrop-blur-sm transition hover:bg-black/70 hover:text-white"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
        SVG
      </button>
      <div className="flex justify-center" dangerouslySetInnerHTML={{ __html: svg }} />
    </div>
  );
};
