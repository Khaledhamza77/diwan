import * as React from "react";
import { AttachedFile } from "@/types/attachedFile";

interface AttachedFilesProps {
  files: AttachedFile[];
  onRemove?: (id: string) => void;
}

export const AttachedFiles: React.FC<AttachedFilesProps> = ({ files, onRemove }) => {
  if (files.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2 px-2">
      {files.map((f) => (
        <div
          key={f.id}
          className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 text-xs text-emerald-300"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span className="truncate max-w-[150px]">{f.name}</span>
          {onRemove != null && (
            <button
              type="button"
              aria-label={`Remove ${f.name}`}
              onClick={() => onRemove(f.id)}
              className="text-gray-500 hover:text-red-400 transition-colors"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}
        </div>
      ))}
    </div>
  );
};
