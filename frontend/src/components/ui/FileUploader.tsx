import { useRef } from "react";
import { cn } from "@/lib/utils";

type FileUploaderProps = {
  accept: string[];
  uploading?: boolean;
  onFileSelect: (file: File) => void;
};

export function FileUploader({
  accept,
  uploading = false,
  onFileSelect,
}: FileUploaderProps) {
  const ref = useRef<HTMLInputElement>(null);

  return (
    <>
      <input
        ref={ref}
        type="file"
        className="hidden"
        accept={accept.join(",")}
        onChange={(e) => {
          const selectedFile = e.target.files?.[0];
          if (selectedFile) {
            onFileSelect(selectedFile);
          }
          if (ref.current) ref.current.value = "";
        }}
      />

      <button
        type="button"
        aria-label="Attach file"
        onClick={() => ref.current?.click()}
        disabled={uploading}
        className={cn(`
          inline-flex items-center justify-center
          h-9 w-9 shrink-0
          rounded-xl
          bg-white/10 text-white/70
          border border-white/15
          ring-1 ring-white/10
          backdrop-blur-md
          shadow-[0_4px_12px_rgba(0,0,0,0.3)]
          hover:bg-white/20 hover:text-white
          active:scale-95
          disabled:opacity-50 disabled:cursor-not-allowed
          transition-all duration-200
        `)}
      >
        {uploading ? (
          <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : (
          /* Paperclip icon */
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
          </svg>
        )}
      </button>
    </>
  );
}
