import * as React from "react";
import { cn } from "@/lib/utils";

export type SendButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  ariaLabel?: string;
};

export const StopButton: React.FC<SendButtonProps> = ({
  className,
  ariaLabel = "Stop",
  ...props
}) => {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      className={cn(
        `
        inline-flex items-center justify-center
        h-9 w-9
        rounded-xl
        bg-red-500/20 text-red-400
        border border-red-500/30
        ring-1 ring-red-500/20
        backdrop-blur-md
        shadow-[0_4px_12px_rgba(0,0,0,0.3)]
        hover:bg-red-500/30
        active:scale-95
        transition-all duration-200
      `,
        className
      )}
      {...props}
    >
      {/* Stop Icon (Square) */}
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="currentColor"
        aria-hidden="true"
      >
        <rect x="4" y="4" width="16" height="16" rx="2" />
      </svg>
    </button>
  );
};

export const SendButton: React.FC<SendButtonProps> = ({
  className,
  ariaLabel = "Send",
  ...props
}) => {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      className={cn(
        `
        inline-flex items-center justify-center
        h-9 w-9
        rounded-xl
        bg-white/10 text-white
        border border-white/15
        ring-1 ring-white/10
        backdrop-blur-md
        shadow-[0_4px_12px_rgba(0,0,0,0.3)]
        hover:bg-white/20
        active:scale-95
        disabled:opacity-40 disabled:cursor-not-allowed
        transition-all duration-200
      `,
        className
      )}
      {...props}
    >
      {/* Paper Plane Icon */}
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="currentColor"
        className="translate-x-[1px]"
        aria-hidden="true"
      >
        <path d="M3.4 20.4L22 12L3.4 3.6L3 10L17 12L3 14L3.4 20.4Z" />
      </svg>
    </button>
  );
};
