import React, { forwardRef } from "react";
import { cn } from "@/lib/utils";

interface InputBarProps {
  children: React.ReactNode;
  topContent?: React.ReactNode;
  className?: string;
}

export const InputBar = forwardRef<HTMLDivElement, InputBarProps>(
  ({ children, topContent, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          `
          fixed bottom-[40px] left-0 w-full
          px-10 md:px-20
          pointer-events-auto
        `,
          className,
        )}
      >
        <div className="w-full max-w-5xl mx-auto flex flex-col gap-2">
          {topContent}
          <div
            dir="rtl"
            className={cn(`
              w-full
              flex items-center gap-2
              rounded-2xl
              bg-white/5 backdrop-blur-sm
              border border-white/10
              px-4 py-3
              text-gray-200
              shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]
              transition
            `)}
          >
            {children}
          </div>
          <p className="text-center text-[11px] text-gray-500 px-2">
            المحتوى مُولَّد بالذكاء الاصطناعي لأغراض إعلامية فقط. تحقق دائماً من المصادر الرسمية قبل اتخاذ أي قرارات مالية
          </p>
        </div>
      </div>
    );
  },
);

InputBar.displayName = "InputBar";
