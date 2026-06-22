// components/ui/top-bar.tsx
import * as React from "react";
import { cn } from "@/lib/utils";

export interface TopBarProps {
  title: string;
  rightTag?: string;
  className?: string;
}

export const TopBar: React.FC<TopBarProps> = ({
  title,
  rightTag,
  className,
}) => {
  return (
    <div
      className={cn(
        `
        w-full h-16
        flex items-center justify-between
        px-6 md:px-10
        border-b border-white/10
        text-gray-100
        relative z-20
      `,
        className
      )}
    >
      <div className="text-lg font-semibold w-full text-right">{title}</div>
      {rightTag && (
        <div className="text-xs uppercase tracking-wider text-gray-300">
          {rightTag}
        </div>
      )}
    </div>
  );
};
