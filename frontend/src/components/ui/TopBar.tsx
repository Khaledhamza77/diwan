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
        bg-gradient-to-b from-[#0c0e18] to-[#0a0f1f]
        text-gray-100
        relative z-20
      `,
        className
      )}
    >
      <div className="text-lg font-semibold">{title}</div>
      {rightTag && (
        <div className="text-xs uppercase tracking-wider text-gray-300">
          {rightTag}
        </div>
      )}
    </div>
  );
};
