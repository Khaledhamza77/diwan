// components/AppShell.tsx
import React from "react";

type AppShellProps = {
  children: React.ReactNode;
  className?: string;
  outerClassName?: string;
};

export const AppShell: React.FC<AppShellProps> = ({
  children,
  className,
  outerClassName,
}) => {
  return (
    <div
      className={`min-h-screen w-full  text-gray-100 bg-[#0a0f1f] relative overflow-hidden ${outerClassName ?? ""}`}
    >
      {/* Blue glow — bottom-left */}
      <div className="pointer-events-none absolute left-0 bottom-0 w-[70vw] h-[70vh] bg-[radial-gradient(circle_at_bottom_left,#30488F,transparent_40%)]" />

      {/* Purple glow — top-right */}
      <div className="pointer-events-none absolute right-0 top-0 w-[70vw] h-[70vh] bg-[radial-gradient(circle_at_top_right,#4C207B,transparent_40%)]" />

      <div className={`relative z-10 mx-auto px-6 md:px-10 ${className ?? ""}`}>
        {children}
      </div>
    </div>
  );
};
