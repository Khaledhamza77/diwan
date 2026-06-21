// Input.tsx (lean)
import * as React from "react";
import { cn } from "@/lib/utils";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", ...props }, ref) => {
    return (
      <input
        type={type}
        ref={ref}
        {...props}
        className={cn(
          `
          flex-1
          bg-transparent
          text-sm text-gray-200
          placeholder:text-gray-400
          outline-none
          disabled:opacity-50 disabled:cursor-not-allowed
          transition
        `,
          className
        )}
      />
    );
  }
);

Input.displayName = "Input";
export { Input };
