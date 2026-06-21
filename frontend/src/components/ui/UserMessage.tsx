// components/ui/user-message.tsx
import * as React from "react";
import { Message, type MessageProps } from "./Message";
import { AttachedFiles } from "./AttachedFiles";
import { cn } from "@/lib/utils";
import type { AttachedFile } from "@/types/attachedFile";

export interface UserMessageProps
  extends Omit<MessageProps, "align" | "bubbleClassName" | "children"> {
  className?: string;
  bubbleClassName?: string;
  files?: AttachedFile[];
}

export const UserMessage: React.FC<UserMessageProps> = React.memo(({
  text,
  meta,
  className,
  bubbleClassName,
  files,
}) => {
  return (
    <div className={cn("flex flex-col items-end gap-2", className)}>
      {files != null && files.length > 0 && <AttachedFiles files={files} />}
      <Message
        text={text}
        meta={meta}
        align="right"
        bubbleClassName={cn(
          `
          relative inline-flex items-center
          rounded-full px-5 py-3
          text-white/95
          bg-[#2d4496]
        `,
          bubbleClassName
        )}
      />
    </div>
  );
});
