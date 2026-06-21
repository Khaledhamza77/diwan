import { Input } from "@/components/ui/input";
import { SendButton } from "@/components/ui/button";

import {
  useChatInteract,
  useChatMessages,
  IStep,
  useChatData,
} from "@chainlit/react-client";
import { useMemo, useState, useRef, useEffect } from "react";
import { AppShell } from "./ui/Shell";
import { InputBar } from "./ui/InputBar";
import { UserMessage } from "./ui/UserMessage";
import { AIMessage } from "./ui/AIMessage";
import { TopBar } from "./ui/TopBar";
import { PoweredByFinarai } from "./ui/PoweredByFinaira";
import { WelcomeCard } from "./ui/WelcomeCard/WelcomeCard";
import { PromptSuggestion } from "./ui/WelcomeCard/PromptSuggestion";
import { FileUploader } from "./ui/FileUploader";
import { AttachedFiles } from "./ui/AttachedFiles";
import type { AttachedFile } from "@/types/attachedFile";
const MAX_MB = 20;
const ACCEPT = [".pdf"];
const EMPTY_ELEMENTS: any[] = [];

const validate = (f: File) => {
  const ext = f.name.split(".").pop()?.toLowerCase();
  if (!ext || !ACCEPT.includes("." + ext)) {
    return `Only ${ACCEPT.join(", ")} files are allowed.`;
  }
  if (f.size > MAX_MB * 1024 * 1024) {
    return `File too big. Max ${MAX_MB} MB allowed.`;
  }
  return null;
};

function flattenMessages(
  messages: IStep[],
  condition: (node: IStep) => boolean,
): IStep[] {
  return messages.reduce((acc: IStep[], node) => {
    if (condition(node)) {
      acc.push(node);
    }

    if (node.steps?.length) {
      acc.push(...flattenMessages(node.steps, condition));
    }

    return acc;
  }, []);
}

const _ARABIC_RE = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/;

export function Playground() {
  const [inputValue, setInputValue] = useState("");
  const isArabic = _ARABIC_RE.test(inputValue);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const fileSnapshotsRef = useRef<AttachedFile[][]>([]);
  const [uploading, setUploading] = useState(false);
  const { sendMessage, uploadFile } = useChatInteract();
  const { messages } = useChatMessages();
  const { loading, disabled, elements } = useChatData();
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputBarRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const autoScrollRef = useRef(autoScroll);
  autoScrollRef.current = autoScroll;
  const flatMessages = useMemo(() => {
    return flattenMessages(messages, (m) => m.type.includes("message"));
  }, [messages]);

  useEffect(() => {
    const THRESHOLD = 10;
    const checkBottom = () => {
      const { scrollHeight, scrollTop, clientHeight } =
        document.documentElement;
      return scrollHeight - scrollTop - clientHeight <= THRESHOLD;
    };

    const onUserScroll = () => {
      setAutoScroll(checkBottom());
    };

    window.addEventListener("wheel", onUserScroll);
    window.addEventListener("touchmove", onUserScroll);
    return () => {
      window.removeEventListener("wheel", onUserScroll);
      window.removeEventListener("touchmove", onUserScroll);
    };
  }, []);

  const handleSendMessage = () => {
    if (loading || disabled) return;
    const content = inputValue.trim();
    if (content) {
      const message = {
        name: "User",
        type: "user_message" as const,
        output: content,
      };
      fileSnapshotsRef.current.push([...attachedFiles]);
      sendMessage(
        message,
        attachedFiles.map((f) => ({ id: f.id })),
      );
      setAttachedFiles([]);
      setInputValue("");
      setAutoScroll(true);
    }
  };
  const handleFileSelect = async (selectedFile: File) => {
    const err = validate(selectedFile);
    if (err) {
      alert(err);
      return;
    }

    setUploading(true);
    try {
      const { promise } = uploadFile(selectedFile, (p) =>
        console.log("progress:", p),
      );
      const { id } = await promise;
      setAttachedFiles((prev) => [...prev, { id, name: selectedFile.name }]);
    } catch {
      alert("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const removeAttachedFile = (id: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== id));
  };
  // Get the last message's output to track streaming updates
  const lastMessageOutput = useMemo(
    () => flatMessages[flatMessages.length - 1]?.output ?? "",
    [flatMessages],
  );


  useEffect(() => {
    if (bottomRef.current && autoScrollRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, lastMessageOutput]);

  // ✅ Build a map of elements by stepId once (no per-item hook calls)
  const elementsByStepId = useMemo(() => {
    const map = new Map<string, any[]>();
    (elements ?? []).forEach((el: any) => {
      const forId = el.forId;
      if (!map.has(forId)) map.set(forId, []);
      map.get(forId)!.push(el);
    });
    return map;
  }, [elements]);

  let userMessageIndex = 0;

  const renderMessage = (message: IStep, isLastMessage: boolean) => {
    const author = (message.name ?? "").trim().toLowerCase();
    let text = message.output ?? "";

    const stepElements = elementsByStepId.get(message.id) ?? EMPTY_ELEMENTS;

    if (author === "user") {
      const files = fileSnapshotsRef.current[userMessageIndex];
      userMessageIndex++;
      return (
        <UserMessage
          key={message.id}
          text={text}
          files={files}
          className="mt-4"
        />
      );
    }

    return (
      <AIMessage
        key={message.id}
        text={text}
        elements={stepElements}
        isError={message.isError || false}
        isLoading={isLastMessage && loading}
        className="mt-4"
      />
    );
  };

  return (
    <>
      <TopBar title="Regulatory AI Assistant" />
      <AppShell
        className={
          flatMessages.length === 0 ? "max-w-[75%] pb-32" : "max-w-[75%]"
        }
        outerClassName={flatMessages.length === 0 ? "flex items-center" : ""}
      >
        <div className="flex-1 overflow-auto px-6 pt-6 pb-32">
          <div
            className={
              flatMessages.length === 0
                ? "flex items-center h-full"
                : "space-y-4"
            }
          >
            {flatMessages.length == 0 && (
              <WelcomeCard
                title="Welcome to the Regulatory AI Assistant"
                subtitle="Your intelligent interface for accounting regulatory documents. Ask a question or choose a topic below to begin."
              >
                <PromptSuggestion
                  disabled={disabled || loading}
                  category="معايير المحاسبة المصرية"
                  text="ما هو معيار المحاسبة المصري رقم ١"
                  onPick={(text) => {
                    if (loading || disabled) return;
                    if (text) {
                      const message = {
                        name: "User",
                        type: "user_message" as const,
                        output: text,
                      };
                      fileSnapshotsRef.current.push([...attachedFiles]);
                      sendMessage(
                        message,
                        attachedFiles.map((f) => ({ id: f.id })),
                      );
                      setAttachedFiles([]);
                      setInputValue("");
                      setAutoScroll(true);
                    }
                  }}
                />
                <PromptSuggestion
                  disabled={disabled || loading}
                  category="معايير المحاسبة المصرية"
                  text="ما هو الغرض من القوائم المالية"
                  onPick={(text) => {
                    if (loading || disabled) return;
                    if (text) {
                      const message = {
                        name: "User",
                        type: "user_message" as const,
                        output: text,
                      };
                      fileSnapshotsRef.current.push([...attachedFiles]);
                      sendMessage(
                        message,
                        attachedFiles.map((f) => ({ id: f.id })),
                      );
                      setAttachedFiles([]);
                      setInputValue("");
                      setAutoScroll(true);
                    }
                  }}
                />
                <PromptSuggestion
                  disabled={disabled || loading}
                  category="معايير المحاسبة المصرية"
                  text="ما هي المجموعة الكاملة من القوائم المالية"
                  onPick={(text) => {
                    if (loading || disabled) return;
                    if (text) {
                      const message = {
                        name: "User",
                        type: "user_message" as const,
                        output: text,
                      };
                      fileSnapshotsRef.current.push([...attachedFiles]);
                      sendMessage(
                        message,
                        attachedFiles.map((f) => ({ id: f.id })),
                      );
                      setAttachedFiles([]);
                      setInputValue("");
                      setAutoScroll(true);
                    }
                  }}
                />
                <PromptSuggestion
                  disabled={disabled || loading}
                  category="معايير المحاسبة المصرية"
                  text="ما هو الحد الادنى لمعلومات المقارنة"
                  onPick={(text) => {
                    if (loading || disabled) return;
                    if (text) {
                      const message = {
                        name: "User",
                        type: "user_message" as const,
                        output: text,
                      };
                      fileSnapshotsRef.current.push([...attachedFiles]);
                      sendMessage(
                        message,
                        attachedFiles.map((f) => ({ id: f.id })),
                      );
                      setAttachedFiles([]);
                      setInputValue("");
                      setAutoScroll(true);
                    }
                  }}
                />
              </WelcomeCard>
            )}
            {flatMessages.map((message, index) =>
              renderMessage(message, index === flatMessages.length - 1),
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <InputBar
          ref={inputBarRef}
          topContent={
            attachedFiles.length > 0 ? (
              <AttachedFiles
                files={attachedFiles}
                onRemove={removeAttachedFile}
              />
            ) : undefined
          }
        >
          <Input
            autoFocus
            className="flex-1"
            id="message-input"
            placeholder="Ask about Egyptian Accounting Standards... اسأل عن معايير المحاسبة المصرية"
            value={inputValue}
            dir={isArabic ? "rtl" : "ltr"}
            style={{ textAlign: isArabic ? "right" : "left" }}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyUp={(e) => {
              if (e.key === "Enter") {
                handleSendMessage();
              }
            }}
          />

          <FileUploader
            accept={ACCEPT}
            uploading={uploading}
            onFileSelect={handleFileSelect}
          />

          <SendButton
            onClick={handleSendMessage}
            type="submit"
            disabled={!inputValue.trim() || disabled || loading}
          />
        </InputBar>
        <PoweredByFinarai
          logoSrc="/powered_by_finaira.png"
          position="bottom-right"
          offset={{ bottom: "2%", right: "13%" }}
        />
      </AppShell>
    </>
  );
}
