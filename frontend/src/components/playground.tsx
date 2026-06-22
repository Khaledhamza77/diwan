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
import { WelcomeCard } from "./ui/WelcomeCard/WelcomeCard";
import { PromptSuggestion } from "./ui/WelcomeCard/PromptSuggestion";

const EMPTY_ELEMENTS: any[] = [];

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

export function Playground() {
  const [inputValue, setInputValue] = useState("");
  const { sendMessage } = useChatInteract();
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
      sendMessage(message, []);
      setInputValue("");
      setAutoScroll(true);
    }
  };

  const handlePickPrompt = (text: string) => {
    if (loading || disabled || !text) return;
    const message = {
      name: "User",
      type: "user_message" as const,
      output: text,
    };
    sendMessage(message, []);
    setInputValue("");
    setAutoScroll(true);
  };

  const lastMessageOutput = useMemo(
    () => flatMessages[flatMessages.length - 1]?.output ?? "",
    [flatMessages],
  );

  useEffect(() => {
    if (bottomRef.current && autoScrollRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, lastMessageOutput]);

  const elementsByStepId = useMemo(() => {
    const map = new Map<string, any[]>();
    (elements ?? []).forEach((el: any) => {
      const forId = el.forId;
      if (!map.has(forId)) map.set(forId, []);
      map.get(forId)!.push(el);
    });
    return map;
  }, [elements]);

  const renderMessage = (message: IStep, isLastMessage: boolean) => {
    const author = (message.name ?? "").trim().toLowerCase();
    const text = message.output ?? "";
    const stepElements = elementsByStepId.get(message.id) ?? EMPTY_ELEMENTS;

    if (author === "user") {
      return (
        <UserMessage
          key={message.id}
          text={text}
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
    <AppShell
      header={<TopBar title="المساعد المالي الذكي" />}
    >
        {flatMessages.length === 0 ? (
          <div className="flex-1 flex items-center pb-32 px-6 w-full max-w-5xl mx-auto">
            <WelcomeCard
              title="مرحباً بك في المساعد المالي الذكي"
              subtitle="واجهتك الذكية للاستفسارات والتحليلات المالية. اطرح سؤالاً أو اختر موضوعاً أدناه للبدء."
            >
              <PromptSuggestion
                disabled={disabled || loading}
                category="معايير المحاسبة المصرية"
                text="ما هو معيار المحاسبة المصري رقم ١"
                onPick={handlePickPrompt}
              />
              <PromptSuggestion
                disabled={disabled || loading}
                category="معايير المحاسبة المصرية"
                text="ما هو الغرض من القوائم المالية"
                onPick={handlePickPrompt}
              />
              <PromptSuggestion
                disabled={disabled || loading}
                category="معايير المحاسبة المصرية"
                text="ما هي المجموعة الكاملة من القوائم المالية"
                onPick={handlePickPrompt}
              />
              <PromptSuggestion
                disabled={disabled || loading}
                category="معايير المحاسبة المصرية"
                text="ما هو الحد الادنى لمعلومات المقارنة"
                onPick={handlePickPrompt}
              />
            </WelcomeCard>
          </div>
        ) : (
          <div className="flex-1 overflow-auto pt-6 pb-32 px-6 w-full max-w-5xl mx-auto space-y-6">
            {flatMessages.map((message, index) =>
              renderMessage(message, index === flatMessages.length - 1),
            )}
            <div ref={bottomRef} />
          </div>
        )}

        <InputBar ref={inputBarRef}>
          <Input
            autoFocus
            className="flex-1"
            id="message-input"
            placeholder="اسأل عن أي موضوع مالي..."
            value={inputValue}
            dir="rtl"
            style={{ textAlign: "right" }}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyUp={(e) => {
              if (e.key === "Enter") {
                handleSendMessage();
              }
            }}
          />

          <SendButton
            onClick={handleSendMessage}
            type="submit"
            disabled={!inputValue.trim() || disabled || loading}
          />
        </InputBar>
    </AppShell>
  );
}
