import React from "react";

type Corner = "top-left" | "top-right" | "bottom-left" | "bottom-right";

type Offset = {
  top?: number | string;
  right?: number | string;
  bottom?: number | string;
  left?: number | string;
};

type PoweredByFinaraiProps = {
  logoSrc: string;

  minPercent?: number;
  basePercent?: number;
  maxPercent?: number;

  position?: Corner;

  offset?: Offset;

  gap?: number | string; // default "0.6rem"

  labelColor?: string;

  alt?: string;

  zIndex?: number;
};

const toCssSize = (v?: number | string): string | undefined => {
  if (v === undefined) return undefined;
  return typeof v === "number" ? `${v}px` : v;
};

export const PoweredByFinarai: React.FC<PoweredByFinaraiProps> = ({
  logoSrc,
  minPercent = 8,
  basePercent = 10,
  maxPercent = 16,
  position = "bottom-right",
  offset,
  gap = "0.6rem",
  labelColor = "rgba(255,255,255,0.35)",
  alt = "Finaria logo",
  zIndex = 40,
}) => {
  const posStyle: React.CSSProperties = { position: "fixed" };

  // Default offsets per corner if none provided
  const defaultCornerOffsets: Record<Exclude<Corner, "custom">, Offset> = {
    "top-left": { top: 40, left: 40 },
    "top-right": { top: 40, right: 40 },
    "bottom-left": { bottom: 40, left: 40 },
    "bottom-right": { bottom: 40, right: 40 },
  };

  const resolvedOffsets: Offset = {
    ...defaultCornerOffsets[position],
    ...(offset ?? {}),
  };

  posStyle.top = toCssSize(resolvedOffsets.top);
  posStyle.right = toCssSize(resolvedOffsets.right);
  posStyle.bottom = toCssSize(resolvedOffsets.bottom);
  posStyle.left = toCssSize(resolvedOffsets.left);

  // Layout container styles
  const containerStyle: React.CSSProperties = {
    ...posStyle,
    zIndex,
    display: "inline-flex",
    alignItems: "center",
    gap: toCssSize(gap),
  };

  // Width scales with the **shorter** viewport side using vmin; keep aspect ratio
  const logoStyle: React.CSSProperties = {
    width: `clamp(${minPercent}vmin, ${basePercent}vmin, ${maxPercent}vmin)`,
    height: "auto",
    display: "block",
  };

  const labelStyle: React.CSSProperties = {
    color: labelColor,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    fontSize: "clamp(10px, 1.6vmin, 14px)",
    lineHeight: 1,
    whiteSpace: "nowrap",
    userSelect: "none",
  };

  return (
    <div style={containerStyle} aria-label="Powered by Finarai">
      <span style={labelStyle}>Powered by</span>
      <img src={logoSrc} alt={alt} style={logoStyle} />
    </div>
  );
};
