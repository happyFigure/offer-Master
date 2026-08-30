import { Cpu, Sparkles } from "lucide-react";
import { CSSProperties, HTMLAttributes, useMemo } from "react";

type AsciiDensity = "compact" | "balanced" | "spacious";

export interface AsciiArtProps extends HTMLAttributes<HTMLDivElement> {
  className?: string;
  style?: CSSProperties;
  rows?: number;
  density?: AsciiDensity;
  intensity?: number;
  label?: string;
  showHeader?: boolean;
  glow?: boolean;
  seedText?: string;
}

const DENSITY_COLUMNS: Record<AsciiDensity, number> = {
  compact: 9,
  balanced: 7,
  spacious: 5,
};

const ASCII_FRAGMENTS = [
  "4f 66 66 65 72 4d 61 73 74 65 72",
  "41 67 65 6e 74 20 4c 6f 6f 70",
  "54 6f 6f 6c 20 52 75 6e 74 69 6d 65",
  "53 6b 69 6c 6c 20 52 65 67 69 73 74 72 79",
  "43 6f 6e 74 65 78 74 20 4d 65 6d 6f 72 79",
  "52 65 73 75 6d 65 20 4a 44 20 4d 61 74 63 68",
];

export function AsciiArt({
  className,
  style,
  rows = 18,
  density = "balanced",
  intensity = 0.36,
  label = "N ASCII",
  showHeader = true,
  glow = true,
  seedText,
  ...props
}: AsciiArtProps) {
  const safeRows = clampNumber(Math.round(rows), 6, 80);
  const safeIntensity = clampNumber(intensity, 0.08, 0.72);
  const asciiRows = useMemo(() => buildAsciiRows(safeRows, density, seedText), [density, safeRows, seedText]);
  const mergedStyle = {
    "--n-ascii-opacity": safeIntensity,
    ...style,
  } as CSSProperties;

  return (
    <div
      className={cn(
        "n-ascii relative isolate overflow-hidden rounded-lg border border-border/35 bg-background/40 text-muted-foreground shadow-sm",
        glow ? "n-ascii-glow" : "",
        className,
      )}
      style={mergedStyle}
      {...props}
    >
      {showHeader ? (
        <div className="n-ascii-header flex items-center justify-between gap-3 border-b border-border/30 px-4 py-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-2 font-medium uppercase tracking-widest">
            <Cpu size={14} aria-hidden="true" />
            {label}
          </span>
          <Sparkles size={14} aria-hidden="true" />
        </div>
      ) : null}
      <pre className="n-ascii-grid pointer-events-none select-none whitespace-pre-wrap break-all font-mono text-[10px] leading-[1.15] tracking-[0.08em] text-muted-foreground/60">
        {asciiRows.join("\n")}
      </pre>
      <div className="n-ascii-scanline absolute inset-x-0 top-0 h-24 bg-primary/10" aria-hidden="true" />
    </div>
  );
}

export function buildAsciiRows(rows: number, density: AsciiDensity = "balanced", seedText = ""): string[] {
  const columns = DENSITY_COLUMNS[density] ?? DENSITY_COLUMNS.balanced;
  const seedFragments = textToHexFragments(seedText);
  const fragments = seedFragments.length ? [...seedFragments, ...ASCII_FRAGMENTS] : ASCII_FRAGMENTS;

  return Array.from({ length: clampNumber(Math.round(rows), 1, 120) }, (_, rowIndex) => {
    const line = Array.from({ length: columns }, (_, columnIndex) => {
      const sourceIndex = (rowIndex + columnIndex * 2) % fragments.length;
      return fragments[sourceIndex];
    }).join("   ");

    return `${rowIndex.toString(16).padStart(4, "0")}  ${line}`;
  });
}

function textToHexFragments(value: string): string[] {
  return value
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 6)
    .map((word) =>
      Array.from(word)
        .map((char) => char.charCodeAt(0).toString(16).padStart(2, "0"))
        .join(" "),
    );
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
