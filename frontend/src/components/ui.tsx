import type { ReactNode } from "react";

export function Badge({ text, tone }: { text: string; tone: "red" | "amber" | "green" | "violet" | "sky" | "slate" }) {
  const tones = {
    red: "bg-red-600 text-white",
    amber: "bg-amber-400 text-slate-950",
    green: "bg-emerald-600 text-white",
    violet: "bg-violet-600 text-white",
    sky: "bg-sky-600 text-white",
    slate: "bg-slate-700 text-slate-100",
  };
  return <span className={`rounded px-2 py-1 text-xs font-bold tracking-wide whitespace-nowrap ${tones[tone]}`}>{text}</span>;
}

export function Panel({ title, right, children, className = "" }: {
  title: string; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-lg border border-slate-800 bg-slate-900/60 ${className}`}>
      <header className="flex items-center gap-2 border-b border-slate-800 px-3 py-2">
        <h2 className="mr-auto text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</h2>
        {right}
      </header>
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Button({ children, onClick, tone = "default", disabled, title }: {
  children: ReactNode; onClick: () => void; tone?: "default" | "primary" | "danger" | "active"; disabled?: boolean;
  title?: string;
}) {
  const tones = {
    default: "bg-slate-800 hover:bg-slate-700 text-slate-100",
    primary: "bg-sky-600 hover:bg-sky-500 text-white",
    danger: "bg-red-700 hover:bg-red-600 text-white",
    active: "bg-amber-400 text-slate-950",
  };
  return (
    <button type="button" title={title} disabled={disabled} onClick={onClick}
      className={`rounded px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 ${tones[tone]}`}>
      {children}
    </button>
  );
}

export function fmt(v: number | null | undefined, digits = 1, unit = ""): string {
  return v === null || v === undefined ? "—" : `${v.toFixed(digits)}${unit}`;
}

export function statusTone(status: string): string {
  if (status === "completed_sim") return "text-emerald-400";
  if (["queued", "accepted"].includes(status)) return "text-sky-300";
  if (["approaching", "grasping", "verifying", "lifting", "transferring", "releasing"].includes(status)) return "text-amber-300";
  return "text-red-400";
}
