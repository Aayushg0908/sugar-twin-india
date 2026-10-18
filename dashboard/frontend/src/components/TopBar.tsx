import { Moon, Pause, Play, SkipBack, SkipForward, Sun } from "lucide-react";
import { useTheme } from "../theme";
import { dateTime } from "../ui";

const STEP = 30 * 60 * 1000;

export default function TopBar({ time, start, end, playing, onTime, onPlay }: {
  time: number; start: number; end: number; playing: boolean; onTime: (t: number) => void; onPlay: (p: boolean) => void;
}) {
  const min = start + 7 * 3600 * 1000;
  const max = end - 2.5 * 3600 * 1000;
  const day = Math.floor((time - start) / 86400000) + 1;
  const { theme, toggle } = useTheme();
  return (
    <header className="flex h-16 shrink-0 items-center gap-6 border-b border-slate-200 bg-white px-5">
      <div className="flex items-center gap-2.5">
        <img src="/favicon.svg" alt="" className="h-8 w-8" />
        <div className="leading-tight">
          <div className="text-base font-extrabold tracking-tight text-slate-900">Sugar Twin</div>
          <div className="text-[11px] font-medium text-slate-400">Diabetes digital twin · doctor view</div>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-3">
        <div className="hidden text-right leading-tight md:block">
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Demo replay · day {day} of 14</div>
          <div className="text-sm font-semibold text-slate-800">{dateTime(time)}</div>
        </div>
        <button onClick={() => onTime(Math.max(min, time - 6 * STEP))} title="Back 3 hours"
          className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"><SkipBack className="h-4 w-4" /></button>
        <button onClick={() => onPlay(!playing)} title={playing ? "Pause" : "Play: advance 30 min per second"}
          className="flex items-center gap-1.5 rounded-lg bg-brand-600 px-3 py-2 text-sm font-semibold text-[#fff] hover:bg-brand-500">
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {playing ? "Pause" : "Replay"}
        </button>
        <button onClick={() => onTime(Math.min(max, time + 6 * STEP))} title="Forward 3 hours"
          className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"><SkipForward className="h-4 w-4" /></button>
        <input type="range" min={min} max={max} step={STEP} value={time} onChange={(e) => onTime(Number(e.target.value))}
          className="w-56 lg:w-80" aria-label="Simulated time" />
        <button onClick={toggle} title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          className="rounded-lg p-2 text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100">
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    </header>
  );
}
