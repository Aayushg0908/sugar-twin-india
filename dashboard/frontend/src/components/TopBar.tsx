import { useState } from "react";
import { Clock, Info, Moon, PanelLeftClose, PanelLeftOpen, Pause, Play, Sun, X } from "lucide-react";
import { useTheme } from "../theme";
import { dateTime } from "../ui";

const STEP = 30 * 60 * 1000;
const TIP_KEY = "sugar-twin-clock-tip-hidden";

function tipHidden() {
  try { return localStorage.getItem(TIP_KEY) === "1"; } catch { return false; }
}

export default function TopBar({ time, start, end, playing, onTime, onPlay, listOpen, onToggleList }: {
  time: number; start: number; end: number; playing: boolean; onTime: (t: number) => void; onPlay: (p: boolean) => void;
  listOpen: boolean; onToggleList: () => void;
}) {
  const min = start + 7 * 3600 * 1000;
  const max = end - 2.5 * 3600 * 1000;
  const day = Math.floor((time - start) / 86400000) + 1;
  const { theme, toggle } = useTheme();
  const [help, setHelp] = useState(false);
  const [tip, setTip] = useState(!tipHidden());
  const hideTip = () => { setTip(false); try { localStorage.setItem(TIP_KEY, "1"); } catch { /* ignore */ } };
  const stepBtn = "rounded-lg px-2 py-1.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100";
  return (
    <>
    <header className="relative flex h-20 shrink-0 items-center gap-4 border-b border-slate-200 bg-panel px-4">
      <button onClick={onToggleList} title={listOpen ? "Hide patient list" : "Show patient list"} aria-label="Toggle patient list"
        className="rounded-lg p-2 text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100">
        {listOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
      </button>
      <div className="flex items-center gap-2.5">
        <img src="/favicon.svg" alt="" className="h-8 w-8" />
        <div className="hidden leading-tight sm:block">
          <div className="text-base font-extrabold tracking-tight text-slate-900">Sugar Twin</div>
          <div className="text-[11px] font-medium text-slate-400">Diabetes digital twin · doctor view</div>
        </div>
      </div>

      {/* simulated clock: moves the whole dashboard through the 14 recorded days */}
      <div className="ml-auto flex items-center gap-3 rounded-xl bg-slate-50 px-3 py-2 ring-1 ring-slate-200">
        <div className="hidden leading-tight md:block">
          <div className="flex items-center gap-1 text-[11px] font-semibold text-blue-700">
            <Clock className="h-3.5 w-3.5" /> Demo time machine
            <button onClick={() => setHelp((h) => !h)} aria-label="What is the demo time machine?" title="What is this?"
              className="ml-0.5 rounded-full text-slate-400 hover:text-slate-700"><Info className="h-3.5 w-3.5" /></button>
          </div>
          <div className="text-sm font-semibold text-slate-800">{dateTime(time)}</div>
        </div>
        <button onClick={() => onTime(Math.max(min, time - 6 * STEP))} title="Move the clock back 3 hours" className={stepBtn}>−3 h</button>
        <button onClick={() => onPlay(!playing)} title={playing ? "Pause the replay" : "Play: the clock moves 30 minutes every second"}
          className="flex items-center gap-1.5 rounded-lg bg-brand-600 px-3 py-2 text-sm font-semibold text-[#fff] hover:bg-brand-500">
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {playing ? "Pause" : "Play replay"}
        </button>
        <button onClick={() => onTime(Math.min(max, time + 6 * STEP))} title="Move the clock forward 3 hours" className={stepBtn}>+3 h</button>
        <div className="hidden sm:block">
          <input type="range" min={min} max={max} step={STEP} value={time} onChange={(e) => onTime(Number(e.target.value))}
            className="block w-40 lg:w-72" aria-label="Simulated time" title="Drag to any moment in the 14 days" />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>Day 1</span><span className="font-semibold text-slate-600">Day {day} of 14</span><span>Day 14</span>
          </div>
        </div>
      </div>
      <button onClick={toggle} title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        className="rounded-lg p-2 text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100">
        {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </button>

      {help && (
        <div className="absolute right-16 top-[4.5rem] z-30 w-[380px] rounded-xl bg-white p-4 text-sm text-slate-600 shadow-lg ring-1 ring-slate-200">
          <div className="mb-1 font-semibold text-slate-800">What is the demo time machine?</div>
          <p>This prototype has <b>14 days of recorded data</b> (CGM, smartwatch and app logs) instead of live patients.
            The clock picks a moment in those days, and the whole dashboard shows only what the twin knew and predicted <b>at that moment</b>.</p>
          <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs">
            <li><b>Play replay</b>: time moves 30 minutes every second, so you can watch alerts appear before spikes and lows happen.</li>
            <li><b>−3 h / +3 h</b> or the slider: jump to any moment.</li>
            <li>In a real clinic this is simply the current time, with live data.</li>
          </ul>
          <button onClick={() => setHelp(false)} className="mt-3 text-xs font-semibold text-brand-700 hover:underline">Got it</button>
        </div>
      )}
    </header>

    {/* always-visible explanation for first-time viewers (doctors, judges); dismissible */}
    {tip && (
      <div className="flex shrink-0 items-center gap-3 border-b border-blue-200 bg-blue-50 px-4 py-2 text-[13px] text-blue-700">
        <Clock className="h-4 w-4 shrink-0" />
        <p className="flex-1">
          <b>Demo time machine:</b> this prototype replays <b>14 days of recorded patient data</b> (CGM, smartwatch, app logs).
          Pick any moment with the slider, or press <b>Play replay</b>, to watch the twin warn about spikes and lows
          <b> up to 2 hours before they happen</b>. In a real clinic this would simply be the live clock.
        </p>
        <button onClick={hideTip} className="flex shrink-0 items-center gap-1 rounded-md border border-blue-200 px-2 py-1 text-xs font-semibold hover:bg-blue-100">
          Got it <X className="h-3.5 w-3.5" />
        </button>
      </div>
    )}
    </>
  );
}
