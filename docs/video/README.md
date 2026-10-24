# Recording the 20-minute video

Everything here is generated from [`docs/VIDEO_TRANSCRIPT.md`](../VIDEO_TRANSCRIPT.md). After editing the transcript, rebuild with:
```bash
python src/make_video_assets.py
```

| File | What it is | Used when |
|---|---|---|
| `teleprompter.html` | Auto-scrolling teleprompter (open in its own browser window) | While recording |
| `practice_guide.webm` | Silent 21-minute walkthrough: slide cards + the dashboard demo clicked at the script's pace | Rehearsing (not in git; see below) |
| `subtitles.srt` | English subtitles with estimated timings | After recording |
| `transcript_plain.txt` | Spoken text only, no timings | After recording (YouTube auto-sync) |
| `segments.json` | Parsed script (parts, screen cues, lines) | Used by the tools above |

## 1. Rehearse
1. Start the dashboard: `powershell -ExecutionPolicy Bypass -File start_dashboard.ps1`.
2. Play `practice_guide.webm` (Chrome or VLC) and read along from the teleprompter.
   The red circle shows where to point; the label in the corner says what to do on screen.

## 2. Record (the teleprompter stays out of the video)
The trick: **record only one window or one screen area**, and put the teleprompter somewhere else.

- **OBS Studio (recommended, free):** add a *Window Capture* source for the browser window with the dashboard / slides,
  and an *Audio Input Capture* for your microphone. The teleprompter window is not captured, wherever you put it.
- **Windows Snipping Tool** (`Win + Shift + S`, then the video icon): drag the recording rectangle over the dashboard only.
- **Xbox Game Bar** (`Win + Alt + R`) records only the active window: keep the dashboard window active, and control the
  teleprompter with its keyboard shortcuts only after clicking back into it (or use a second monitor).

Place the teleprompter **just under your webcam** (or on a second monitor near the camera) so your eyes stay close to the lens.

**Teleprompter controls**

| Key | Action |
|---|---|
| Space | Play / pause |
| ↑ / ↓ | Faster / slower (words per minute) |
| + / − | Bigger / smaller text |
| ← / → | Previous / next part |
| Click a line | Start from that line |
| C | Hide the yellow screen cues |
| M | Mirror (for a glass teleprompter) |
| F | Full screen |
| R | Back to the start |

The top-right clock shows elapsed time, the current part and whether you are ahead of or behind the plan.
It keeps time even if the window is partly covered.

## 3. Subtitles
Subtitles are added **after** recording, so they never appear while you are speaking.

- **YouTube (best):** upload the video, then *Subtitles → Add language → English → Upload file → "Without timing"*
  and choose `transcript_plain.txt`. YouTube matches the text to your voice, so the timings are exact.
- **Any editor (Clipchamp, CapCut, Premiere, VLC):** import `subtitles.srt`. Its timings assume 140 words per minute plus
  2 seconds per screen action, so shift or stretch them to match your actual recording.

## Option B: silent final video + voice-over
`final_video.webm` is the finished picture (1920×1080, about 21 minutes): real slides, the architecture pan,
the three reports and the live dashboard demo, timed to the script at 140 words per minute. The GitHub hold at the
end is left out. Record only your voice and lay it under the video:

1. Open `final_video.webm` in VLC with `final_video.srt` as its subtitles (*Subtitle → Add subtitle file*).
   The subtitles show each sentence exactly when it should be spoken, so they work as a voice-over prompter.
2. Record your voice while the video plays (Audacity, Windows Voice Recorder, or the editor's voice-over button),
   with headphones on so the microphone does not pick anything up.
3. In the editor (Clipchamp, CapCut, DaVinci Resolve), put the video and your audio on the timeline and export.
   Nudge any sentence that drifts; the video has short pauses after every screen action.

Re-record it (needs the dashboard running; about 21 minutes plus a few minutes of encoding):
```bash
python src/make_video_assets.py
node docs/video/record_final_video.cjs
python src/make_video_assets.py --final-srt
```

## Re-record the practice guide
Needs the dashboard running, Node 18+ and Playwright 1.62 (`npm i -g playwright@1.62.1`, then `npx playwright install chromium`):
```bash
node docs/video/record_practice_guide.cjs            # about 21 minutes, writes docs/video/practice_guide.webm
```
The video is not committed (it is large); share it through Google Drive if teammates need it.
