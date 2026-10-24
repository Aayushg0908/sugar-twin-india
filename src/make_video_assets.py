"""
Build the video-recording aids from docs/VIDEO_TRANSCRIPT.md (the single source of truth):

  docs/video/teleprompter.html      auto-scrolling teleprompter for recording (open it in its own window)
  docs/video/subtitles.srt          English subtitles with estimated timings (~140 words per minute)
  docs/video/transcript_plain.txt   spoken text only, for YouTube's "auto-sync" (exact timings from your audio)
  docs/video/segments.json          parsed parts / cues / lines (used by the practice-guide recorder)

Edit the transcript, then re-run:
    python src/make_video_assets.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "VIDEO_TRANSCRIPT.md"
OUT = ROOT / "docs" / "video"
WPM = 140                 # speaking pace used for subtitle timings
CUE_PAUSE = 2.0           # seconds of silence assumed for each on-screen action
MAX_CHARS = 84            # per subtitle (two lines of ~42)


def clean(text: str) -> str:
    text = re.sub(r"\*\(optional\)\*", "", text).replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def spoken(text: str) -> str:
    """Drop inline screen actions like _[Point.]_ so they never reach subtitles."""
    return re.sub(r"\s+", " ", re.sub(r"_\[.+?\]_", "", text)).strip()


def parse():
    md = SRC.read_text(encoding="utf-8")
    body = md.split("\n## Part 1", 1)[1].split("\n**Length check:**", 1)[0]
    parts = []
    for block in ("## Part 1" + body).split("\n## Part ")[0:]:
        block = block.removeprefix("## Part ")
        head, *lines = block.splitlines()
        m = re.match(r"(\d+) · (.+?) \((\d+:\d+) – (\d+:\d+)\)", head.strip())
        if not m:
            continue
        items = []
        for raw in lines:
            line = raw.strip()
            if not line or line == "---":
                continue
            optional = "*(optional)*" in line
            cue = re.match(r"_\[(.+?)\]_", line)
            if cue:
                items.append({"type": "cue", "text": cue.group(1).strip(), "optional": optional})
            else:
                items.append({"type": "say", "text": clean(line), "optional": optional})
        parts.append({"n": int(m.group(1)), "title": m.group(2), "start": m.group(3), "end": m.group(4), "items": items})
    return parts


def split_caption(sentence: str):
    """Split a sentence into subtitle-sized chunks, preferring commas, then word boundaries."""
    if len(sentence) <= MAX_CHARS:
        return [sentence]
    chunks, cur = [], ""
    for piece in re.split(r"(?<=,)\s+", sentence):
        if len(cur) + len(piece) + 1 <= MAX_CHARS:
            cur = f"{cur} {piece}".strip()
            continue
        if cur:
            chunks.append(cur)
        cur = ""
        while len(piece) > MAX_CHARS:
            cut = piece.rfind(" ", 0, MAX_CHARS)
            chunks.append(piece[:cut])
            piece = piece[cut + 1:]
        cur = piece
    if cur:
        chunks.append(cur)
    return chunks


def two_lines(text: str) -> str:
    if len(text) <= 42:
        return text
    mid = len(text) // 2
    cut = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - mid))
    return text[:cut] + "\n" + text[cut + 1:]


def ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(parts):
    t, n, out = 0.0, 0, []
    for p in parts:
        for it in p["items"]:
            if it["type"] == "cue":
                t += CUE_PAUSE
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", spoken(it["text"])):
                for chunk in split_caption(sentence):
                    dur = max(1.2, len(chunk.split()) / WPM * 60)
                    n += 1
                    out.append(f"{n}\n{ts(t)} --> {ts(t + dur - 0.05)}\n{two_lines(chunk)}\n")
                    t += dur
    return "\n".join(out), t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = parse()
    (OUT / "segments.json").write_text(json.dumps(parts, indent=1, ensure_ascii=False), encoding="utf-8")
    srt, total = build_srt(parts)
    (OUT / "subtitles.srt").write_text(srt, encoding="utf-8")
    plain = "\n\n".join(spoken(it["text"]) for p in parts for it in p["items"] if it["type"] == "say")
    (OUT / "transcript_plain.txt").write_text(plain + "\n", encoding="utf-8")
    tpl = (OUT / "teleprompter.template.html").read_text(encoding="utf-8")
    (OUT / "teleprompter.html").write_text(tpl.replace("/*__SEGMENTS__*/[]", json.dumps(parts, ensure_ascii=False)), encoding="utf-8")
    render_slides()
    words = len(plain.split())
    print(f"{len(parts)} parts, {words} spoken words, subtitles end at {ts(total)[:8]} -> {OUT}")


def render_slides():
    """Render the presentation PDF to 1920x1080 PNGs for the final-video recorder (docs/video/slides/, not in git)."""
    try:
        import pymupdf
    except ImportError:
        print("pymupdf not installed: skipping slide images (only needed for record_final_video.cjs)")
        return
    pdf = ROOT / "docs" / "Sugar_Twin_India_presentation.pdf"
    out = OUT / "slides"
    out.mkdir(exist_ok=True)
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            zoom = 1920 / page.rect.width
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(out / f"slide_{i:02d}.png")


def build_final_srt(timeline_path: Path):
    """Subtitles timed to the recorded final video (record_final_video.cjs writes the timeline)."""
    events = json.loads(timeline_path.read_text(encoding="utf-8"))
    out, n = [], 0
    for ev in events:
        chunks = split_caption(ev["text"])
        words = [len(c.split()) for c in chunks]
        t = ev["start"]
        for chunk, w in zip(chunks, words):
            dur = (ev["end"] - ev["start"]) * w / sum(words)
            n += 1
            out.append(f"{n}\n{ts(t)} --> {ts(t + dur - 0.05)}\n{two_lines(chunk)}\n")
            t += dur
    return "\n".join(out)


def voiceover_lines(timeline_path: Path):
    """Every spoken sentence of the final video with its start/end time and part."""
    events = json.loads(timeline_path.read_text(encoding="utf-8"))
    # the recorder speaks the parts in order, sentence by sentence: walk both lists together to find part breaks
    owner = []
    for p in parse():
        for it in p["items"]:
            if it["type"] == "say":
                owner += [p] * len([s for s in re.split(r"(?<=[.!?])\s+", spoken(it["text"])) if s])
    return [{"start": round(ev["start"], 2), "end": round(ev["end"], 2), "text": spoken(ev["text"]), "part": p["n"], "title": p["title"]}
            for ev, p in zip(events, owner)]


def build_voiceover_script(lines) -> str:
    """Readable voice-over script for the final video: every sentence with the moment to say it, grouped by part."""
    out = ["Sugar Twin India: voice-over script for final_video.webm",
           "Say each line when the video clock reaches its time. Screen actions are already in the video.", ""]
    part = None
    for l in lines:
        if l["part"] != part:
            part = l["part"]
            out += ["", f"=== Part {l['part']}: {l['title']}  (starts {ts(l['start'])[3:8]}) ===", ""]
        out.append(f"[{ts(l['start'])[3:8]}]  {l['text']}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--final-srt":
        (OUT / "final_video.srt").write_text(build_final_srt(OUT / "final_timeline.json"), encoding="utf-8")
        lines = voiceover_lines(OUT / "final_timeline.json")
        (OUT / "final_voiceover_script.txt").write_text(build_voiceover_script(lines), encoding="utf-8")
        tpl = (OUT / "final_teleprompter.template.html").read_text(encoding="utf-8")
        (OUT / "final_teleprompter.html").write_text(tpl.replace("/*__LINES__*/[]", json.dumps(lines, ensure_ascii=False)), encoding="utf-8")
        print(f"wrote final_video.srt, final_voiceover_script.txt and final_teleprompter.html in {OUT}")
    else:
        main()
