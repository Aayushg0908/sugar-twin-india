/*
 * Records the FINAL silent video (1920x1080) that follows docs/VIDEO_TRANSCRIPT.md, ready for a voice-over:
 *   - real slides (docs/video/slides/*.png, rendered by `python src/make_video_assets.py`)
 *   - docs/architecture.png with a slow left-to-right pan, and the three HTML reports scrolled while they are discussed
 *   - the live dashboard demo with a mouse pointer, no practice labels
 * Timing: each spoken sentence gets its reading time at WPM (default 140), so the video runs about 21 minutes.
 * The GitHub hold at the very end is left out on purpose.
 *
 * Frames are captured with the DevTools screencast (sharper than Playwright's built-in recorder) and encoded afterwards,
 * so encoding speed never affects timing.
 *
 * Needs: the API + dashboard running (start_dashboard.ps1), Node 18+, Playwright 1.62 with Chromium and its ffmpeg.
 * Run:    node docs/video/record_final_video.cjs          (about 21 min recording + encoding)
 *         set QUICK=1 && node docs/video/record_final_video.cjs   (10x faster dry run)
 * Output: docs/video/final_video.webm and docs/video/final_timeline.json
 *         then `python src/make_video_assets.py --final-srt` writes final_video.srt (subtitles timed to this video)
 */
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const DIR = __dirname;
const ROOT = path.resolve(DIR, "..", "..");
const SEGMENTS = JSON.parse(fs.readFileSync(path.join(DIR, "segments.json"), "utf8"));
const URL = process.env.DASHBOARD_URL || "http://localhost:5173";
const WPM = Number(process.env.WPM || 140);
const QUICK = process.env.QUICK === "1";
const FPS = 25;
const VIEW = { width: 1440, height: 810 };          // rendered at 4/3 scale = 1920x1080 frames
const OUT_W = 1920, OUT_H = 1080;
const FRAMES = path.join(DIR, "final_frames");
const sleep = (ms) => new Promise((r) => setTimeout(r, QUICK ? ms / 10 : ms));
const words = (t) => t.split(/\s+/).filter(Boolean).length;
const speakMs = (t) => (words(t) / WPM) * 60000;
const log = (...a) => console.log(new Date().toLocaleTimeString(), ...a);
const fileUrl = (rel) => "file:///" + path.join(ROOT, rel).replace(/\\/g, "/");

// ---------- capture: keep the newest screencast frame, sample it at FPS against the wall clock
const cap = { t0: 0, latest: null, fresh: false, saved: [], timer: null };
function startSampler() {
  fs.rmSync(FRAMES, { recursive: true, force: true });
  fs.mkdirSync(FRAMES, { recursive: true });
  cap.t0 = Date.now();
  cap.timer = setInterval(() => {
    if (!cap.fresh || !cap.latest) return;
    const idx = Math.floor(((Date.now() - cap.t0) / 1000) * FPS);
    const last = cap.saved[cap.saved.length - 1];
    if (last && last.idx === idx) return;               // one frame per output slot
    const file = path.join(FRAMES, `${String(cap.saved.length).padStart(6, "0")}.jpg`);
    fs.writeFileSync(file, cap.latest);
    cap.saved.push({ idx, file });
    cap.fresh = false;
  }, 1000 / FPS / 2);
}
const now = () => (Date.now() - cap.t0) / 1000;

// ---------- timeline of spoken sentences (video seconds), for the voice-over subtitles
const timeline = [];
async function speak(text, hooks = []) {
  // inline screen actions like _[Point.]_ run at that spot; hooks can also fire at the start of a sentence
  const pieces = text.split(/(_\[.+?\]_)/).filter((s) => s.trim());
  for (const piece of pieces) {
    const inline = piece.match(/^_\[(.+?)\]_$/);
    if (inline) { for (const h of hooks) if (h.re.test(inline[1])) await h.run(); continue; }
    for (const sentence of piece.trim().split(/(?<=[.!?])\s+/)) {
      for (const h of hooks) if (!h.inline && h.re.test(sentence)) await h.run();
      const start = now();
      await sleep(speakMs(sentence));
      timeline.push({ start, end: now(), text: sentence });
    }
  }
}

// ---------- stage page for slides and images (crossfade between two layers)
const dataUrl = (abs) => "data:image/png;base64," + fs.readFileSync(abs).toString("base64");
async function ensureStage(page) {
  if (await page.evaluate(() => !!document.getElementById("stage")).catch(() => false)) return;
  await page.setContent(`<!doctype html><html><body id="stage" style="margin:0;background:#fff;overflow:hidden;width:100vw;height:100vh">
    <style>.layer{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0;transition:opacity .5s ease;transform-origin:0 0}
    .layer.on{opacity:1}</style>
    <img class="layer" id="a"><img class="layer" id="b"></body></html>`);
}
async function showImage(page, abs, bg = "#fff") {
  await ensureStage(page);
  await page.evaluate(([src, bg]) => {
    const [a, b] = ["a", "b"].map((id) => document.getElementById(id));
    const [front, back] = a.classList.contains("on") ? [a, b] : [b, a];
    document.body.style.background = bg;
    back.style.transition = "none"; back.style.transform = "none"; back.style.transition = "";
    back.onload = () => { back.classList.add("on"); front.classList.remove("on"); };
    back.src = src;
  }, [dataUrl(abs), bg]);
  await sleep(700);
}
const showSlide = (page, n) => showImage(page, path.join(DIR, "slides", `slide_${String(n).padStart(2, "0")}.png`));

// pan over the architecture picture: zoom in, then move the focus across the columns
async function panTo(page, fx, fy = 0.42, zoom = 1.75) {
  await page.evaluate(([fx, fy, zoom]) => {
    const img = document.querySelector(".layer.on");
    const W = innerWidth, H = innerHeight;
    // image box after object-fit: contain
    const r = Math.min(W / img.naturalWidth, H / img.naturalHeight);
    const iw = img.naturalWidth * r, ih = img.naturalHeight * r, ox = (W - iw) / 2, oy = (H - ih) / 2;
    const px = ox + fx * iw, py = oy + fy * ih;                    // point to centre
    const tx = Math.min(0, Math.max(W - W * zoom, W / 2 - px * zoom));
    const ty = Math.min(0, Math.max(H - H * zoom, H / 2 - py * zoom));
    img.style.transition = "opacity .5s ease, transform 2.6s ease-in-out";
    img.style.transform = zoom === 1 ? "none" : `translate(${tx}px, ${ty}px) scale(${zoom})`;
  }, [fx, fy, zoom]);
}

// open an HTML report and scroll through it while it is discussed
async function showReport(page, rel, ms) {
  await page.goto(fileUrl(rel), { waitUntil: "load" });
  await page.addStyleTag({ content: "html{scrollbar-width:none} ::-webkit-scrollbar{display:none}" });
  await page.evaluate((ms) => {
    const max = document.documentElement.scrollHeight - innerHeight;
    if (max <= 0) return;
    const hold = Math.min(4000, ms * 0.15), t0 = performance.now();
    const id = setInterval(() => {
      const p = Math.min(1, Math.max(0, (performance.now() - t0 - hold) / (ms - 2 * hold)));
      scrollTo(0, max * (0.5 - Math.cos(Math.PI * p) / 2));
      if (p >= 1) clearInterval(id);
    }, 30);
  }, QUICK ? ms / 10 : ms);
}

// ---------- dashboard pointer
async function pointer(page) {
  await page.evaluate(() => {
    if (document.getElementById("pg-cursor")) return;
    const c = document.createElement("div");
    c.id = "pg-cursor";
    c.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2l15 11.2-6.6.9 3.9 7.4-2.8 1.4-3.8-7.5L4 20z"
      fill="#111" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
    c.style.cssText = "position:fixed;left:720px;top:405px;z-index:99999;pointer-events:none;transition:left .9s ease-in-out,top .9s ease-in-out,transform .15s;filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))";
    document.body.append(c);
  });
}
async function pointAt(page, locator) {
  await locator.scrollIntoViewIfNeeded({ timeout: 4000 }).catch(() => {});
  const box = await locator.boundingBox({ timeout: 4000 }).catch(() => null);
  if (!box) { log("  (not found)"); return false; }
  await page.evaluate(([x, y]) => { const c = document.getElementById("pg-cursor"); if (c) { c.style.left = x + "px"; c.style.top = y + "px"; } },
    [box.x + Math.min(box.width / 2, 60), box.y + box.height / 2]);
  await sleep(1000);
  return true;
}
async function clickAt(page, locator) {
  if (!(await pointAt(page, locator))) return;
  await page.evaluate(() => { const c = document.getElementById("pg-cursor"); if (c) { c.style.transform = "scale(.8)"; setTimeout(() => (c.style.transform = ""), 160); } });
  await locator.click({ timeout: 4000 }).catch((e) => log("  click failed:", e.message.split("\n")[0]));
  await sleep(700);
}
async function scrollToText(page, text) {
  const loc = page.getByText(text, { exact: false }).first();
  await loc.evaluate((el) => el.scrollIntoView({ behavior: "smooth", block: "start" })).catch(() => {});
  await sleep(1300);
  await pointAt(page, loc);
}
async function setClock(page, ms) {
  await page.evaluate((v) => {
    const input = document.querySelector('input[type="range"][aria-label="Simulated time"]');
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(input, String(v));
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, ms);
  await sleep(1500);
}

// ---------- the live demo (Part 6): one action per screen cue, in transcript order
function demoActions(page, state) {
  const btn = (name) => page.getByRole("button", { name });
  return [
    async () => { await page.goto(URL); await page.getByText("Clinical summary").first().waitFor({ timeout: 120000 }); await pointer(page);
                  state.home = Number(await page.locator('input[type="range"][aria-label="Simulated time"]').inputValue()); },
    async () => { await pointAt(page, page.getByText("Demo time machine").first()); },
    async () => { await pointAt(page, page.getByText(/urgent/i).first()); },
    async () => { await clickAt(page, page.getByText("P00152", { exact: true }).first()); },
    async () => { await clickAt(page, page.getByText("P01695", { exact: true }).first()); },
    async () => { await clickAt(page, page.getByText(/Sunita/).first()); },
    async () => { await scrollToText(page, "Clinical summary"); },
    async () => { await scrollToText(page, "Glucose: last 12 hours"); },
    async () => { await clickAt(page, btn(/Reveal what really happened/)); },
    async () => { await clickAt(page, btn(/Play replay/)); },
    async () => { await clickAt(page, btn(/Pause/)); await setClock(page, state.home); await pointAt(page, page.getByText("Right now").first()); },
    async () => { await scrollToText(page, "What the twin suggests"); },
    // the twin profile lives inside the what-if dropdown: open it first
    async () => { await clickAt(page, btn(/What if/)); await sleep(1000); await pointAt(page, page.getByText(/per roti/i).first()); },
    async () => { await clickAt(page, btn(/Try it/).nth(1)); },
    async () => { await clickAt(page, btn(/Health record/)); await sleep(1500); await clickAt(page, btn(/Genetic markers/)); },
    async () => { await clickAt(page, page.getByTitle(/Switch to dark theme/)); await sleep(2500); await clickAt(page, page.getByTitle(/Switch to light theme/)); },
  ];
}
function demoHooks(page) {
  return [
    { inline: true, re: /Got it/, run: () => clickAt(page, page.getByRole("button", { name: /Got it/ }).first()) },
    { inline: true, re: /^Point\.$/, run: () => pointAt(page, page.getByText("High sugar", { exact: true }).first()) },
    { inline: true, re: /Move the sliders/, run: () => clickAt(page, page.getByRole("button", { name: /Roti/ }).first()) },
    { re: /^If I mark a skipped dose/, run: () => clickAt(page, page.locator("button", { hasText: /Metformin|Sulfonylurea|DPP-4|SGLT2|insulin/i }).first()) },
  ];
}

// ---------- slide parts: cues name the slide or file to show
const ARCH_FOCUS = [0.17, 0.36, 0.55, 0.85];        // data, fusion, models, serving + dashboard
async function slidePart(page, part) {
  const items = part.items;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    if (it.type === "cue") {
      if (/GitHub link/i.test(it.text)) continue;    // the GitHub hold is left out
      const says = [];
      for (let j = i + 1; j < items.length && items[j].type === "say"; j++) says.push(items[j]);
      const spokenMs = says.reduce((s, x) => s + speakMs(x.text), 0);
      const slide = it.text.match(/slide (\d+)/i);
      const file = it.text.match(/`([^`]+)`/);
      if (slide) await showSlide(page, Number(slide[1]));
      if (file) {
        if (slide) await sleep(Math.min(5000, spokenMs * 0.2));
        if (file[1].endsWith(".png")) {
          await showImage(page, path.join(ROOT, file[1]));
          await sleep(2500);
          for (let k = 0; k < says.length; k++) {
            await panTo(page, ARCH_FOCUS[Math.min(k, ARCH_FOCUS.length - 1)]);
            await speak(says[k].text);
          }
          await panTo(page, 0.5, 0.5, 1);
          await sleep(2600);
          i += says.length;
          continue;
        }
        await showReport(page, file[1], spokenMs);
      }
      await sleep(1200);
    } else {
      await speak(it.text);
    }
  }
}

(async () => {
  const ffmpeg = path.join(os.homedir(), "AppData", "Local", "ms-playwright", "ffmpeg-1011", "ffmpeg-win64.exe");
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: VIEW, deviceScaleFactor: OUT_W / VIEW.width, colorScheme: "light" });
  const page = await context.newPage();

  // warm the API and the dashboard bundle so the demo opens without a long spinner
  await page.goto(URL);
  await page.getByText("Clinical summary").first().waitFor({ timeout: 180000 });
  await ensureStage(page);

  const cdp = await context.newCDPSession(page);
  cdp.on("Page.screencastFrame", ({ data, sessionId }) => {
    cap.latest = Buffer.from(data, "base64");
    cap.fresh = true;
    cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: OUT_W, maxHeight: OUT_H, everyNthFrame: 1 });
  await showSlide(page, 1);
  await new Promise((r) => setTimeout(r, 500));
  startSampler();

  for (const part of SEGMENTS) {
    log(`Part ${part.n}: ${part.title}  (video ${Math.floor(now() / 60)}:${String(Math.floor(now() % 60)).padStart(2, "0")})`);
    if (part.n === 6) {
      const state = {};
      const actions = demoActions(page, state);
      const hooks = demoHooks(page);
      let a = 0;
      for (const it of part.items) {
        if (it.type === "cue") { log("  cue:", it.text.slice(0, 70)); if (actions[a]) await actions[a++](); }
        else await speak(it.text, hooks);
      }
      await sleep(1000);
    } else {
      await slidePart(page, part);
    }
  }
  await sleep(3000);
  clearInterval(cap.timer);
  const total = now();
  await cdp.send("Page.stopScreencast").catch(() => {});
  await browser.close();
  fs.writeFileSync(path.join(DIR, "final_timeline.json"), JSON.stringify(timeline, null, 1));
  log(`Recorded ${(total / 60).toFixed(1)} min, ${cap.saved.length} distinct frames. Encoding...`);

  // encode: every output frame repeats the newest captured frame for its slot
  const out = path.join(DIR, QUICK ? "final_video_quick.webm" : "final_video.webm");
  const enc = spawn(ffmpeg, ["-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "pipe:0",
    "-vf", `scale=${OUT_W}:${OUT_H}`, "-c:v", "libvpx", "-b:v", "4M", "-crf", "6", "-qmin", "2", "-qmax", "36",
    "-deadline", "good", "-cpu-used", "4", "-threads", String(Math.min(8, os.cpus().length)), "-an", "-y", out], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((r) => enc.on("close", r));
  const n = Math.ceil(total * FPS);
  let p = 0, buf = fs.readFileSync(cap.saved[0].file);
  for (let i = 0; i < n; i++) {
    let moved = false;
    while (p + 1 < cap.saved.length && cap.saved[p + 1].idx <= i) { p++; moved = true; }
    if (moved) buf = fs.readFileSync(cap.saved[p].file);
    if (!enc.stdin.write(buf)) await new Promise((r) => enc.stdin.once("drain", r));
    if (i % (FPS * 60) === 0) log(`  encoded ${Math.round(i / FPS / 60)} of ${Math.round(total / 60)} min`);
  }
  enc.stdin.end();
  await done;
  try { fs.rmSync(FRAMES, { recursive: true, force: true, maxRetries: 5 }); } catch { log(`  could not delete ${FRAMES} (locked); delete it by hand`); }
  log(`Saved ${out} (${(fs.statSync(out).size / 1e6).toFixed(1)} MB)`);
})();
