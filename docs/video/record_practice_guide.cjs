/*
 * Records a SILENT practice-guide video of the whole 20-minute script (page only, not your desktop).
 * Timing follows docs/video/segments.json (built from docs/VIDEO_TRANSCRIPT.md) at 140 words per minute:
 *   - slide parts show a card: which part, which slide to show, target time
 *   - the live-demo part drives the real dashboard (clicks, scrolls, replay) with a visible pointer
 * Use it to rehearse with the teleprompter; it is not the final video (no voice).
 *
 * Needs: the API + dashboard running (start_dashboard.ps1), Node 18+, and Playwright 1.62:
 *   npm i -g playwright@1.62.1 && npx playwright install chromium
 * Run:    node docs/video/record_practice_guide.cjs            (about 21 minutes, real time)
 *         set QUICK=1 && node docs/video/record_practice_guide.cjs   (10x faster dry run)
 * Output: docs/video/practice_guide.webm
 */
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const DIR = __dirname;
const SEGMENTS = JSON.parse(fs.readFileSync(path.join(DIR, "segments.json"), "utf8"));
const URL = process.env.DASHBOARD_URL || "http://localhost:5173";
const WPM = Number(process.env.WPM || 140);
const QUICK = process.env.QUICK === "1";
const SIZE = { width: 1280, height: 800 };
const sleep = (ms) => new Promise((r) => setTimeout(r, QUICK ? ms / 10 : ms));
const speakMs = (text) => (text.split(/\s+/).length / WPM) * 60000;
const log = (...a) => console.log(new Date().toLocaleTimeString(), ...a);

// ---------- overlays (practice aids drawn on top of the page)
async function overlays(page) {
  await page.evaluate(() => {
    if (document.getElementById("pg-label")) return;
    const label = document.createElement("div");
    label.id = "pg-label";
    label.style.cssText = "position:fixed;left:12px;bottom:12px;z-index:99999;max-width:560px;padding:8px 12px;border-radius:10px;" +
      "background:rgba(17,24,39,.88);color:#fde68a;font:600 13px Inter,Segoe UI,sans-serif;pointer-events:none;box-shadow:0 4px 14px rgba(0,0,0,.3)";
    const cursor = document.createElement("div");
    cursor.id = "pg-cursor";
    cursor.style.cssText = "position:fixed;left:640px;top:400px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;z-index:99999;" +
      "background:rgba(239,68,68,.35);border:2px solid #ef4444;pointer-events:none;transition:left .7s ease,top .7s ease,transform .15s";
    document.body.append(label, cursor);
  });
}
const setLabel = (page, text) => page.evaluate((t) => { const l = document.getElementById("pg-label"); if (l) l.textContent = t; }, text);

async function pointAt(page, locator) {
  await locator.scrollIntoViewIfNeeded({ timeout: 4000 }).catch(() => {});
  const box = await locator.boundingBox({ timeout: 4000 }).catch(() => null);
  if (!box) return false;
  await page.evaluate(([x, y]) => { const c = document.getElementById("pg-cursor"); if (c) { c.style.left = x + "px"; c.style.top = y + "px"; } },
    [box.x + box.width / 2, box.y + box.height / 2]);
  await sleep(800);
  return true;
}
async function clickAt(page, locator) {
  if (!(await pointAt(page, locator))) return log("  (not found, skipped)");
  await page.evaluate(() => { const c = document.getElementById("pg-cursor"); if (c) { c.style.transform = "scale(.6)"; setTimeout(() => (c.style.transform = ""), 180); } });
  await locator.click({ timeout: 4000 }).catch((e) => log("  click failed:", e.message.split("\n")[0]));
  await sleep(600);
}
async function scrollToText(page, text) {
  const loc = page.getByText(text, { exact: false }).first();
  await loc.evaluate((el) => el.scrollIntoView({ behavior: "smooth", block: "start" })).catch(() => {});
  await sleep(1200);
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

// ---------- slide parts: a cue card
async function card(page, part, cue) {
  await page.setContent(`<!doctype html><html><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
    background:#0f172a;color:#e2e8f0;font-family:Inter,Segoe UI,sans-serif">
    <div style="max-width:900px;padding:40px">
      <div style="color:#5eead4;font-weight:700;letter-spacing:.08em;text-transform:uppercase">Practice guide · Part ${part.n} of 9</div>
      <h1 style="font-size:48px;margin:12px 0 6px">${part.title}</h1>
      <div style="color:#94a3b8;font-size:20px">Target ${part.start} – ${part.end}</div>
      <div style="margin-top:40px;font-size:15px;color:#94a3b8">ON SCREEN NOW</div>
      <div id="cue" style="font-size:30px;color:#fde68a;margin-top:6px">${cue}</div>
      <div style="margin-top:40px;font-size:15px;color:#64748b">Read this part from the teleprompter.</div>
    </div></body></html>`);
}

// ---------- the live demo (Part 6): one action per screen cue, in transcript order
function demoActions(page, state) {
  const btn = (name) => page.getByRole("button", { name });
  return [
    async () => { await page.goto(URL); await page.getByText("Clinical summary").first().waitFor({ timeout: 120000 }); await overlays(page);
                  state.home = Number(await page.locator('input[type="range"][aria-label="Simulated time"]').inputValue()); },
    async () => { await pointAt(page, page.getByText("Demo time machine").first()); await sleep(2500); await clickAt(page, btn(/Got it/).first()); },
    async () => { await pointAt(page, page.getByText(/urgent/i).first()); },
    async () => { await clickAt(page, page.getByText("P00152", { exact: true }).first()); },
    async () => { await clickAt(page, page.getByText("P01695", { exact: true }).first()); },
    async () => { await clickAt(page, page.getByText(/Sunita/).first()); },
    async () => { await scrollToText(page, "Clinical summary"); },
    async () => { await scrollToText(page, "Glucose: last 12 hours"); },
    async () => { await clickAt(page, btn(/Reveal what really happened/)); },
    async () => { await clickAt(page, btn(/Play replay/)); await sleep(18000); await clickAt(page, btn(/Pause/)); },
    async () => { await setClock(page, state.home); await pointAt(page, page.getByText("Right now").first()); },
    async () => { await scrollToText(page, "What the twin suggests"); },
    // the twin profile lives inside the what-if dropdown: open it first
    async () => { await clickAt(page, btn(/What if/)); await sleep(1000); await pointAt(page, page.getByText(/per roti/i).first()); },
    async () => { state.inWhatIf = true; await clickAt(page, btn(/Try it/).nth(1)); },
    async () => { await clickAt(page, btn(/Health record/)); await sleep(2500); await clickAt(page, btn(/Genetic markers/)); },
    async () => { await clickAt(page, page.getByTitle(/Switch to dark theme/)); await sleep(2500); await clickAt(page, page.getByTitle(/Switch to light theme/)); },
  ];
}

// while the what-if line is spoken, change the food type and skip a dose at the moments the script mentions them
async function speakWhatIf(page, ms) {
  await sleep(ms * 0.45);
  await clickAt(page, page.getByRole("button", { name: /Roti/ }).first());
  await sleep(ms * 0.25);
  const skip = page.locator("button", { hasText: /Metformin|Sulfonylurea|DPP-4|SGLT2|insulin/i }).first();
  await clickAt(page, skip);
  await sleep(ms * 0.2);
}

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: DIR, size: SIZE }, colorScheme: "light" });
  const page = await context.newPage();
  const t0 = Date.now();
  for (const part of SEGMENTS) {
    log(`Part ${part.n}: ${part.title}`);
    if (part.n === 6) {
      const state = {};
      const actions = demoActions(page, state);
      let a = 0;
      for (const it of part.items) {
        if (it.type === "cue") {
          if (a > 0) await setLabel(page, `Part 6 · ${it.text}`);
          log("  cue:", it.text.slice(0, 70));
          if (actions[a]) await actions[a++]();
          if (process.env.SNAP) await page.screenshot({ path: path.join(DIR, `snap_${String(a).padStart(2, "0")}.png`) });
          if (a === 1) await setLabel(page, `Part 6 · ${it.text}`);
        } else if (state.inWhatIf && /what-if simulator opens/.test(it.text)) {
          await speakWhatIf(page, speakMs(it.text));
        } else {
          await sleep(speakMs(it.text));
        }
      }
    } else {
      let shown = false;
      for (const it of part.items) {
        if (it.type === "cue") {
          if (!shown) { await card(page, part, it.text); shown = true; }
          else await page.evaluate((t) => { document.getElementById("cue").textContent = t; }, it.text).catch(() => {});
          await sleep(2000);
        } else {
          if (!shown) { await card(page, part, "Slides"); shown = true; }
          await sleep(speakMs(it.text));
        }
      }
    }
    log(`  elapsed ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  }
  const video = page.video();
  await context.close();
  await browser.close();
  const out = path.join(DIR, QUICK ? "practice_guide_quick.webm" : "practice_guide.webm");
  fs.renameSync(await video.path(), out);
  log(`Saved ${out} (${(fs.statSync(out).size / 1e6).toFixed(1)} MB)`);
})();
