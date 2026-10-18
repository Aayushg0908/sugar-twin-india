import type { Genetics } from "../api";
import { Pill } from "../ui";

const COPIES = [
  { label: "No risk copies", style: "bg-green-50 text-green-700 ring-green-200" },
  { label: "1 risk copy", style: "bg-amber-50 text-amber-700 ring-amber-200" },
  { label: "2 risk copies", style: "bg-red-50 text-red-700 ring-red-200" },
];

/** The four type 2 diabetes risk variants and the overall polygenic risk score. */
export default function GeneticsPanel({ g }: { g: Genetics | null | undefined }) {
  // undefined = the API is an older build that does not send genetics yet; null = no test for this patient
  if (g === undefined) {
    return <p className="text-sm text-amber-700">Genetic data was not sent by the API. Restart the backend (uvicorn) to load it.</p>;
  }
  if (!g) return <p className="text-sm text-slate-500">No genetic test on record.</p>;
  const z = g.prs_z;
  const prsWord = z == null ? null : z >= 1 ? "High" : z <= -1 ? "Low" : "Average";
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
      <ul className="divide-y divide-slate-100 text-sm">
        {g.variants.map((v) => (
          <li key={v.rsid} className="flex items-center justify-between gap-3 py-2">
            <div className="min-w-0">
              <div className="font-semibold text-slate-800">{v.gene} <span className="text-xs font-normal text-slate-400">{v.rsid} · risk allele {v.risk_allele}</span></div>
              <div className="text-xs text-slate-500">{v.role}</div>
            </div>
            <Pill className={COPIES[Math.min(v.copies, 2)].style}>{COPIES[Math.min(v.copies, 2)].label}</Pill>
          </li>
        ))}
      </ul>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <div className="text-[11px] font-medium text-slate-400">Polygenic risk score</div>
        {z == null ? (
          <div className="mt-1 text-sm text-slate-500">Not calculated for this patient</div>
        ) : (
          <>
            <div className={`text-2xl font-extrabold ${prsWord === "High" ? "text-red-600" : prsWord === "Low" ? "text-green-600" : "text-slate-800"}`}>
              {prsWord}
            </div>
            <div className="text-xs text-slate-500">Higher than {g.prs_percentile}% of people (z = {z > 0 ? "+" : ""}{z.toFixed(2)})</div>
          </>
        )}
        <p className="mt-3 text-[11px] leading-snug text-slate-400">
          Inherited risk adds to lifestyle risk; it does not change today's treatment on its own. The twin's models use these markers as inputs.
        </p>
      </div>
    </div>
  );
}
