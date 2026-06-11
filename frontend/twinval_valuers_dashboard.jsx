import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceDot,
} from "recharts";

// Route: app.twinval.com/valuer   (wire-up: valuer.html + src/valuer.jsx + vercel.json)
//
// TwinVal Valuers — Independent Appraisal Workbench
// For LPPEH Registered Valuers. Aligned to Malaysian Valuation Standards (MVS)
// 7th Edition 2025: MVS 2, 4, 6, 7, 8, 9, 19.
//
// DESIGN RULE 1: engine output and Valuer input are visually distinct at all
// times — teal "ENGINE DATA" labels vs gold "VALUER INPUT" labels. The final
// Opinion of Value is always the Valuer's figure, never the engine RTPMV.
// DESIGN RULE 2: this surface never describes itself as providing a valuation.
//
// Engine data below is a simulated snapshot of the BGL-PILOT-01 bungalow pilot
// (consistent with TwinVal_RTPMV_Simulation.xlsx, Rev B calibration).
// Production wiring replaces ENGINE with fetches to the Go backend:
//   GET  /api/v1/valuer/properties
//   GET  /api/v1/valuer/properties/{id}
//   GET  /api/v1/valuer/properties/{id}/rtpmv
//   GET  /api/v1/valuer/properties/{id}/rtpmv/history
//   GET  /api/v1/valuer/properties/{id}/observations
//   POST /api/v1/valuer/properties/{id}/observations
//   POST/GET/PUT /api/v1/valuer/engagements[/{id}]
//   POST /api/v1/valuer/engagements/{id}/report

// ── Colour System (TwinVal navy + gold) ──────────────────────────────────────
const C = {
  bg:        "#0D1B2A",
  surface:   "#13263B",
  surfaceAlt:"#172C44",
  border:    "#23405E",
  borderHi:  "#2F537A",
  gold:      "#C9A84C",
  goldDim:   "#8F7A3C",
  teal:      "#3FC1C9",
  tealDim:   "#2B8A90",
  text:      "#F2EDE3",
  textDim:   "#9DB0C4",
  textMuted: "#647E99",
  success:   "#4CAF7D",
  warning:   "#E8A045",
  danger:    "#E05C5C",
  paper:     "#F7F4ED",
  ink:       "#1A1A18",
};
const SERIF = "'Cormorant Garamond', Georgia, serif";
const SANS = "'Inter', -apple-system, 'Segoe UI', sans-serif";

const fmtRM = (v, dp = 0) =>
  v === "" || v == null || isNaN(v) ? "—" :
  "RM " + Number(v).toLocaleString("en-MY", { minimumFractionDigits: dp, maximumFractionDigits: dp });
const num = (v) => { const n = parseFloat(v); return isNaN(n) ? 0 : n; };
const todayISO = () => new Date().toISOString().slice(0, 10);

// ── Simulated engine snapshot (replace with Go backend API in production) ───
function lcg(seed) { let s = seed; return () => (s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296; }
function buildHistory(base, seed, dipIdx, dipDepth) {
  const rnd = lcg(seed); const out = [];
  for (let i = 29; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    let v = base * (0.994 + 0.012 * rnd());
    if (29 - i === dipIdx) v = base - dipDepth;
    out.push({
      day: d.toLocaleDateString("en-MY", { day: "2-digit", month: "short" }),
      rtpmv: Math.round(v),
      event: 29 - i === dipIdx,
    });
  }
  return out;
}

const ENGINE = {
  properties: [
    {
      id: "BGL-PILOT-01",
      name: "Pilot Bungalow, Sungai Buloh",
      shortName: "Pilot Bungalow",
      typeLong: "Single-storey detached bungalow (landed residential)",
      sensorStatus: "ACTIVE",
      lastSync: "10 Jun 2026, 14:32:10 MYT",
      landValue: 1000000, structureValue: 500000,
      rtpmv: 1392447, hf: 0.7849, trading: "ACTIVE",
      factors: { SHF: 1.0, ESF: 1.0, USS: 0.0, PDP: 0.8736, CI: 0.8985 },
      ci: 0.8985,
      ciBreakdown: [
        { k: "Uptime", v: 0.995, note: "5,652,012 of 5,702,400 expected readings received (6-day window)" },
        { k: "Consistency", v: 1.0, note: "Single sensor per zone/type — scores 1.0 by specification" },
        { k: "Calibration", v: 1.0, note: "17 days since calibration; decay begins after 90 days" },
        { k: "Tamper", v: 0.99997, note: "16 flagged step-changes across 570,178 consecutive pairs" },
        { k: "Human observation delta", v: -0.10, note: "35 Normal (credit capped at +0.10), 1 Watch, 1 Alert in 7-day window" },
      ],
      hash: "a3f8c91d4e2b7706c5d1e8f2a9b04c37d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1",
      prevHash: "7b2e9f4a1c8d3506e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0",
      tokenId: "TV-BGL01-2026-0610-1432",
      title: {
        address: "Lot 4521, Jalan Meranti 7, Sungai Buloh, Selangor",
        titleNo: "Geran 87231", lotNo: "Lot 4521",
        mukim: "Mukim of Sungai Buloh, District of Petaling, Selangor",
        tenure: "Freehold", category: "Building",
        expressConditions: "This land shall be used for a residential building only",
        encumbrances: "Nil",
        lastTransaction: "Transfer — RM 1,180,000 on 14 Aug 2024 (within 2 years: disclose per MVS 6.2.2(n))",
      },
      zones: [
        { zone: "Roof void", sensors: "Temp/humidity + vibration (structural)", last: "10 Jun 2026 14:32:10" },
        { zone: "Living", sensors: "Temp/humidity + occupancy (PIR)", last: "10 Jun 2026 14:32:08" },
        { zone: "Kitchen", sensors: "Temp/humidity", last: "10 Jun 2026 14:32:06" },
        { zone: "Master bedroom", sensors: "Temp/humidity", last: "10 Jun 2026 14:32:09" },
        { zone: "Bedroom 2", sensors: "Portal observations only", last: "Weekly walkthrough" },
        { zone: "Bedroom 3", sensors: "Portal observations only", last: "Weekly walkthrough" },
        { zone: "Utility / DB", sensors: "Whole-house electrical load", last: "10 Jun 2026 14:32:05" },
      ],
      events: [
        { date: "03 Jun 2026 10:00", label: "Vibration event", detail: "Roof-beam vibration 14.98 mm/s² (neighbour renovation). SHF dipped to 0.436; RTPMV −RM 250,685 for the hour. Recovered by 13:00." },
        { date: "03 Jun 2026 11:45", label: "Watch observation", detail: "Field inspector logged Watch on roof zone: sustained vibration, monitor roof beam." },
        { date: "05 Jun 2026 09:30", label: "Alert observation", detail: "Water staining on kitchen ceiling below bathroom. Alert remains open — costs ≈ RM 44,000 of CI-weighted value until resolved." },
      ],
      observations: [
        { date: "05 Jun 2026", zone: "kitchen", severity: "Alert", observer: "A. Rahman (intern)", note: "Water staining on kitchen ceiling below bathroom", status: "ACTIVE" },
        { date: "03 Jun 2026", zone: "roof", severity: "Watch", observer: "A. Rahman (intern)", note: "Sustained vibration during neighbour renovation; monitor roof beam", status: "ACTIVE" },
        { date: "02 Jun 2026", zone: "living", severity: "Normal", observer: "A. Rahman (intern)", note: "Routine walkthrough — no defects noted", status: "ACTIVE" },
        { date: "02 Jun 2026", zone: "bed2", severity: "Normal", observer: "A. Rahman (intern)", note: "Duplicate entry — voided by admin", status: "VOIDED" },
        { date: "01 Jun 2026", zone: "roof", severity: "Normal", observer: "A. Rahman (intern)", note: "Baseline walkthrough — roof void dry, trusses sound", status: "ACTIVE" },
      ],
      history: buildHistory(1392447, 42, 22, 206332),
      buildingAge: 12,
    },
    {
      id: "VIL-DMS-002",
      name: "Villa Damansara, Petaling Jaya",
      shortName: "Villa Damansara",
      typeLong: "Two-storey detached villa (landed residential)",
      sensorStatus: "PENDING",
      lastSync: "Sensors ordered — human observations only",
      landValue: 950000, structureValue: 500000,
      rtpmv: 1404120, hf: 0.8082, trading: "RESTRICTED",
      factors: { SHF: 1.0, ESF: 0.98, USS: 0.0, PDP: 0.8430, CI: 0.7240 },
      ci: 0.724,
      ciBreakdown: [
        { k: "Uptime", v: 0.0, note: "No sensors installed — uptime not scored (neutralised)" },
        { k: "Consistency", v: 1.0, note: "Not applicable — defaults to 1.0" },
        { k: "Calibration", v: 1.0, note: "Not applicable" },
        { k: "Tamper", v: 1.0, note: "Not applicable" },
        { k: "Human observation delta", v: 0.06, note: "3 Normal observations in window — portal-only data source" },
      ],
      hash: "c5d4e3f2a1b0980716253443526170899a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d",
      prevHash: "e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
      tokenId: "TV-VIL02-2026-0608-0915",
      title: {
        address: "No. 18, Jalan Setia 3, Damansara, Petaling Jaya, Selangor",
        titleNo: "Geran 45102", lotNo: "Lot 2210",
        mukim: "Mukim of Sungai Buloh, District of Petaling, Selangor",
        tenure: "Freehold", category: "Building",
        expressConditions: "Residential building only",
        encumbrances: "Charged to Maybank Berhad (Presentation No. 4451/2023)",
        lastTransaction: "No transaction within 2 years",
      },
      zones: [{ zone: "All zones", sensors: "Portal observations only (sensors pending install)", last: "Weekly walkthrough" }],
      events: [{ date: "08 Jun 2026 09:15", label: "Onboarding", detail: "Property onboarded; pilot sensor kit on order (est. install Jul 2026)." }],
      observations: [
        { date: "08 Jun 2026", zone: "exterior", severity: "Normal", observer: "M. Quraishi", note: "Onboarding walkthrough — façade and roofline sound", status: "ACTIVE" },
      ],
      history: buildHistory(1404120, 7, -1, 0),
      buildingAge: 18,
    },
    {
      id: "SHP-GTN-003",
      name: "Heritage Shophouse, George Town",
      shortName: "Heritage Shophouse",
      typeLong: "Pre-war double-storey shophouse (commercial, heritage zone)",
      sensorStatus: "OFFLINE",
      lastSync: "Last sync 26 May 2026, 03:11 MYT — gateway offline 15 days",
      landValue: 1100000, structureValue: 580000,
      rtpmv: 1283540, hf: 0.3165, trading: "HALTED",
      factors: { SHF: 0.94, ESF: 0.81, USS: 0.04, PDP: 0.5430, CI: 0.4100 },
      ci: 0.41,
      ciBreakdown: [
        { k: "Uptime", v: 0.12, note: "Gateway offline since 26 May — readings stale" },
        { k: "Consistency", v: 1.0, note: "Single sensor per zone" },
        { k: "Calibration", v: 0.62, note: "124 days since calibration (decay threshold 90 days)" },
        { k: "Tamper", v: 0.97, note: "Step-change flags before outage" },
        { k: "Human observation delta", v: -0.15, note: "1 Alert (rear wall damp penetration) — no recent confirmations" },
      ],
      hash: "f1e2d3c4b5a6978869504132231405968a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d",
      prevHash: "98a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7",
      tokenId: "TV-SHP03-2026-0526-0311",
      title: {
        address: "No. 88, Lebuh Armenian, George Town, Pulau Pinang",
        titleNo: "Geran Mukim 1124", lotNo: "Lot 388, Seksyen 17",
        mukim: "Town of George Town, North East District, Pulau Pinang",
        tenure: "Freehold", category: "Building",
        expressConditions: "Subject to George Town UNESCO heritage controls",
        encumbrances: "Nil",
        lastTransaction: "Transfer — RM 1,520,000 on 02 Nov 2025 (within 2 years: disclose per MVS 6.2.2(n))",
      },
      zones: [{ zone: "All zones", sensors: "Gateway OFFLINE — data stale 15 days", last: "26 May 2026 03:11" }],
      events: [{ date: "26 May 2026 03:11", label: "Gateway offline", detail: "Site gateway lost connectivity. CI degrading daily; trading HALTED." }],
      observations: [
        { date: "20 May 2026", zone: "interior", severity: "Alert", observer: "K. Lim", note: "Damp penetration on rear party wall, ground floor", status: "ACTIVE" },
      ],
      history: buildHistory(1283540, 99, -1, 0),
      buildingAge: 96,
    },
  ],
};

// ── MVS reference (UI tooltips) ──────────────────────────────────────────────
const MVS_DEFS = {
  marketValue: "Market Value: estimated amount for which an asset should exchange on the valuation date between a willing buyer and willing seller in an arm's-length transaction after proper marketing — MVS 4.3.1",
  basis: "Basis of Value: fundamental premises on which the reported values are based — MVS Definitions",
  hbu: "Highest and Best Use: the use that maximises the asset's potential — physically possible, legally permissible, financially feasible — MVS E",
  ci: "Confidence Index: TwinVal composite score — uptime + consistency + calibration + tamper detection + human observations — Patent [0052]",
  rtpmv: "RTPMV: Real-Time Property Market Value = Land + Structure × SHF × ESF × (1−USS) × PDP × CI — Patent [0048–0052]",
  effAge: "Effective Age: age computed from sensor-verified maintenance history — may be lower than chronological age — Patent [0070]",
  assumption: "Additional Assumption: an assumption not yet realised at the valuation date — requires the MVS 9.2.3 bold-capitals proviso — MVS Definitions",
};
const PURPOSES = ["Financing", "Sale & Purchase", "Financial Reporting", "Insurance", "Rating", "Capital Market", "Compulsory Acquisition", "Other"];
const BASES = ["Market Value", "Existing Use Value", "Investment Value", "Forced Sale Value", "Fair Value"];
const LIMITING_CONDITIONS = [
  { id: "title", text: "The title to the property is assumed to be good, marketable and free from encumbrances except as noted." },
  { id: "structural", text: "No structural survey has been undertaken; the valuation assumes the structure is sound except where defects are expressly noted." },
  { id: "info", text: "Information provided by the client, solicitors and statutory authorities is assumed to be correct and complete." },
  { id: "site", text: "No site, soil or geotechnical investigation has been carried out; the land is assumed free of contamination." },
  { id: "pm", text: "Plant, machinery, furniture and movable equipment are excluded from this valuation." },
  { id: "purpose", text: "This valuation is valid only for the stated purpose, client and valuation date." },
  { id: "thirdparty", text: "No responsibility is accepted to any third party for the whole or any part of this report." },
  { id: "confidential", text: "This report is confidential to the client and their professional advisers." },
];
const MVS9_PROVISO = "THIS VALUATION IS BASED ON THE ADDITIONAL ASSUMPTION(S) STATED HEREIN. THE VALUE REPORTED MAY NOT BE REALISED SHOULD THE ADDITIONAL ASSUMPTION(S) NOT MATERIALISE.";
const MVS12_DISCLOSURE = "MVS 12.3.2 Disclosure: a Forced Sale Value has been reported for financing purposes. Forced Sale Value assumes a constrained marketing period and does not represent Market Value as defined in MVS 4.3.1. The shortfall between Market Value and Forced Sale Value is disclosed in this report.";

const emptyComp = () => ({ id: "", address: "", date: "", consideration: "", desc: "", area: "", tenure: "", source: "", adjPct: "", adjReason: "" });
const defaultEngagement = (age) => ({
  client: "", purpose: "", valuationDate: todayISO(), intendedUsers: "", basis: "",
  assistant: "", conflict: false,
  inspectionDate: todayISO(), inspector: "valuer",
  zoneNotes: { roof: "", exterior: "", interior: "", services: "" },
  defects: "", accessLimited: false, accessNotes: "", buildingAge: age, ccc: "",
  breach: "no", breachNotes: "", measurementBasis: "",
  landArea: "", builtUpArea: "", comps: [emptyComp(), emptyComp(), emptyComp()],
  landRate: "", structureRate: "", adjustmentNarrative: "",
  income: { gross: "", mgmt: "", maint: "", ins: "", assess: "", capRate: "", yieldSource: "", voidPct: "" },
  cost: { crc: "", phys: "", physBasis: "", func: "", funcBasis: "", econ: "", econBasis: "" },
  ciRating: "", ciNarrative: "", localObs: [],
  opinionOfValue: "", additionalAssumptions: "", asIsValue: "",
  limiting: Object.fromEntries(LIMITING_CONDITIONS.map((l) => [l.id, true])),
});

// ── Small building blocks ────────────────────────────────────────────────────
const Badge = ({ kind }) => {
  const isEngine = kind === "engine";
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, letterSpacing: "0.12em", padding: "2px 7px",
      borderRadius: 3, fontFamily: SANS, whiteSpace: "nowrap",
      color: isEngine ? C.teal : C.gold,
      border: `1px solid ${isEngine ? C.tealDim : C.goldDim}`,
      background: isEngine ? "rgba(63,193,201,0.08)" : "rgba(201,168,76,0.08)",
    }}>{isEngine ? "ENGINE DATA" : "VALUER INPUT"}</span>
  );
};
const Mvs8 = () => (
  <span title="Mandatory report content — MVS 8.2.2" style={{
    fontSize: 8, fontWeight: 700, letterSpacing: "0.1em", padding: "1px 5px",
    borderRadius: 3, color: C.bg, background: C.textDim, fontFamily: SANS, marginLeft: 6,
  }}>MVS 8</span>
);
const Tip = ({ text }) => (
  <span title={text} style={{ color: C.textMuted, cursor: "help", marginLeft: 5, fontSize: 11 }}>ⓘ</span>
);
const inputStyle = {
  width: "100%", boxSizing: "border-box", background: C.bg, color: C.text,
  border: `1px solid ${C.border}`, borderRadius: 4, padding: "8px 10px",
  fontSize: 13, fontFamily: SANS, outline: "none",
};
const Label = ({ children, mvs8, tip }) => (
  <div style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: "0.08em", color: C.textDim, textTransform: "uppercase", marginBottom: 5 }}>
    {children}{mvs8 && <Mvs8 />}{tip && <Tip text={tip} />}
  </div>
);
const Field = ({ label, mvs8, tip, children }) => (
  <div style={{ marginBottom: 14 }}>
    <Label mvs8={mvs8} tip={tip}>{label}</Label>
    {children}
  </div>
);
const Card = ({ title, badge, children, style }) => (
  <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20, ...style }}>
    {(title || badge) && (
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, gap: 10 }}>
        <div style={{ fontFamily: SERIF, fontSize: 19, color: C.text, fontWeight: 600 }}>{title}</div>
        {badge && <Badge kind={badge} />}
      </div>
    )}
    {children}
  </div>
);
const KV = ({ k, v, mono }) => (
  <div style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
    <span style={{ color: C.textDim, fontSize: 12 }}>{k}</span>
    <span style={{ color: C.text, fontSize: 12, textAlign: "right", fontFamily: mono ? "monospace" : SANS, wordBreak: mono ? "break-all" : "normal", maxWidth: "62%" }}>{v}</span>
  </div>
);
const SectionShell = ({ id, num, title, mvsRef, children }) => (
  <section id={id} style={{ marginBottom: 36, scrollMarginTop: 130 }}>
    <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 6 }}>
      <span style={{ fontFamily: SERIF, fontSize: 15, color: C.goldDim }}>S{num}</span>
      <h2 style={{ fontFamily: SERIF, fontSize: 27, color: C.text, margin: 0, fontWeight: 600 }}>{title}</h2>
    </div>
    <div style={{ fontSize: 11, color: C.textMuted, letterSpacing: "0.05em", marginBottom: 16 }}>{mvsRef}</div>
    {children}
  </section>
);
const factorBand = (k, v) => {
  if (k === "USS") return v <= 0.2 ? C.success : v <= 0.45 ? C.warning : C.danger;
  return v >= 0.8 ? C.success : v >= 0.6 ? C.warning : C.danger;
};
const StatusDot = ({ status }) => {
  const col = status === "ACTIVE" ? C.success : status === "PENDING" ? C.warning : C.danger;
  return <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, background: col, marginRight: 6 }} />;
};

// ── Gate page (credentialled demo gate; production: TwinVal auth + role:valuer) ──
// Credential check is a client-side SHA-256 of "accessId:password" — adequate
// for a demonstration deployment, replaced by backend auth in production.
const GATE_HASH = "519477887c0ebd1958901e804c9cf1b6ddbdc852cd750d4f40ae32c18d92a4a6";
async function sha256Hex(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}
function Gate({ onEnter }) {
  const [accessId, setAccessId] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [reg, setReg] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setErr("");
    if (!accessId.trim() || !password) { setErr("Enter your access ID and password."); return; }
    if (reg.trim() && !/^V-\d{3,5}$/i.test(reg.trim())) { setErr("LPPEH registration number must be in the format V-XXXX."); return; }
    setBusy(true);
    const ok = (await sha256Hex(`${accessId.trim().toLowerCase()}:${password}`)) === GATE_HASH;
    setBusy(false);
    if (!ok) { setErr("Invalid access ID or password."); return; }
    onEnter({
      name: name.trim() || "TwinVal Administrator",
      reg: reg.trim() ? reg.trim().toUpperCase() : "V-0001",
      role: "admin",
      accessId: accessId.trim().toLowerCase(),
    });
  };
  return (
    <div style={{ minHeight: "100vh", background: C.bg, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: SANS, padding: 20 }}>
      <div style={{ maxWidth: 460, width: "100%", background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: 36 }}>
        <div style={{ fontFamily: SERIF, fontSize: 30, color: C.gold, marginBottom: 4 }}>TwinVal</div>
        <div style={{ fontSize: 11, letterSpacing: "0.22em", color: C.textDim, marginBottom: 22 }}>VALUERS — INDEPENDENT APPRAISAL WORKBENCH</div>
        <p style={{ color: C.textDim, fontSize: 13, lineHeight: 1.6 }}>
          Access is restricted to authorised users and LPPEH Registered Valuers. The
          engine provides sensor-verified condition data; the professional opinion of
          value remains the Valuer's at all times, in accordance with the Malaysian
          Valuation Standards (7th Edition, 2025).
        </p>
        <Field label="Access ID"><input style={inputStyle} value={accessId} onChange={(e) => setAccessId(e.target.value)} autoComplete="username" /></Field>
        <Field label="Password"><input type="password" style={inputStyle} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" onKeyDown={(e) => e.key === "Enter" && submit()} /></Field>
        <div style={{ borderTop: `1px solid ${C.border}`, margin: "4px 0 14px" }} />
        <Field label="Valuer name (optional — for the report signature block)"><input style={inputStyle} value={name} onChange={(e) => setName(e.target.value)} placeholder="As registered with LPPEH" /></Field>
        <Field label="LPPEH registration number (optional)"><input style={inputStyle} value={reg} onChange={(e) => setReg(e.target.value)} placeholder="V-1234" /></Field>
        {err && <div style={{ color: C.danger, fontSize: 12, marginBottom: 12 }}>{err}</div>}
        <button onClick={submit} disabled={busy} style={{ width: "100%", background: C.gold, color: C.bg, border: "none", borderRadius: 5, padding: "11px 0", fontWeight: 700, fontSize: 13, letterSpacing: "0.08em", cursor: "pointer", opacity: busy ? 0.6 : 1 }}>
          {busy ? "VERIFYING…" : "ENTER WORKBENCH"}
        </button>
        <div style={{ marginTop: 18, fontSize: 11.5, color: C.textMuted, textAlign: "center" }}>
          Not yet registered with TwinVal?{" "}
          <a href="https://twinval.com/#contact" style={{ color: C.teal }}>Contact TwinVal to register as a Valuer</a>
        </div>
        <div style={{ marginTop: 14, fontSize: 10, color: C.textMuted, textAlign: "center" }}>
          Demonstration access gate — production deployments use TwinVal authentication with a role:valuer check.
        </div>
      </div>
    </div>
  );
}

// ── Main dashboard ───────────────────────────────────────────────────────────
export default function ValuersDashboard() {
  const [auth, setAuth] = useState(() => {
    // Sessions created before the credential gate carry no role — force re-login.
    try {
      const a = JSON.parse(localStorage.getItem("tv_valuer_auth"));
      return a && a.role ? a : null;
    } catch { return null; }
  });
  const [propId, setPropId] = useState(ENGINE.properties[0].id);
  const P = ENGINE.properties.find((p) => p.id === propId);
  const [eng, setEng] = useState(() => loadEngagement(ENGINE.properties[0]));
  const [tab, setTab] = useState("market");
  const [range, setRange] = useState(30);
  const [savedAt, setSavedAt] = useState(null);
  const [showReport, setShowReport] = useState(false);
  const [obsForm, setObsForm] = useState({ date: todayISO(), zone: "", severity: "Normal", note: "" });
  const engRef = useRef(eng);
  engRef.current = eng;

  function loadEngagement(prop) {
    try {
      const raw = localStorage.getItem("tv_valuer_engagement_" + prop.id);
      if (raw) return { ...defaultEngagement(prop.buildingAge), ...JSON.parse(raw) };
    } catch { /* fall through */ }
    return defaultEngagement(prop.buildingAge);
  }
  const persist = useCallback((id) => {
    localStorage.setItem("tv_valuer_engagement_" + id, JSON.stringify(engRef.current));
    setSavedAt(new Date().toLocaleTimeString("en-MY"));
  }, []);

  // Autosave every 60s (production: PUT /api/v1/valuer/engagements/{id}) + on unload
  useEffect(() => {
    const t = setInterval(() => persist(propId), 60000);
    const onUnload = () => persist(propId);
    window.addEventListener("beforeunload", onUnload);
    return () => { clearInterval(t); window.removeEventListener("beforeunload", onUnload); };
  }, [propId, persist]);

  const switchProperty = (id) => {
    persist(propId);
    setPropId(id);
    setEng(loadEngagement(ENGINE.properties.find((p) => p.id === id)));
  };
  const set = (patch) => setEng((e) => ({ ...e, ...patch }));
  const setDeep = (key, patch) => setEng((e) => ({ ...e, [key]: { ...e[key], ...patch } }));

  // ── Derived values ──
  const compsComplete = eng.comps.filter((c) => c.id && c.consideration && c.date && c.area);
  const valuerMV = num(eng.landArea) * num(eng.landRate) + num(eng.builtUpArea) * num(eng.structureRate) * P.hf;
  const deltaPct = P.rtpmv ? ((valuerMV - P.rtpmv) / P.rtpmv) * 100 : 0;
  const deltaConsistent = Math.abs(deltaPct) <= 10;
  const income = eng.income;
  const outgoings = num(income.mgmt) + num(income.maint) + num(income.ins) + num(income.assess);
  const netRent = num(income.gross) - outgoings;
  const capValue = num(income.capRate) > 0 ? netRent / (num(income.capRate) / 100) : 0;
  const investmentValue = capValue * (1 - num(income.voidPct) / 100);
  const cost = eng.cost;
  const drc = num(cost.crc) * (1 - num(cost.phys) / 100) * (1 - num(cost.func) / 100) * (1 - num(cost.econ) / 100);
  const marketLandValue = num(eng.landArea) * num(eng.landRate);
  const drcValue = drc + marketLandValue;
  const approachesUsed = [
    compsComplete.length >= 3 && eng.landRate && eng.structureRate ? "Market/Comparison Approach" : null,
    num(income.gross) > 0 && num(income.capRate) > 0 ? "Income Approach" : null,
    num(cost.crc) > 0 ? "Cost Approach (DRC)" : null,
  ].filter(Boolean);
  const s1Complete = eng.client && eng.purpose && eng.basis && eng.valuationDate && eng.conflict;
  const incomeRelevant = eng.purpose === "Financing" || eng.purpose === "Financial Reporting";
  const forcedSale = eng.purpose === "Financing" && eng.basis === "Forced Sale Value";

  // MVS 8 mandatory checklist → blocks report generation
  const outstanding = [];
  if (!eng.client) outstanding.push("Client name (S1)");
  if (!eng.purpose) outstanding.push("Purpose of valuation (S1)");
  if (!eng.basis) outstanding.push("Basis of value (S1)");
  if (!eng.conflict) outstanding.push("Conflict of interest declaration (S1)");
  if (!eng.inspectionDate) outstanding.push("Inspection date (S2)");
  if (!eng.measurementBasis) outstanding.push("Measurements basis (S2)");
  if (compsComplete.length < 3) outstanding.push(`Minimum 3 complete comparables — ${compsComplete.length} of 3 (S3)`);
  if (!eng.landRate || !eng.structureRate) outstanding.push("Reconciled land and structure rates (S3)");
  if (!eng.landArea || !eng.builtUpArea) outstanding.push("Land area and built-up area (S3)");
  if (!eng.ciRating) outstanding.push("Valuer's CI reliability assessment (S4)");
  if (!eng.opinionOfValue) outstanding.push("Valuer's Opinion of Value (S5)");
  const canGenerate = outstanding.length === 0;

  const submitObservation = () => {
    if (!obsForm.zone || !obsForm.note.trim()) return;
    // Production: POST /api/v1/valuer/properties/{id}/observations (write-once)
    set({ localObs: [...eng.localObs, { ...obsForm, observer: `${auth.name} (${auth.reg})`, status: "ACTIVE", pending: true }] });
    setObsForm({ date: todayISO(), zone: "", severity: "Normal", note: "" });
  };
  const voidObservation = (idx) => {
    const reason = window.prompt("Observations are write-once (tamper-evident audit trail). Enter a reason to VOID this observation:");
    if (!reason) return;
    set({ localObs: eng.localObs.map((o, i) => i === idx ? { ...o, status: "VOIDED", voidReason: reason } : o) });
  };
  const exportRecord = () => {
    persist(propId);
    const record = {
      exportedAt: new Date().toISOString(),
      retention: "MVS 7.2.3 Record of Valuation Work",
      valuer: auth,
      propertyId: P.id,
      engagement: eng,
      engineSnapshot: {
        rtpmv: P.rtpmv, healthFactor: P.hf, factors: P.factors, tradingStatus: P.trading,
        sensorStatus: P.sensorStatus, lastSync: P.lastSync,
        hash: P.hash, prevHash: P.prevHash, valuationTokenId: P.tokenId,
        note: "Engine data is sensor-verified condition analysis — supplementary to, not a substitute for, the Valuer's opinion of value.",
      },
    };
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `TwinVal_Valuation_Record_${P.id}_${eng.valuationDate}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (!auth) return <Gate onEnter={(a) => { localStorage.setItem("tv_valuer_auth", JSON.stringify(a)); setAuth(a); }} />;

  const allObs = [...eng.localObs.map((o, i) => ({ ...o, _local: i })), ...P.observations];

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: SANS }}>
      {/* ── Header ── */}
      <header className="no-print" style={{ position: "sticky", top: 0, zIndex: 50, background: "rgba(13,27,42,0.97)", borderBottom: `1px solid ${C.border}`, backdropFilter: "blur(6px)" }}>
        <div style={{ maxWidth: 1240, margin: "0 auto", padding: "14px 24px", display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <div>
            <span style={{ fontFamily: SERIF, fontSize: 24, color: C.gold, fontWeight: 600 }}>TwinVal</span>
            <span style={{ fontSize: 10, letterSpacing: "0.2em", color: C.textDim, marginLeft: 12 }}>VALUERS — INDEPENDENT APPRAISAL WORKBENCH</span>
          </div>
          <span style={{ fontSize: 9.5, color: C.textMuted, border: `1px solid ${C.border}`, borderRadius: 3, padding: "3px 8px", letterSpacing: "0.08em" }}>
            MVS 7TH EDITION 2025 · LPPEH
          </span>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 14, fontSize: 11.5, color: C.textDim }}>
            <span>{savedAt ? `Autosaved ${savedAt} · MVS 7.2.3 record retained` : "Autosave: every 60 s"}</span>
            <span style={{ color: C.text }}>{auth.name} <span style={{ color: C.gold }}>({auth.reg})</span></span>
            <button onClick={() => { localStorage.removeItem("tv_valuer_auth"); setAuth(null); }}
              style={{ background: "none", border: `1px solid ${C.border}`, color: C.textDim, borderRadius: 4, padding: "4px 10px", fontSize: 11, cursor: "pointer" }}>
              Sign out
            </button>
          </div>
        </div>
        {/* Section nav */}
        <div style={{ maxWidth: 1240, margin: "0 auto", padding: "0 24px 10px", display: "flex", gap: 8, flexWrap: "wrap" }}>
          {[["s1", "1 · Engagement"], ["s2", "2 · Inspection"], ["s3", "3 · Valuation"], ["s4", "4 · Confidence & Audit"], ["s5", "5 · Report"]].map(([id, label]) => (
            <a key={id} href={"#" + id} style={{ fontSize: 11.5, color: C.textDim, textDecoration: "none", border: `1px solid ${C.border}`, borderRadius: 4, padding: "5px 12px" }}>{label}</a>
          ))}
        </div>
      </header>

      {/* ── CI banners (Design Rule 8 + MVS 8.2.2(j)) ── */}
      <div className="no-print" style={{ maxWidth: 1240, margin: "0 auto", padding: "16px 24px 0" }}>
        {P.ci < 0.65 && (
          <div style={{ background: "rgba(224,92,92,0.12)", border: `1px solid ${C.danger}`, borderRadius: 6, padding: "10px 16px", fontSize: 12.5, color: C.danger, marginBottom: 10 }}>
            ⚠ Confidence Index {P.ci.toFixed(2)} — below the RESTRICTED threshold (0.65). MVS 7.2.1.1 requires the Valuer to consider the reliability of this data source and disclose any limitations.
          </div>
        )}
        {P.ci >= 0.65 && P.ci < 0.8 && (
          <div style={{ background: "rgba(232,160,69,0.1)", border: `1px solid ${C.warning}`, borderRadius: 6, padding: "10px 16px", fontSize: 12.5, color: C.warning, marginBottom: 10 }}>
            Low Confidence Index ({P.ci.toFixed(2)}) — MVS 8.2.2(j) requires disclosure of data limitations in the report.
          </div>
        )}
      </div>

      <main className="no-print" style={{ maxWidth: 1240, margin: "0 auto", padding: "18px 24px 80px" }}>

        {/* ════ SECTION 1 — Engagement ════ */}
        <SectionShell id="s1" num={1} title="Property Selection & Engagement Setup" mvsRef="MVS 2 (Conditions of Engagement) · MVS 3 (Purpose of Valuation)">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(290px, 1fr))", gap: 12, marginBottom: 18 }}>
            {ENGINE.properties.map((p) => (
              <button key={p.id} onClick={() => switchProperty(p.id)}
                style={{
                  textAlign: "left", cursor: "pointer", borderRadius: 8, padding: 16,
                  background: p.id === propId ? C.surfaceAlt : C.surface,
                  border: `1px solid ${p.id === propId ? C.gold : C.border}`,
                  color: C.text, fontFamily: SANS,
                }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontFamily: SERIF, fontSize: 17 }}>{p.shortName}</span>
                  <span style={{ fontSize: 10.5, color: C.textDim }}><StatusDot status={p.sensorStatus} />{p.sensorStatus}</span>
                </div>
                <div style={{ fontSize: 11, color: C.textMuted }}>{p.id} · {p.typeLong}</div>
                <div style={{ fontSize: 10.5, color: C.textMuted, marginTop: 4 }}>Last sync: {p.lastSync}</div>
              </button>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))", gap: 16 }}>
            <Card title="Engine snapshot" badge="engine">
              <KV k="Sensor onboarding status" v={<span><StatusDot status={P.sensorStatus} />{P.sensorStatus}</span>} />
              <KV k="Last sensor sync" v={P.lastSync} />
              <KV k={<span>Engine RTPMV <Tip text={MVS_DEFS.rtpmv} /></span>} v={<strong style={{ color: C.teal, fontSize: 15 }}>{fmtRM(P.rtpmv)}</strong>} />
              <KV k="Trading status (information only)" v={P.trading} />
              <div style={{ marginTop: 12, fontSize: 11, color: C.textMuted, fontStyle: "italic" }}>
                Engine RTPMV is sensor-verified condition data — not a valuation opinion. The opinion of value is formed by the Valuer in Section 5.
              </div>
            </Card>

            <Card title="Conditions of engagement" badge="valuer">
              <Field label="Client name" mvs8><input style={inputStyle} value={eng.client} onChange={(e) => set({ client: e.target.value })} /></Field>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Field label="Purpose of valuation" mvs8 tip="MVS 3.2.2 purpose categories">
                  <select style={inputStyle} value={eng.purpose} onChange={(e) => set({ purpose: e.target.value })}>
                    <option value="">Select…</option>{PURPOSES.map((p) => <option key={p}>{p}</option>)}
                  </select>
                </Field>
                <Field label="Basis of value" mvs8 tip={MVS_DEFS.basis}>
                  <select style={inputStyle} value={eng.basis} onChange={(e) => set({ basis: e.target.value })}>
                    <option value="">Select…</option>{BASES.map((b) => <option key={b}>{b}</option>)}
                  </select>
                </Field>
                <Field label="Valuation date" mvs8><input type="date" style={inputStyle} value={eng.valuationDate} onChange={(e) => set({ valuationDate: e.target.value })} /></Field>
                <Field label="Intended user(s)"><input style={inputStyle} value={eng.intendedUsers} onChange={(e) => set({ intendedUsers: e.target.value })} placeholder="e.g. Maybank Berhad" /></Field>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Field label="Valuer (locked for session)"><input style={{ ...inputStyle, color: C.textDim }} value={`${auth.name} — ${auth.reg}`} readOnly /></Field>
                <Field label="Designated assistant (optional)"><input style={inputStyle} value={eng.assistant} onChange={(e) => set({ assistant: e.target.value })} /></Field>
              </div>
              <label style={{ display: "flex", gap: 10, alignItems: "flex-start", fontSize: 12.5, color: C.text, cursor: "pointer" }}>
                <input type="checkbox" checked={eng.conflict} onChange={(e) => set({ conflict: e.target.checked })} style={{ marginTop: 2 }} />
                <span>I confirm no conflict of interest exists in respect of this engagement — MVS 1.2.6 <Mvs8 /></span>
              </label>
              <a href="#s2" onClick={() => persist(propId)}
                style={{
                  display: "inline-block", marginTop: 16, padding: "10px 22px", borderRadius: 5, fontSize: 12.5, fontWeight: 700,
                  letterSpacing: "0.06em", textDecoration: "none",
                  background: s1Complete ? C.gold : C.surfaceAlt, color: s1Complete ? C.bg : C.textMuted,
                  border: `1px solid ${s1Complete ? C.gold : C.border}`,
                  pointerEvents: s1Complete ? "auto" : "none",
                }}>
                PROCEED TO INSPECTION →
              </a>
              {!s1Complete && <div style={{ fontSize: 11, color: C.textMuted, marginTop: 8 }}>Complete all mandatory fields and the conflict declaration to proceed.</div>}
            </Card>
          </div>
        </SectionShell>

        {/* ════ SECTION 2 — Inspection ════ */}
        <SectionShell id="s2" num={2} title="Inspection & Investigation" mvsRef="MVS 6 (Inspection and Investigation)">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))", gap: 16 }}>
            <Card title="Title & identification" badge="engine">
              <KV k="Address" v={P.title.address} />
              <KV k="Title / Lot" v={`${P.title.titleNo} · ${P.title.lotNo}`} />
              <KV k="Mukim / District / State" v={P.title.mukim} />
              <KV k="Tenure" v={P.title.tenure} />
              <KV k="Category of land use" v={P.title.category} />
              <KV k="Express conditions" v={P.title.expressConditions} />
              <KV k="Encumbrances" v={P.title.encumbrances} />
              <KV k="Last transaction (2-yr check, MVS 6.2.2(n))" v={P.title.lastTransaction} />
              <div style={{ marginTop: 14 }}>
                <Label>Sensor zones — last readings</Label>
                {P.zones.map((z) => (
                  <KV key={z.zone} k={`${z.zone} — ${z.sensors}`} v={z.last} />
                ))}
              </div>
            </Card>

            <Card title="Valuer's inspection record" badge="valuer">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Field label="Inspection date" mvs8><input type="date" style={inputStyle} value={eng.inspectionDate} onChange={(e) => set({ inspectionDate: e.target.value })} /></Field>
                <Field label="Inspector">
                  <div style={{ display: "flex", gap: 14, paddingTop: 8, fontSize: 12.5 }}>
                    {[["valuer", "Valuer"], ["assistant", "Designated Assistant"]].map(([v, l]) => (
                      <label key={v} style={{ display: "flex", gap: 6, cursor: "pointer" }}>
                        <input type="radio" checked={eng.inspector === v} onChange={() => set({ inspector: v })} />{l}
                      </label>
                    ))}
                  </div>
                </Field>
              </div>
              {[["roof", "Roof"], ["exterior", "Exterior"], ["interior", "Interior"], ["services", "M&E services"]].map(([k, l]) => (
                <Field key={k} label={`Physical condition — ${l}`}>
                  <textarea rows={2} style={{ ...inputStyle, resize: "vertical" }} value={eng.zoneNotes[k]} onChange={(e) => setDeep("zoneNotes", { [k]: e.target.value })} />
                </Field>
              ))}
              <Field label="Visible defects notepad" tip="Non-empty defects are flagged and disclosed per MVS 6.2.2(e)">
                <textarea rows={2} style={{ ...inputStyle, resize: "vertical", border: `1px solid ${eng.defects.trim() ? C.danger : C.border}` }} value={eng.defects} onChange={(e) => set({ defects: e.target.value })} />
                {eng.defects.trim() && <div style={{ color: C.danger, fontSize: 11, marginTop: 4 }}>⚠ Defects recorded — will be disclosed in the report (MVS 6.2.2(e)).</div>}
              </Field>
              <label style={{ display: "flex", gap: 8, fontSize: 12.5, marginBottom: 10, cursor: "pointer" }}>
                <input type="checkbox" checked={eng.accessLimited} onChange={(e) => set({ accessLimited: e.target.checked })} />
                Access limitations encountered (triggers MVS 6.2.2(e) limitation disclosure)
              </label>
              {eng.accessLimited && (
                <Field label="Access limitation details"><textarea rows={2} style={{ ...inputStyle, resize: "vertical" }} value={eng.accessNotes} onChange={(e) => set({ accessNotes: e.target.value })} /></Field>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                <Field label="Building age (yrs)" tip={MVS_DEFS.effAge}><input type="number" style={inputStyle} value={eng.buildingAge} onChange={(e) => set({ buildingAge: e.target.value })} /></Field>
                <Field label="CF / CCC status">
                  <select style={inputStyle} value={eng.ccc} onChange={(e) => set({ ccc: e.target.value })}>
                    <option value="">Select…</option><option>Available</option><option>Not Available</option><option>Not Applicable</option>
                  </select>
                </Field>
                <Field label="Measurements basis" mvs8 tip="MVS 6.2.2(h)">
                  <select style={inputStyle} value={eng.measurementBasis} onChange={(e) => set({ measurementBasis: e.target.value })}>
                    <option value="">Select…</option><option>UMMB</option><option>IPMS</option><option>Both</option>
                  </select>
                </Field>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 12, alignItems: "start" }}>
                <Field label="Statutory breach observed?">
                  <select style={inputStyle} value={eng.breach} onChange={(e) => set({ breach: e.target.value })}>
                    <option value="no">No</option><option value="yes">Yes</option>
                  </select>
                </Field>
                {eng.breach === "yes" && (
                  <Field label="Breach details"><input style={inputStyle} value={eng.breachNotes} onChange={(e) => set({ breachNotes: e.target.value })} /></Field>
                )}
              </div>
            </Card>
          </div>
        </SectionShell>

        {/* ════ SECTION 3 — Valuation approaches ════ */}
        <SectionShell id="s3" num={3} title="Comparable Evidence & Valuation Method" mvsRef="MVS 7 (Approaches to Valuations) · MVS 4.3.8">
          <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
            {[["market", "Market / Comparison"], ["income", `Income${incomeRelevant ? " (recommended)" : ""}`], ["cost", "Cost (DRC)"]].map(([k, l]) => (
              <button key={k} onClick={() => setTab(k)} style={{
                padding: "8px 18px", borderRadius: 5, fontSize: 12.5, cursor: "pointer", fontFamily: SANS,
                background: tab === k ? C.gold : C.surface, color: tab === k ? C.bg : C.textDim,
                border: `1px solid ${tab === k ? C.gold : C.border}`, fontWeight: tab === k ? 700 : 400,
              }}>{l}</button>
            ))}
          </div>

          {tab === "market" && (
            <div style={{ display: "grid", gap: 16 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))", gap: 16 }}>
                <Card title="Engine condition analysis" badge="engine">
                  <KV k="Engine baseline land value" v={fmtRM(P.landValue)} />
                  <KV k="Engine baseline structure value" v={fmtRM(P.structureValue)} />
                  <KV k="Health Factor (HF)" v={P.hf.toFixed(4)} />
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, margin: "14px 0" }}>
                    {Object.entries(P.factors).map(([k, v]) => (
                      <div key={k} style={{ textAlign: "center", background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 4px" }}>
                        <div style={{ fontSize: 10, color: C.textDim, letterSpacing: "0.08em" }}>{k}{k === "USS" && <Tip text="Applied as (1 − USS); high usage is the only inverted factor" />}</div>
                        <div style={{ fontSize: 16, fontWeight: 700, color: factorBand(k, v), fontFamily: SERIF }}>{v.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                  <Label>RTPMV trend — {range} days</Label>
                  <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                    {[7, 30].map((r) => (
                      <button key={r} onClick={() => setRange(r)} style={{
                        fontSize: 11, padding: "3px 10px", borderRadius: 4, cursor: "pointer",
                        background: range === r ? C.tealDim : C.bg, color: range === r ? C.bg : C.textDim, border: `1px solid ${C.border}`,
                      }}>{r}d</button>
                    ))}
                  </div>
                  <div style={{ height: 150 }}>
                    <ResponsiveContainer>
                      <AreaChart data={P.history.slice(-range)} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
                        <defs>
                          <linearGradient id="tvg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={C.teal} stopOpacity={0.35} />
                            <stop offset="100%" stopColor={C.teal} stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
                        <XAxis dataKey="day" tick={{ fill: C.textMuted, fontSize: 9 }} interval="preserveStartEnd" />
                        <YAxis tick={{ fill: C.textMuted, fontSize: 9 }} domain={["auto", "auto"]} tickFormatter={(v) => (v / 1000000).toFixed(2) + "M"} width={42} />
                        <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, fontSize: 11 }} formatter={(v) => [fmtRM(v), "RTPMV"]} />
                        <Area dataKey="rtpmv" stroke={C.teal} fill="url(#tvg)" strokeWidth={1.5} />
                        {P.history.slice(-range).map((d, i) => d.event ? <ReferenceDot key={i} x={d.day} y={d.rtpmv} r={4} fill={C.danger} stroke="none" /> : null)}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <Label>Engine event log</Label>
                    {P.events.map((ev, i) => (
                      <div key={i} style={{ fontSize: 11.5, color: C.textDim, padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                        <span style={{ color: C.teal }}>{ev.date}</span> · <strong style={{ color: C.text }}>{ev.label}</strong> — {ev.detail}
                      </div>
                    ))}
                  </div>
                </Card>

                <Card title="Valuer's reconciliation" badge="valuer">
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    <Field label="Land area (sq ft)" mvs8><input type="number" style={inputStyle} value={eng.landArea} onChange={(e) => set({ landArea: e.target.value })} /></Field>
                    <Field label="Built-up area (sq ft)" mvs8><input type="number" style={inputStyle} value={eng.builtUpArea} onChange={(e) => set({ builtUpArea: e.target.value })} /></Field>
                    <Field label="Reconciled land rate (RM/sq ft)" mvs8><input type="number" style={inputStyle} value={eng.landRate} onChange={(e) => set({ landRate: e.target.value })} /></Field>
                    <Field label="Reconciled structure rate (RM/sq ft)" mvs8><input type="number" style={inputStyle} value={eng.structureRate} onChange={(e) => set({ structureRate: e.target.value })} /></Field>
                  </div>
                  <Field label="Adjustment narrative" tip="MVS 7.2.1.1(d) — explain the adjustments made to comparables">
                    <textarea rows={3} style={{ ...inputStyle, resize: "vertical" }} value={eng.adjustmentNarrative} onChange={(e) => set({ adjustmentNarrative: e.target.value })} />
                  </Field>
                  <div style={{ background: C.bg, border: `1px solid ${C.borderHi}`, borderRadius: 6, padding: 14, marginTop: 6 }}>
                    <KV k="Valuer's Market Value (computed)" v={<strong style={{ color: C.gold, fontSize: 15 }}>{fmtRM(valuerMV)}</strong>} />
                    <KV k="Engine RTPMV (reference)" v={<span style={{ color: C.teal }}>{fmtRM(P.rtpmv)}</span>} />
                    <KV k="Delta" v={<span style={{ color: deltaConsistent ? C.success : C.warning }}>{valuerMV ? `${deltaPct >= 0 ? "+" : ""}${deltaPct.toFixed(1)}%` : "—"}</span>} />
                    <div style={{ fontSize: 11, color: C.textMuted, marginTop: 8 }}>
                      Valuer MV = (Land Area × Land Rate) + (Built-up Area × Structure Rate × HF {P.hf.toFixed(4)} from engine).{" "}
                      {valuerMV > 0 && (deltaConsistent
                        ? "Delta within ±10% — engine condition indicators are consistent with the Valuer's field assessment."
                        : "Delta exceeds ±10% — engine condition indicators are inconsistent with the Valuer's field assessment; reconcile and disclose.")}
                    </div>
                  </div>
                </Card>
              </div>

              <Card title="Comparable transactions (minimum 3 — MVS 7.2.1.1(g))" badge="valuer">
                <div style={{ overflowX: "auto" }}>
                  <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 11.5, minWidth: 1050 }}>
                    <thead>
                      <tr>
                        {["Lot / Title No.", "Address", "Txn date", "Consideration (RM)", "Description", "Area (sq ft)", "Tenure", "Source", "Adj. %", "Adj. reason", "Adj. RM/sq ft", ""].map((h) => (
                          <th key={h} style={{ textAlign: "left", color: C.textDim, fontWeight: 600, padding: "6px 8px", borderBottom: `1px solid ${C.borderHi}`, whiteSpace: "nowrap" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {eng.comps.map((c, i) => {
                        const adj = num(c.area) > 0 ? (num(c.consideration) * (1 + num(c.adjPct) / 100)) / num(c.area) : 0;
                        const cell = (key, type = "text", w = 110) => (
                          <td style={{ padding: 4 }}>
                            <input type={type} style={{ ...inputStyle, padding: "5px 7px", fontSize: 11.5, width: w }} value={c[key]}
                              onChange={(e) => set({ comps: eng.comps.map((x, xi) => xi === i ? { ...x, [key]: e.target.value } : x) })} />
                          </td>
                        );
                        return (
                          <tr key={i}>
                            {cell("id")}{cell("address", "text", 170)}{cell("date", "date", 125)}{cell("consideration", "number")}
                            {cell("desc", "text", 130)}{cell("area", "number", 85)}
                            <td style={{ padding: 4 }}>
                              <select style={{ ...inputStyle, padding: "5px 7px", fontSize: 11.5, width: 95 }} value={c.tenure}
                                onChange={(e) => set({ comps: eng.comps.map((x, xi) => xi === i ? { ...x, tenure: e.target.value } : x) })}>
                                <option value="">—</option><option>Freehold</option><option>Leasehold</option>
                              </select>
                            </td>
                            <td style={{ padding: 4 }}>
                              <select style={{ ...inputStyle, padding: "5px 7px", fontSize: 11.5, width: 110 }} value={c.source}
                                onChange={(e) => set({ comps: eng.comps.map((x, xi) => xi === i ? { ...x, source: e.target.value } : x) })}>
                                <option value="">—</option><option>JPPH</option><option>S&P Agreement</option><option>Bursa</option><option>Other</option>
                              </select>
                            </td>
                            {cell("adjPct", "number", 60)}{cell("adjReason", "text", 130)}
                            <td style={{ padding: "4px 8px", color: C.gold, whiteSpace: "nowrap" }}>{adj ? adj.toFixed(0) : "—"}</td>
                            <td style={{ padding: 4 }}>
                              {eng.comps.length > 3 && (
                                <button onClick={() => set({ comps: eng.comps.filter((_, xi) => xi !== i) })}
                                  style={{ background: "none", border: "none", color: C.danger, cursor: "pointer", fontSize: 14 }}>×</button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <button onClick={() => set({ comps: [...eng.comps, emptyComp()] })}
                  style={{ marginTop: 10, background: C.surfaceAlt, border: `1px solid ${C.border}`, color: C.textDim, borderRadius: 4, padding: "6px 14px", fontSize: 12, cursor: "pointer" }}>
                  + Add comparable
                </button>
                <span style={{ marginLeft: 12, fontSize: 11.5, color: compsComplete.length >= 3 ? C.success : C.warning }}>
                  {compsComplete.length} of 3 minimum comparables complete
                </span>
              </Card>
            </div>
          )}

          {tab === "income" && (
            <Card title={`Income approach${incomeRelevant ? "" : " — optional (purpose is not Financing / Financial Reporting)"}`} badge="valuer">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
                <Field label="Gross annual rent (RM)"><input type="number" style={inputStyle} value={income.gross} onChange={(e) => setDeep("income", { gross: e.target.value })} /></Field>
                <Field label="Management fee (RM)"><input type="number" style={inputStyle} value={income.mgmt} onChange={(e) => setDeep("income", { mgmt: e.target.value })} /></Field>
                <Field label="Maintenance (RM)"><input type="number" style={inputStyle} value={income.maint} onChange={(e) => setDeep("income", { maint: e.target.value })} /></Field>
                <Field label="Insurance (RM)"><input type="number" style={inputStyle} value={income.ins} onChange={(e) => setDeep("income", { ins: e.target.value })} /></Field>
                <Field label="Assessment / quit rent (RM)"><input type="number" style={inputStyle} value={income.assess} onChange={(e) => setDeep("income", { assess: e.target.value })} /></Field>
                <Field label="Capitalisation rate (%)"><input type="number" style={inputStyle} value={income.capRate} onChange={(e) => setDeep("income", { capRate: e.target.value })} /></Field>
                <Field label="Yield evidence source"><input style={inputStyle} value={income.yieldSource} onChange={(e) => setDeep("income", { yieldSource: e.target.value })} placeholder="Comparable yield evidence" /></Field>
                <Field label="Void allowance (%)"><input type="number" style={inputStyle} value={income.voidPct} onChange={(e) => setDeep("income", { voidPct: e.target.value })} /></Field>
              </div>
              <div style={{ background: C.bg, border: `1px solid ${C.borderHi}`, borderRadius: 6, padding: 14, marginTop: 8, maxWidth: 460 }}>
                <KV k="Net annual rent" v={fmtRM(netRent)} />
                <KV k="Capitalised value" v={fmtRM(capValue)} />
                <KV k="Investment Value" v={<strong style={{ color: C.gold }}>{fmtRM(investmentValue)}</strong>} />
              </div>
            </Card>
          )}

          {tab === "cost" && (
            <Card title="Cost approach (DRC) — optional, for specialised property" badge="valuer">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
                <Field label="Current replacement cost (RM)" tip="From QS estimate or contractor quote"><input type="number" style={inputStyle} value={cost.crc} onChange={(e) => setDeep("cost", { crc: e.target.value })} /></Field>
                <Field label="Physical depreciation (%)"><input type="number" style={inputStyle} value={cost.phys} onChange={(e) => setDeep("cost", { phys: e.target.value })} /></Field>
                <Field label="Physical basis"><input style={inputStyle} value={cost.physBasis} onChange={(e) => setDeep("cost", { physBasis: e.target.value })} /></Field>
                <Field label="Functional obsolescence (%)"><input type="number" style={inputStyle} value={cost.func} onChange={(e) => setDeep("cost", { func: e.target.value })} /></Field>
                <Field label="Functional basis"><input style={inputStyle} value={cost.funcBasis} onChange={(e) => setDeep("cost", { funcBasis: e.target.value })} /></Field>
                <Field label="Economic obsolescence (%)"><input type="number" style={inputStyle} value={cost.econ} onChange={(e) => setDeep("cost", { econ: e.target.value })} /></Field>
                <Field label="Economic basis"><input style={inputStyle} value={cost.econBasis} onChange={(e) => setDeep("cost", { econBasis: e.target.value })} /></Field>
              </div>
              <div style={{ background: C.bg, border: `1px solid ${C.borderHi}`, borderRadius: 6, padding: 14, marginTop: 8, maxWidth: 460 }}>
                <KV k="Depreciated replacement cost" v={fmtRM(drc)} />
                <KV k="Land value (auto-linked from Market tab)" v={fmtRM(marketLandValue)} />
                <KV k="DRC Value" v={<strong style={{ color: C.gold }}>{fmtRM(drcValue)}</strong>} />
              </div>
            </Card>
          )}
        </SectionShell>

        {/* ════ SECTION 4 — CI & audit trail ════ */}
        <SectionShell id="s4" num={4} title="Confidence Index & Audit Trail" mvsRef="MVS 7.2.3 (Record of Valuation Work) · Patent hash-chain">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))", gap: 16 }}>
            <Card title={<span>Confidence Index breakdown <Tip text={MVS_DEFS.ci} /></span>} badge="engine">
              {P.ciBreakdown.map((row) => (
                <div key={row.k} style={{ padding: "8px 0", borderBottom: `1px solid ${C.border}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
                    <span style={{ color: C.textDim }}>{row.k}</span>
                    <strong style={{ color: row.v < 0 ? C.warning : row.v >= 0.8 ? C.success : row.v >= 0.6 ? C.warning : C.danger }}>
                      {row.v >= 0 ? row.v.toFixed(row.v === 1 ? 1 : 3) : row.v.toFixed(2)}
                    </strong>
                  </div>
                  <div style={{ fontSize: 10.5, color: C.textMuted }}>{row.note}</div>
                </div>
              ))}
              <KV k="Composite CI" v={<strong style={{ color: P.ci >= 0.8 ? C.success : P.ci >= 0.65 ? C.warning : C.danger, fontSize: 15 }}>{P.ci.toFixed(4)}</strong>} />
              <div style={{ marginTop: 14 }}>
                <Label>Cryptographic record (tamper-evident)</Label>
                <KV k="Valuation token ID" v={P.tokenId} mono />
                <KV k="Current record hash (SHA-256)" v={P.hash} mono />
                <KV k="Previous record hash (chain link)" v={P.prevHash} mono />
              </div>
            </Card>

            <Card title="Valuer's reliability assessment & observations" badge="valuer">
              <Field label="Data reliability conclusion" mvs8>
                <select style={inputStyle} value={eng.ciRating} onChange={(e) => set({ ciRating: e.target.value })}>
                  <option value="">Select…</option><option>Reliable</option><option>Partially Reliable</option><option>Unreliable</option>
                </select>
              </Field>
              <Field label="Assessment narrative">
                <textarea rows={3} style={{ ...inputStyle, resize: "vertical" }} value={eng.ciNarrative} onChange={(e) => set({ ciNarrative: e.target.value })}
                  placeholder="Based on field inspection, the sensor data is considered … because …" />
              </Field>
              <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 14, marginTop: 4 }}>
                <Label>Submit field observation (write-once — feeds engine CI)</Label>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}>
                  <input type="date" style={inputStyle} value={obsForm.date} onChange={(e) => setObsForm({ ...obsForm, date: e.target.value })} />
                  <select style={inputStyle} value={obsForm.zone} onChange={(e) => setObsForm({ ...obsForm, zone: e.target.value })}>
                    <option value="">Zone…</option>
                    {["roof", "living", "kitchen", "bed1", "bed2", "bed3", "utility", "exterior", "interior"].map((z) => <option key={z}>{z}</option>)}
                  </select>
                  <select style={inputStyle} value={obsForm.severity} onChange={(e) => setObsForm({ ...obsForm, severity: e.target.value })}>
                    <option>Normal</option><option>Watch</option><option>Alert</option>
                  </select>
                </div>
                <textarea rows={2} style={{ ...inputStyle, resize: "vertical", marginBottom: 10 }} placeholder="Observation description…"
                  value={obsForm.note} onChange={(e) => setObsForm({ ...obsForm, note: e.target.value })} />
                <button onClick={submitObservation} disabled={!obsForm.zone || !obsForm.note.trim()}
                  style={{
                    background: obsForm.zone && obsForm.note.trim() ? C.gold : C.surfaceAlt,
                    color: obsForm.zone && obsForm.note.trim() ? C.bg : C.textMuted,
                    border: "none", borderRadius: 4, padding: "8px 18px", fontSize: 12, fontWeight: 700, cursor: "pointer",
                  }}>
                  SUBMIT TO ENGINE
                </button>
                <div style={{ fontSize: 10.5, color: C.textMuted, marginTop: 8 }}>
                  Observations are write-once. Once submitted they cannot be deleted — only voided with a recorded reason, preserving the tamper-evident audit trail.
                </div>
              </div>
            </Card>
          </div>

          <Card title="Observation log" badge="engine" style={{ marginTop: 16 }}>
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
                <thead>
                  <tr>{["Date", "Zone", "Severity", "Observer", "Note", "Status", ""].map((h) => (
                    <th key={h} style={{ textAlign: "left", color: C.textDim, fontWeight: 600, padding: "6px 10px", borderBottom: `1px solid ${C.borderHi}` }}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {allObs.map((o, i) => (
                    <tr key={i} style={{ opacity: o.status === "VOIDED" ? 0.45 : 1 }}>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}`, whiteSpace: "nowrap" }}>{o.date}</td>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}` }}>{o.zone}</td>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}` }}>
                        <span style={{ color: o.severity === "Alert" ? C.danger : o.severity === "Watch" ? C.warning : C.success, fontWeight: 600 }}>{o.severity}</span>
                      </td>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}` }}>{o.observer}</td>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}` }}>{o.note}{o.voidReason ? ` (void reason: ${o.voidReason})` : ""}</td>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}` }}>
                        {o.status}{o.pending && <span style={{ color: C.teal, fontSize: 10, marginLeft: 6 }}>PENDING SYNC</span>}
                      </td>
                      <td style={{ padding: "7px 10px", borderBottom: `1px solid ${C.border}` }}>
                        {o._local != null && o.status === "ACTIVE" && (
                          <button onClick={() => voidObservation(o._local)} style={{ background: "none", border: `1px solid ${C.border}`, color: C.textDim, borderRadius: 3, fontSize: 10.5, padding: "2px 8px", cursor: "pointer" }}>Void…</button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </SectionShell>

        {/* ════ SECTION 5 — Report ════ */}
        <SectionShell id="s5" num={5} title="Report Generation" mvsRef="MVS 8 (Valuation Reports) · MVS 9 (Assumptions) · MVS 19 (Limiting Conditions)">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))", gap: 16 }}>
            <Card title="Engagement summary (auto-assembled)">
              <KV k="Property" v={`${P.name} (${P.id})`} />
              <KV k="Client / Purpose" v={`${eng.client || "—"} · ${eng.purpose || "—"}`} />
              <KV k={<span>Basis of value <Tip text={MVS_DEFS.marketValue} /></span>} v={eng.basis || "—"} />
              <KV k="Valuation date / Inspection date" v={`${eng.valuationDate} / ${eng.inspectionDate}`} />
              <KV k="Inspector" v={eng.inspector === "valuer" ? `${auth.name} (Valuer)` : `${eng.assistant || "—"} (Designated Assistant)`} />
              <KV k="Approaches used" v={approachesUsed.length ? approachesUsed.join("; ") : "—"} />
              <KV k="Engine RTPMV (reference only)" v={<span style={{ color: C.teal }}>{fmtRM(P.rtpmv)}</span>} />
              <KV k="Trading status (information only)" v={P.trading} />
              <div style={{ marginTop: 16 }}>
                <Label mvs8>Valuer's Opinion of Value (RM)</Label>
                <input type="number" value={eng.opinionOfValue} onChange={(e) => set({ opinionOfValue: e.target.value })}
                  style={{ ...inputStyle, fontSize: 22, fontFamily: SERIF, color: C.gold, border: `1px solid ${C.gold}`, padding: "12px 14px", fontWeight: 700 }} />
                <div style={{ fontSize: 11, color: C.textMuted, marginTop: 6 }}>
                  The professional opinion of value — formed by the Valuer, never the engine output. <Badge kind="valuer" />
                </div>
              </div>
            </Card>

            <Card title="Assumptions & limiting conditions">
              <Label tip="MVS 19 standard limiting conditions — pre-ticked; untick any that do not apply">Limiting conditions (MVS 19)</Label>
              <div style={{ maxHeight: 200, overflowY: "auto", marginBottom: 14, paddingRight: 4 }}>
                {LIMITING_CONDITIONS.map((l) => (
                  <label key={l.id} style={{ display: "flex", gap: 8, fontSize: 11.5, color: C.textDim, padding: "5px 0", cursor: "pointer" }}>
                    <input type="checkbox" checked={!!eng.limiting[l.id]} onChange={(e) => setDeep("limiting", { [l.id]: e.target.checked })} style={{ marginTop: 2 }} />
                    {l.text}
                  </label>
                ))}
              </div>
              <Field label="Additional assumptions (MVS 9)" tip={MVS_DEFS.assumption}>
                <textarea rows={2} style={{ ...inputStyle, resize: "vertical" }} value={eng.additionalAssumptions} onChange={(e) => set({ additionalAssumptions: e.target.value })} />
              </Field>
              {eng.additionalAssumptions.trim() && (
                <>
                  <div style={{ fontSize: 11, color: C.warning, border: `1px solid ${C.warning}`, borderRadius: 5, padding: 10, marginBottom: 12, fontWeight: 700 }}>
                    {MVS9_PROVISO}
                    <div style={{ fontWeight: 400, marginTop: 4, color: C.textMuted }}>Auto-inserted in bold capitals in the generated report (MVS 9.2.3).</div>
                  </div>
                  <Field label='"As Is" value (RM) — required when additional assumptions are made (MVS 9.2.2)'>
                    <input type="number" style={inputStyle} value={eng.asIsValue} onChange={(e) => set({ asIsValue: e.target.value })} />
                  </Field>
                </>
              )}
              {forcedSale && (
                <div style={{ fontSize: 11, color: C.textDim, border: `1px solid ${C.border}`, borderRadius: 5, padding: 10, marginBottom: 12 }}>
                  {MVS12_DISCLOSURE}
                  <div style={{ marginTop: 4, color: C.textMuted }}>Auto-inserted: Purpose = Financing and Basis = Forced Sale Value.</div>
                </div>
              )}

              {!canGenerate && (
                <div style={{ border: `1px solid ${C.warning}`, borderRadius: 6, padding: 12, marginBottom: 14 }}>
                  <div style={{ fontSize: 11.5, color: C.warning, fontWeight: 700, marginBottom: 6 }}>MVS 8 checklist — outstanding before report generation:</div>
                  {outstanding.map((o) => <div key={o} style={{ fontSize: 11, color: C.textDim }}>· {o}</div>)}
                </div>
              )}
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button onClick={() => { persist(propId); setShowReport(true); }} disabled={!canGenerate}
                  style={{ background: canGenerate ? C.gold : C.surfaceAlt, color: canGenerate ? C.bg : C.textMuted, border: "none", borderRadius: 5, padding: "10px 18px", fontWeight: 700, fontSize: 12, cursor: canGenerate ? "pointer" : "not-allowed" }}>
                  GENERATE DRAFT REPORT
                </button>
                <button onClick={() => { if (canGenerate) { persist(propId); setShowReport(true); setTimeout(() => window.print(), 350); } }} disabled={!canGenerate}
                  style={{ background: "none", color: canGenerate ? C.gold : C.textMuted, border: `1px solid ${canGenerate ? C.gold : C.border}`, borderRadius: 5, padding: "10px 18px", fontWeight: 700, fontSize: 12, cursor: canGenerate ? "pointer" : "not-allowed" }}>
                  DOWNLOAD PDF
                </button>
                <button onClick={exportRecord}
                  style={{ background: "none", color: C.teal, border: `1px solid ${C.tealDim}`, borderRadius: 5, padding: "10px 18px", fontWeight: 700, fontSize: 12, cursor: "pointer" }}>
                  EXPORT VALUATION RECORD (JSON)
                </button>
              </div>
            </Card>
          </div>
        </SectionShell>

        <footer style={{ borderTop: `1px solid ${C.border}`, paddingTop: 18, fontSize: 10.5, color: C.textMuted, lineHeight: 1.7 }}>
          TwinVal Valuers is an independent appraisal workbench. The TwinVal engine provides sensor-verified condition data;
          the Registered Valuer provides the professional opinion of value in accordance with the Malaysian Valuation Standards
          (7th Edition, 2025) issued by LPPEH. Indian Patent Application No. 202641030498. Demonstration data shown.
        </footer>
      </main>

      {/* ── Draft report (MVS 8.2.2 structure) ── */}
      {showReport && (
        <ReportView P={P} eng={eng} auth={auth} approachesUsed={approachesUsed} valuerMV={valuerMV}
          investmentValue={investmentValue} drcValue={drcValue} forcedSale={forcedSale}
          compsComplete={compsComplete} onClose={() => setShowReport(false)} />
      )}
      <style>{`
        @media print {
          .no-print { display: none !important; }
          #tv-report-overlay { position: static !important; overflow: visible !important; background: white !important; padding: 0 !important; }
          #tv-report { box-shadow: none !important; max-width: 100% !important; }
        }
        select option { background: ${C.surface}; }
        input[type="date"]::-webkit-calendar-picker-indicator { filter: invert(0.7); }
      `}</style>
    </div>
  );
}

// ── Draft report view ────────────────────────────────────────────────────────
function ReportView({ P, eng, auth, approachesUsed, valuerMV, investmentValue, drcValue, forcedSale, compsComplete, onClose }) {
  const H = ({ n, t }) => (
    <h3 style={{ fontFamily: SERIF, fontSize: 16, margin: "18px 0 6px", color: C.ink }}>{n}. {t}</h3>
  );
  const Pg = ({ children }) => <p style={{ fontSize: 12, lineHeight: 1.65, margin: "4px 0", color: C.ink }}>{children}</p>;
  const basisDefinition = eng.basis === "Market Value"
    ? "Market Value is the estimated amount for which an asset should exchange on the valuation date between a willing buyer and a willing seller in an arm's-length transaction, after proper marketing and where the parties had each acted knowledgeably, prudently and without compulsion (MVS 4.3.1)."
    : `${eng.basis} as defined in MVS 4 / MVS 5.`;
  return (
    <div id="tv-report-overlay" style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(8,16,26,0.85)", overflowY: "auto", padding: "30px 16px" }}>
      <div id="tv-report" style={{ maxWidth: 860, margin: "0 auto", background: C.paper, color: C.ink, borderRadius: 6, padding: "46px 54px", fontFamily: SANS, boxShadow: "0 18px 60px rgba(0,0,0,0.5)" }}>
        <div className="no-print" style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginBottom: 12 }}>
          <button onClick={() => window.print()} style={{ background: C.bg, color: C.gold, border: "none", borderRadius: 4, padding: "7px 16px", fontSize: 12, cursor: "pointer", fontWeight: 700 }}>Print / Save as PDF</button>
          <button onClick={onClose} style={{ background: "none", border: `1px solid #999`, color: "#555", borderRadius: 4, padding: "7px 16px", fontSize: 12, cursor: "pointer" }}>Close</button>
        </div>
        <div style={{ textAlign: "center", borderBottom: `2px solid ${C.gold}`, paddingBottom: 14, marginBottom: 8 }}>
          <div style={{ fontFamily: SERIF, fontSize: 26, fontWeight: 700 }}>VALUATION REPORT — DRAFT</div>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", color: "#666" }}>PREPARED IN ACCORDANCE WITH THE MALAYSIAN VALUATION STANDARDS (7TH EDITION, 2025)</div>
        </div>
        <Pg><em>Draft for review — not valid until signed by the Registered Valuer.</em></Pg>

        <H n={1} t="Instructions to Value" /><Pg>Instructed by {eng.client}. Intended user(s): {eng.intendedUsers || eng.client}.</Pg>
        <H n={2} t="Interest to be Valued" /><Pg>{P.title.tenure} interest in {P.title.address} ({P.title.titleNo}, {P.title.lotNo}).</Pg>
        <H n={3} t="Purpose of Valuation" /><Pg>{eng.purpose} (MVS 3.2.2).</Pg>
        <H n={4} t="Valuation Date and Inspection Date" />
        <Pg>Valuation date: {eng.valuationDate}. Inspection date: {eng.inspectionDate}. Inspected by: {eng.inspector === "valuer" ? `${auth.name}, Registered Valuer ${auth.reg}` : `${eng.assistant} (Designated Assistant), under the supervision of ${auth.name} (${auth.reg})`}. Measurements basis: {eng.measurementBasis} (MVS 6.2.2(h)).</Pg>
        <H n={5} t="Title Particulars" />
        <Pg>{P.title.mukim}. Tenure: {P.title.tenure}. Category of land use: {P.title.category}. Express conditions: {P.title.expressConditions}. Encumbrances: {P.title.encumbrances}. {P.title.lastTransaction}.</Pg>
        <H n={6} t="Description of Property" />
        <Pg>{P.typeLong}. Building age (chronological): {eng.buildingAge} years. CF/CCC: {eng.ccc || "—"}.</Pg>
        {Object.entries(eng.zoneNotes).filter(([, v]) => v.trim()).map(([k, v]) => <Pg key={k}><strong style={{ textTransform: "capitalize" }}>{k}:</strong> {v}</Pg>)}
        {eng.defects.trim() && <Pg><strong>Visible defects (MVS 6.2.2(e)):</strong> {eng.defects}</Pg>}
        {eng.accessLimited && <Pg><strong>Inspection limitation (MVS 6.2.2(e)):</strong> {eng.accessNotes || "Access limitations were encountered during inspection."}</Pg>}
        {eng.breach === "yes" && <Pg><strong>Statutory breach observed:</strong> {eng.breachNotes}</Pg>}
        <H n={7} t="Tenancy / Lease Details" /><Pg>{num(eng.income.gross) > 0 ? `Gross annual rent ${fmtRM(num(eng.income.gross))}; see Income Approach workings.` : "Owner-occupied / no tenancy details provided."}</Pg>
        <H n={8} t="Planning Details" /><Pg>Category of land use: {P.title.category}. Express conditions: {P.title.expressConditions}.</Pg>
        <H n={9} t="Assumptions" />
        {eng.additionalAssumptions.trim() ? (
          <>
            <Pg>{eng.additionalAssumptions}</Pg>
            <Pg><strong>{MVS9_PROVISO}</strong></Pg>
            {eng.asIsValue && <Pg>"As Is" value (MVS 9.2.2): {fmtRM(num(eng.asIsValue))}.</Pg>}
          </>
        ) : <Pg>No additional assumptions have been made (MVS 9).</Pg>}
        <H n={10} t="Basis of Value" /><Pg>{basisDefinition}</Pg>
        {forcedSale && <Pg><strong>{MVS12_DISCLOSURE}</strong></Pg>}
        <H n={11} t="Approaches and Methods" /><Pg>{approachesUsed.join("; ") || "—"} (MVS 7).</Pg>
        <H n={12} t="Evidence of Value" />
        <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 10.5, margin: "6px 0" }}>
          <thead><tr>{["Lot/Title", "Address", "Date", "Consideration", "Area", "Adj.", "Adj. RM/sq ft"].map((h) => (
            <th key={h} style={{ border: "1px solid #999", padding: "4px 6px", textAlign: "left", background: "#EAE5D8" }}>{h}</th>
          ))}</tr></thead>
          <tbody>
            {compsComplete.map((c, i) => (
              <tr key={i}>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{c.id}</td>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{c.address}</td>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{c.date}</td>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{fmtRM(num(c.consideration))}</td>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{c.area} sq ft</td>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{c.adjPct || 0}% — {c.adjReason}</td>
                <td style={{ border: "1px solid #999", padding: "4px 6px" }}>{num(c.area) > 0 ? ((num(c.consideration) * (1 + num(c.adjPct) / 100)) / num(c.area)).toFixed(0) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {eng.adjustmentNarrative.trim() && <Pg><strong>Adjustment narrative (MVS 7.2.1.1(d)):</strong> {eng.adjustmentNarrative}</Pg>}
        <Pg>Reconciled rates: land {fmtRM(num(eng.landRate))}/sq ft on {eng.landArea} sq ft; structure {fmtRM(num(eng.structureRate))}/sq ft on {eng.builtUpArea} sq ft (adjusted by engine Health Factor {P.hf.toFixed(4)}). Market Approach workings: {fmtRM(valuerMV)}.{investmentValue > 0 ? ` Income Approach: Investment Value ${fmtRM(investmentValue)}.` : ""}{drcValue > 0 && num(eng.cost.crc) > 0 ? ` Cost Approach: DRC Value ${fmtRM(drcValue)}.` : ""}</Pg>
        <H n={13} t="TwinVal Engine Condition Analysis — Supplementary Appendix" />
        <div style={{ border: "1px solid #999", background: "#EFF6F5", padding: "10px 12px", fontSize: 11, lineHeight: 1.6 }}>
          <strong>SUPPLEMENTARY CONDITION DATA — NOT THE VALUATION.</strong> Engine RTPMV {fmtRM(P.rtpmv)} (Health Factor {P.hf.toFixed(4)}:
          SHF {P.factors.SHF.toFixed(3)}, ESF {P.factors.ESF.toFixed(3)}, USS {P.factors.USS.toFixed(3)}, PDP {P.factors.PDP.toFixed(3)}, CI {P.factors.CI.toFixed(3)}).
          Sensor status: {P.sensorStatus}; last sync {P.lastSync}. Confidence Index {P.ci.toFixed(4)} — assessed by the Valuer as
          {" "}<strong>{eng.ciRating}</strong>{eng.ciNarrative.trim() ? ` — ${eng.ciNarrative}` : ""}.
          Tamper-evident record: token {P.tokenId}; SHA-256 {P.hash.slice(0, 16)}…; chained to {P.prevHash.slice(0, 16)}…
          Trading status (information only): {P.trading}.
        </div>
        <H n={14} t="Opinion of Value" />
        <div style={{ border: `2px solid ${C.gold}`, padding: "12px 16px", margin: "8px 0", fontSize: 15 }}>
          The Valuer's opinion of the {eng.basis} of the subject property, as at {eng.valuationDate}, is{" "}
          <strong style={{ fontFamily: SERIF, fontSize: 19 }}>{fmtRM(num(eng.opinionOfValue))}</strong>.
        </div>
        <H n={15} t="Limiting Conditions" />
        {LIMITING_CONDITIONS.filter((l) => eng.limiting[l.id]).map((l) => <Pg key={l.id}>· {l.text}</Pg>)}
        <H n={16} t="Valuer" />
        <div style={{ marginTop: 26, display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <div>
            <div style={{ borderTop: "1px solid #555", width: 240, paddingTop: 6, fontSize: 12 }}>
              <strong>{auth.name}</strong><br />Registered Valuer, LPPEH No. {auth.reg}<br />Date: {todayISO()}
            </div>
          </div>
          <div style={{ fontSize: 10, color: "#777", textAlign: "right" }}>
            Generated by TwinVal Valuers — independent appraisal workbench.<br />
            Engagement record retained per MVS 7.2.3.
          </div>
        </div>
      </div>
    </div>
  );
}
