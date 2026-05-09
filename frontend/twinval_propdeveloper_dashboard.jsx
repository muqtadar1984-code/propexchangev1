import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, Legend, ReferenceLine,
} from "recharts";

// Route hint (informational only — wire-up handled outside this file):
//   app.twinval.com/propdeveloper
//
// TwinVal PropDeveloper — Property Developer Dashboard
// Simulated demo for Indian property developer audiences.
// Patent: India Application No. 202641030498 (Muqtadar Musawir Quraishi)

// ── Colour System ─────────────────────────────────────────────────────────────
const C = {
  bg:        "#0a0a0a",
  surface:   "#111111",
  surfaceAlt:"#161616",
  border:    "#1e1e1e",
  borderHi:  "#2a2a2a",
  gold:      "#c9a96e",
  goldDim:   "#8a7548",
  text:      "#f5f0e8",
  textDim:   "#8a8a8a",
  textMuted: "#5a5a5a",
  success:   "#4caf7d",
  warning:   "#e8a045",
  danger:    "#e05c5c",
};

// ── Static Demo Data ──────────────────────────────────────────────────────────
const PROJECTS = [
  {
    id: "EMPRESS",
    name: "Empress Skyline Tower",
    city: "Hyderabad",
    fullLocation: "Financial District, Hyderabad, Telangana",
    type: "Grade A Commercial",
    typeLong: "Grade A Commercial Office",
    progress: 68,
    buildHealth: 82,
    builtUpArea: "1,24,000 sq ft",
    floors: "28 (B2 to L26)",
    developer: "Aethel Developments Pvt. Ltd.",
    structuralEng: "[Placeholder — live: registered engineer + licence]",
    govtValue: 310,
    govtRef: "GHMC Reference: HYD/2025/COMM/4412",
    startDate: "14 March 2024",
    completionDate: "30 September 2026",
    currentPhase: "MEP & Interior Finishing",
    rtpmvCurrent: 287,
    rtpmvCompletion: 334,
    landValue: 95,
    structureValue: 215,
  },
  {
    id: "NAVIMUMBAI",
    name: "Navi Mumbai Logistics Park",
    city: "Navi Mumbai",
    fullLocation: "Taloja MIDC, Navi Mumbai, Maharashtra",
    type: "Industrial/Warehouse",
    typeLong: "Industrial / Warehousing",
    progress: 45,
    buildHealth: 71,
    builtUpArea: "98,000 sq ft",
    floors: "G + 2",
    developer: "Aethel Developments Pvt. Ltd.",
    structuralEng: "[Placeholder — live: registered engineer + licence]",
    govtValue: 168,
    govtRef: "MIDC Reference: TLJ/2024/IND/0918",
    startDate: "10 January 2025",
    completionDate: "15 December 2026",
    currentPhase: "Superstructure",
    rtpmvCurrent: 142,
    rtpmvCompletion: 187,
    landValue: 70,
    structureValue: 98,
  },
  {
    id: "WHITEFIELD",
    name: "Whitefield Residences Phase 2",
    city: "Bengaluru",
    fullLocation: "Whitefield, Bengaluru, Karnataka",
    type: "Residential",
    typeLong: "Premium Residential",
    progress: 91,
    buildHealth: 88,
    builtUpArea: "1,02,000 sq ft",
    floors: "G + 18",
    developer: "Aethel Developments Pvt. Ltd.",
    structuralEng: "[Placeholder — live: registered engineer + licence]",
    govtValue: 358,
    govtRef: "BBMP Reference: BLR/2023/RES/2274",
    startDate: "5 May 2023",
    completionDate: "30 July 2026",
    currentPhase: "Testing & Commissioning",
    rtpmvCurrent: 384,
    rtpmvCompletion: 412,
    landValue: 140,
    structureValue: 218,
  },
  {
    id: "PUNETECH",
    name: "Pune Tech Campus Block C",
    city: "Pune",
    fullLocation: "Hinjewadi Phase III, Pune, Maharashtra",
    type: "Mixed-Use",
    typeLong: "Mixed-Use IT Campus",
    progress: 23,
    buildHealth: 64,
    builtUpArea: "60,000 sq ft",
    floors: "G + 9",
    developer: "Aethel Developments Pvt. Ltd.",
    structuralEng: "[Placeholder — live: registered engineer + licence]",
    govtValue: 112,
    govtRef: "PMRDA Reference: PUN/2025/MIX/3380",
    startDate: "20 August 2025",
    completionDate: "30 March 2027",
    currentPhase: "Foundation & Substructure",
    rtpmvCurrent: 98,
    rtpmvCompletion: 156,
    landValue: 52,
    structureValue: 60,
  },
];

const PHASES_BY_PROJECT = {
  EMPRESS: [
    { name: "Foundation & Substructure",      status: "done",   range: "14 Mar 2024 to 30 Jun 2024" },
    { name: "Superstructure (Columns, Slabs)",status: "done",   range: "1 Jul 2024 to 31 Jan 2025" },
    { name: "Façade & Roofing",               status: "done",   range: "1 Feb 2025 to 31 Jul 2025" },
    { name: "MEP & Interior Finishing",       status: "active", range: "1 Aug 2025 to present", pct: 68 },
    { name: "Testing, Commissioning & Handover", status: "pending", range: "Est. Jun 2026" },
  ],
  NAVIMUMBAI: [
    { name: "Foundation & Substructure",      status: "done",   range: "10 Jan 2025 to 30 Apr 2025" },
    { name: "Superstructure (Columns, Slabs)",status: "active", range: "1 May 2025 to present", pct: 45 },
    { name: "Façade & Roofing",               status: "pending", range: "Est. Q1 2026" },
    { name: "MEP & Interior Finishing",       status: "pending", range: "Est. Q3 2026" },
    { name: "Testing, Commissioning & Handover", status: "pending", range: "Est. Dec 2026" },
  ],
  WHITEFIELD: [
    { name: "Foundation & Substructure",      status: "done", range: "5 May 2023 to 30 Sep 2023" },
    { name: "Superstructure (Columns, Slabs)",status: "done", range: "1 Oct 2023 to 30 Apr 2024" },
    { name: "Façade & Roofing",               status: "done", range: "1 May 2024 to 31 Oct 2024" },
    { name: "MEP & Interior Finishing",       status: "done", range: "1 Nov 2024 to 30 Mar 2026" },
    { name: "Testing, Commissioning & Handover", status: "active", range: "1 Apr 2026 to present", pct: 91 },
  ],
  PUNETECH: [
    { name: "Foundation & Substructure",      status: "active", range: "20 Aug 2025 to present", pct: 23 },
    { name: "Superstructure (Columns, Slabs)",status: "pending", range: "Est. Q2 2026" },
    { name: "Façade & Roofing",               status: "pending", range: "Est. Q4 2026" },
    { name: "MEP & Interior Finishing",       status: "pending", range: "Est. Q1 2027" },
    { name: "Testing, Commissioning & Handover", status: "pending", range: "Est. Mar 2027" },
  ],
};

const SENSOR_PILLS = [
  { name: "Structural",   active: 24, status: "amber" },
  { name: "Environmental",active: 18, status: "green" },
  { name: "Electrical",   active: 22, status: "green" },
  { name: "Plumbing",     active: 14, status: "green" },
  { name: "Safety",       active: 12, status: "amber" },
];

const FLOOR_OPTIONS = ["B2","B1","G","L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12","L13","L14","L15","L16","L17","L18","L19","L20","L21","L22","L23","L24","L25","L26"];

const SENSOR_NODES = [
  { id: "STR-14-NW-01", x: 90,  y: 60,  status: "green",  type: "Structural",   reading: 0.42, threshold: "< 0.55 flagged" },
  { id: "ENV-14-N-03",  x: 200, y: 50,  status: "green",  type: "Environmental",reading: 0.38, threshold: "< 0.50 flagged" },
  { id: "MEP-14-NE-04", x: 510, y: 60,  status: "green",  type: "MEP",          reading: 0.31, threshold: "< 0.55 flagged" },
  { id: "ENV-14-NW-07", x: 80,  y: 200, status: "amber",  type: "Humidity",     reading: 0.72, threshold: "> 0.65 flagged" },
  { id: "STR-14-W-05",  x: 90,  y: 320, status: "green",  type: "Structural",   reading: 0.45, threshold: "< 0.55 flagged" },
  { id: "SAF-14-W-08",  x: 180, y: 350, status: "green",  type: "Safety",       reading: 0.28, threshold: "< 0.50 flagged" },
  { id: "STR-14-SW-02", x: 320, y: 360, status: "red",    type: "Structural",   reading: 0.41, threshold: "> 0.65 flagged" },
  { id: "ENV-14-S-09",  x: 420, y: 350, status: "amber",  type: "Air Quality",  reading: 0.74, threshold: "> 0.65 flagged" },
  { id: "MEP-14-SE-10", x: 520, y: 350, status: "green",  type: "MEP",          reading: 0.36, threshold: "< 0.55 flagged" },
  { id: "STR-14-E-06",  x: 530, y: 200, status: "green",  type: "Structural",   reading: 0.44, threshold: "< 0.55 flagged" },
  { id: "ENV-14-E-11",  x: 510, y: 270, status: "amber",  type: "Temperature",  reading: 0.69, threshold: "> 0.65 flagged" },
  { id: "SAF-14-N-12",  x: 300, y: 50,  status: "green",  type: "Safety",       reading: 0.30, threshold: "< 0.50 flagged" },
];

const INITIAL_FEED = [
  { time: "11:42:18", id: "STR-14-NW-01", type: "Structural",   reading: 0.42, status: "OK" },
  { time: "11:42:15", id: "ENV-14-NW-07", type: "Humidity",     reading: 0.72, status: "WATCH" },
  { time: "11:42:12", id: "MEP-14-NE-04", type: "MEP",          reading: 0.31, status: "OK" },
  { time: "11:42:09", id: "STR-14-SW-02", type: "Structural",   reading: 0.41, status: "ALERT" },
  { time: "11:42:06", id: "ENV-14-E-11",  type: "Temperature",  reading: 0.69, status: "WATCH" },
  { time: "11:42:03", id: "SAF-14-N-12",  type: "Safety",       reading: 0.30, status: "OK" },
  { time: "11:42:00", id: "ENV-14-S-09",  type: "Air Quality",  reading: 0.74, status: "WATCH" },
  { time: "11:41:57", id: "MEP-14-SE-10", type: "MEP",          reading: 0.36, status: "OK" },
];

const INDICATORS = [
  { code: "SHF", name: "Structural Health Factor",       value: 0.82, status: "GOOD",  desc: "Foundation load within design limits. No anomalous strain detected.", trend: [0.81, 0.82, 0.83, 0.81, 0.79, 0.80, 0.82] },
  { code: "ESF", name: "Environmental Stability Factor", value: 0.76, status: "GOOD",  desc: "Temperature and humidity within ASHRAE curing range. Air quality acceptable.", trend: [0.74, 0.75, 0.77, 0.78, 0.75, 0.76, 0.76] },
  { code: "USS", name: "Usage Stress Score",             value: 0.61, status: "WATCH", desc: "Construction equipment load at 61% of structural design capacity. Within limits.", trend: [0.52, 0.55, 0.57, 0.58, 0.59, 0.60, 0.61] },
  { code: "PDP", name: "Predictive Deterioration Penalty", value: 0.89, status: "GOOD", desc: "Material age and cure time on track. No accelerated deterioration predicted.", trend: [0.89, 0.88, 0.89, 0.90, 0.89, 0.89, 0.89] },
  { code: "CI",  name: "Confidence Index",               value: 0.84, status: "GOOD",  desc: "84% of sensor nodes active and reporting. Data reliability high.", trend: [0.84, 0.85, 0.84, 0.82, 0.84, 0.85, 0.84] },
];

const HEALTH_TREND_7D = [
  { day: "Day 1", score: 76.2 },
  { day: "Day 2", score: 76.8 },
  { day: "Day 3", score: 77.4 },
  { day: "Day 4", score: 77.1 },
  { day: "Day 5", score: 77.9 },
  { day: "Day 6", score: 78.1 },
  { day: "Day 7", score: 78.4 },
];

const RTPMV_SENSITIVITY = [
  { score: 60,  rtpmv: 221 },
  { score: 70,  rtpmv: 256 },
  { score: 78,  rtpmv: 287 },
  { score: 85,  rtpmv: 308 },
  { score: 92,  rtpmv: 329 },
  { score: 100, rtpmv: 355 },
];

const MILESTONES = [
  { n: 1,  name: "Foundation pour complete",        phase: "Foundation",    planned: "30 Apr 2024", actual: "28 Apr 2024", status: "done",     verified: "yes"     },
  { n: 2,  name: "Substructure waterproofing",      phase: "Foundation",    planned: "30 Jun 2024", actual: "2 Jul 2024",  status: "done",     verified: "yes"     },
  { n: 3,  name: "Ground floor slab",               phase: "Superstructure",planned: "31 Aug 2024", actual: "29 Aug 2024", status: "done",     verified: "yes"     },
  { n: 4,  name: "L1–L7 superstructure complete",   phase: "Superstructure",planned: "31 Oct 2024", actual: "5 Nov 2024",  status: "done",     verified: "yes"     },
  { n: 5,  name: "L8–L14 superstructure complete",  phase: "Superstructure",planned: "31 Dec 2024", actual: "29 Dec 2024", status: "done",     verified: "yes"     },
  { n: 6,  name: "L15–L26 superstructure complete", phase: "Superstructure",planned: "31 Jan 2025", actual: "31 Jan 2025", status: "done",     verified: "yes"     },
  { n: 7,  name: "Façade glazing — South",          phase: "Façade",        planned: "31 Mar 2025", actual: "5 Apr 2025",  status: "done",     verified: "yes"     },
  { n: 8,  name: "Façade glazing — All elevations", phase: "Façade",        planned: "31 Jul 2025", actual: "29 Jul 2025", status: "done",     verified: "yes"     },
  { n: 9,  name: "Roofing & waterproofing",         phase: "Façade",        planned: "31 Jul 2025", actual: "31 Jul 2025", status: "done",     verified: "yes"     },
  { n: 10, name: "MEP rough-in B2–G",               phase: "MEP",           planned: "31 Dec 2025", actual: "20 Dec 2025", status: "done",     verified: "yes"     },
  { n: 11, name: "MEP Commissioning L1–L14",        phase: "MEP",           planned: "28 Feb 2026", actual: "15 Mar 2026", status: "done",     verified: "yes"     },
  { n: 12, name: "MEP Commissioning L15–L20",       phase: "MEP",           planned: "30 May 2026", actual: "—",            status: "active",   verified: "pending" },
  { n: 13, name: "Fire safety system test",         phase: "Safety",        planned: "30 Jun 2026", actual: "—",            status: "pending",  verified: "pending" },
  { n: 14, name: "Occupancy Permit application",    phase: "Regulatory",    planned: "15 Jul 2026", actual: "—",            status: "overdue",  verified: "pending" },
];

const INITIAL_AUDIT = [
  { n: 15, ts: "2026-05-09 11:32:14", type: "Sensor Batch",         desc: "L14 hourly batch — 1,248 readings",                             hash: "a7c3f81e9b22", prev: "92ed14b03f7a" },
  { n: 14, ts: "2026-05-09 10:32:11", type: "Sensor Batch",         desc: "L14 hourly batch — 1,251 readings",                             hash: "92ed14b03f7a", prev: "5fb820c19ed4" },
  { n: 13, ts: "2026-05-09 09:14:02", type: "Milestone Sign-off",   desc: "Milestone 11 — MEP Commissioning L1–L14 verified",              hash: "5fb820c19ed4", prev: "31aa0fde4c87" },
  { n: 12, ts: "2026-05-09 09:00:00", type: "RTPMV Computation",    desc: "Daily RTPMV refresh — ₹287 Cr",                                  hash: "31aa0fde4c87", prev: "fce47109b2a5" },
  { n: 11, ts: "2026-05-08 18:42:33", type: "Sensor Batch",         desc: "L14 hourly batch — 1,238 readings",                             hash: "fce47109b2a5", prev: "8a0d2c5e1734" },
  { n: 10, ts: "2026-05-08 14:10:21", type: "Compliance Report",    desc: "Q2 compliance report generated for GHMC submission",            hash: "8a0d2c5e1734", prev: "62fb39e7d401" },
  { n: 9,  ts: "2026-05-08 11:20:09", type: "Sensor Batch",         desc: "L14 hourly batch — 1,242 readings",                             hash: "62fb39e7d401", prev: "419e0bcd23a7" },
  { n: 8,  ts: "2026-05-07 16:55:18", type: "RTPMV Computation",    desc: "Daily RTPMV refresh — ₹285 Cr",                                  hash: "419e0bcd23a7", prev: "7b21fa4eb5c0" },
  { n: 7,  ts: "2026-05-07 12:42:00", type: "Value Override",       desc: "Developer override: ₹310 Cr → ₹318 Cr (basis: market comps)",   hash: "7b21fa4eb5c0", prev: "20de8a17f4b8" },
  { n: 6,  ts: "2026-05-07 10:14:55", type: "Sensor Batch",         desc: "L14 hourly batch — 1,260 readings",                             hash: "20de8a17f4b8", prev: "ba9c41d520e3" },
  { n: 5,  ts: "2026-05-06 15:02:48", type: "Milestone Sign-off",   desc: "Milestone 10 — MEP rough-in B2–G verified",                     hash: "ba9c41d520e3", prev: "0e7f59c83a14" },
  { n: 4,  ts: "2026-05-06 09:00:00", type: "RTPMV Computation",    desc: "Daily RTPMV refresh — ₹284 Cr",                                  hash: "0e7f59c83a14", prev: "fa28d6b1e95c" },
  { n: 3,  ts: "2026-05-05 13:35:11", type: "Sensor Batch",         desc: "L14 hourly batch — 1,255 readings",                             hash: "fa28d6b1e95c", prev: "61ea4928c70f" },
  { n: 2,  ts: "2026-05-05 11:18:24", type: "Compliance Report",    desc: "Monthly structural integrity report — auto-generated",          hash: "61ea4928c70f", prev: "07c3b15f2ad9" },
  { n: 1,  ts: "2026-05-05 09:00:00", type: "RTPMV Computation",    desc: "Daily RTPMV refresh — ₹282 Cr",                                  hash: "07c3b15f2ad9", prev: "—"             },
];

const READINESS_DIMS = [
  { name: "Structural Integrity",  pct: 92, status: "ready" },
  { name: "Environmental Quality", pct: 88, status: "ready" },
  { name: "MEP Commissioning",     pct: 61, status: "active" },
  { name: "Safety Compliance",     pct: 55, status: "active" },
  { name: "Data Completeness",     pct: 71, status: "active" },
];

const PORTFOLIO_RTPMV = [
  { name: "Empress Skyline",   value: 334 },
  { name: "Navi Mumbai Log.",  value: 187 },
  { name: "Whitefield Res.",   value: 412 },
  { name: "Pune Tech Camp.",   value: 156 },
];

const PORTFOLIO_HEALTH_30D = (() => {
  const out = [];
  for (let d = 1; d <= 30; d++) {
    out.push({
      day: d,
      EMPRESS:    +(74 + d * 0.28 + Math.sin(d * 0.6) * 1.4).toFixed(1),
      NAVIMUMBAI: +(64 + d * 0.24 + Math.cos(d * 0.5) * 1.6).toFixed(1),
      WHITEFIELD: +(80 + d * 0.27 + Math.sin(d * 0.4) * 1.2).toFixed(1),
      PUNETECH:   +(56 + d * 0.27 + Math.cos(d * 0.7) * 1.8).toFixed(1),
    });
  }
  return out;
})();

const ALL_ALERTS = [
  { project: "Empress Skyline",   floor: "L14", id: "STR-14-SW-02", type: "Structural", reading: 0.41, threshold: "0.65", status: "ALERT",  since: "09:14" },
  { project: "Empress Skyline",   floor: "L14", id: "ENV-14-NW-07", type: "Humidity",   reading: 0.72, threshold: "0.65", status: "WATCH",  since: "11:02" },
  { project: "Navi Mumbai Log.",  floor: "G",   id: "STR-G-SE-04",  type: "Structural", reading: 0.43, threshold: "0.55", status: "WATCH",  since: "08:55" },
  { project: "Navi Mumbai Log.",  floor: "L1",  id: "ENV-L1-N-02",  type: "Temperature",reading: 0.71, threshold: "0.65", status: "WATCH",  since: "10:21" },
  { project: "Pune Tech Camp.",   floor: "B1",  id: "MEP-B1-W-01",  type: "Electrical", reading: 0.39, threshold: "0.55", status: "ALERT",  since: "07:48" },
];

// ── Formatters ─────────────────────────────────────────────────────────────────
const fmt = {
  cr:  (v) => `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 1 })} Cr`,
  pct: (v) => `${v.toFixed(0)}%`,
  num: (v) => v.toLocaleString("en-IN"),
};

const healthColor = (v) => v > 75 ? C.success : v >= 50 ? C.warning : C.danger;
const statusColor = (s) => s === "GOOD" || s === "OK" ? C.success
                         : s === "WATCH" ? C.warning
                         : s === "ALERT" ? C.danger
                         : C.textDim;
const dotColor = (s) => s === "green" ? C.success : s === "amber" ? C.warning : C.danger;

const randHex = (n) => Array.from({ length: n }, () => "0123456789abcdef"[Math.floor(Math.random() * 16)]).join("");

// ── Sub-components ─────────────────────────────────────────────────────────────
function Card({ children, style = {} }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
      padding: "16px 18px", ...style,
    }}>{children}</div>
  );
}

function SectionHeader({ children, style = {} }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 700, color: C.gold, letterSpacing: 1.5,
      textTransform: "uppercase", marginBottom: 12, ...style,
    }}>{children}</div>
  );
}

function ProgressBar({ value, max = 100, color = C.gold, height = 6 }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ background: C.surfaceAlt, borderRadius: 4, height, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.6s ease" }} />
    </div>
  );
}

function HealthBadge({ value, size = "sm" }) {
  const col = healthColor(value);
  const fs = size === "lg" ? 14 : 10;
  const pad = size === "lg" ? "4px 10px" : "2px 7px";
  return (
    <span style={{
      background: `${col}22`, color: col, border: `1px solid ${col}55`,
      padding: pad, borderRadius: 12, fontSize: fs, fontWeight: 700,
    }}>{value.toFixed(1)}</span>
  );
}

function Sparkline({ data, color = C.gold, w = 110, h = 28 }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}

function Gauge({ value, max = 100 }) {
  const cx = 130, cy = 130, r = 100;
  const pct = Math.max(0, Math.min(1, value / max));
  const angle = pct * 180;
  const rad = (angle * Math.PI) / 180;
  const nx = cx - r * Math.cos(rad);
  const ny = cy - r * Math.sin(rad);
  const arc = (startA, endA) => {
    const s = (startA * Math.PI) / 180;
    const e = (endA * Math.PI) / 180;
    return `M ${cx - r * Math.cos(s)} ${cy - r * Math.sin(s)} A ${r} ${r} 0 0 1 ${cx - r * Math.cos(e)} ${cy - r * Math.sin(e)}`;
  };
  return (
    <svg width={260} height={160} viewBox="0 0 260 160" style={{ display: "block", margin: "0 auto" }}>
      <path d={arc(0, 60)}   stroke={C.danger}  strokeWidth={14} fill="none" />
      <path d={arc(60, 120)} stroke={C.warning} strokeWidth={14} fill="none" />
      <path d={arc(120, 180)} stroke={C.success} strokeWidth={14} fill="none" />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={C.text} strokeWidth={3} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={6} fill={C.gold} />
      <text x={cx} y={cy + 30} textAnchor="middle" fill={C.text} fontSize={28} fontWeight={700}>{value.toFixed(1)}</text>
      <text x={cx} y={cy + 48} textAnchor="middle" fill={C.textDim} fontSize={11}>/ {max}</text>
    </svg>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding: "10px 18px", cursor: "pointer", fontSize: 12, fontWeight: 600,
      color: active ? C.gold : C.textDim,
      background: "transparent",
      border: "none",
      boxShadow: active ? `inset 0 -2px 0 ${C.gold}` : "none",
      outline: "none", letterSpacing: 0.4,
    }}>{children}</button>
  );
}

function Toast({ msg, onClose }) {
  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(onClose, 4500);
    return () => clearTimeout(t);
  }, [msg, onClose]);
  if (!msg) return null;
  return (
    <div style={{
      position: "fixed", bottom: 24, right: 24, zIndex: 1000,
      background: C.surface, border: `1px solid ${C.gold}`,
      borderLeft: `4px solid ${C.gold}`, borderRadius: 6,
      padding: "12px 18px", color: C.text, fontSize: 12,
      maxWidth: 420, boxShadow: "0 6px 24px rgba(0,0,0,0.5)",
    }}>{msg}</div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function PropDeveloperDashboard() {
  const [activeProjectId, setActiveProjectId] = useState(PROJECTS[0].id);
  const [activeTab, setActiveTab] = useState("overview");
  const [activeFloor, setActiveFloor] = useState("L14");
  const [twinToggles, setTwinToggles] = useState({ Structural: true, Environmental: true, Safety: true, MEP: true });
  const [feed, setFeed] = useState(INITIAL_FEED);
  const [auditLog, setAuditLog] = useState(INITIAL_AUDIT);
  const [overrideValue, setOverrideValue] = useState(310);
  const [overrideInput, setOverrideInput] = useState("310");
  const [overrideReason, setOverrideReason] = useState("");
  const [lastOverride, setLastOverride] = useState(null);
  const [hoverSensor, setHoverSensor] = useState(null);
  const [now, setNow] = useState(new Date());
  const [toast, setToast] = useState("");

  const activeProject = useMemo(
    () => PROJECTS.find(p => p.id === activeProjectId) || PROJECTS[0],
    [activeProjectId]
  );

  // Live clock — every 1s
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  // Live sensor stream — every 3s, prepend 1, drop oldest
  useEffect(() => {
    const id = setInterval(() => {
      setFeed(prev => {
        const types = ["Structural", "Temperature", "Humidity", "Vibration", "Air Quality"];
        const ids   = ["STR-14-NW-01","ENV-14-NW-07","MEP-14-NE-04","STR-14-SW-02","ENV-14-E-11","SAF-14-N-12","ENV-14-S-09","MEP-14-SE-10","STR-14-E-06","STR-14-W-05"];
        const type = types[Math.floor(Math.random() * types.length)];
        const id   = ids[Math.floor(Math.random() * ids.length)];
        const reading = +(0.25 + Math.random() * 0.55).toFixed(2);
        const status = reading > 0.70 ? "WATCH" : reading > 0.78 ? "ALERT" : "OK";
        const d = new Date();
        const time = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
        return [{ time, id, type, reading, status }, ...prev].slice(0, 8);
      });
    }, 3000);
    return () => clearInterval(id);
  }, []);

  // Override handler
  const applyOverride = useCallback(() => {
    const num = parseFloat(overrideInput);
    if (isNaN(num) || num <= 0) {
      setToast("Invalid override value. Enter a positive number in ₹ Cr.");
      return;
    }
    setOverrideValue(num);
    const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
    setLastOverride({ value: num, reason: overrideReason || "(no reason provided)", ts: stamp });
    setAuditLog(prev => [{
      n: prev.length + 1,
      ts: stamp,
      type: "Value Override",
      desc: `Developer override: ₹${num} Cr — ${overrideReason || "no reason"}`,
      hash: randHex(12),
      prev: prev[0]?.hash || "—",
    }, ...prev]);
    setToast(`Override applied: ₹${num} Cr. Logged to Audit Chain.`);
  }, [overrideInput, overrideReason]);

  const generateComplianceReport = useCallback(() => {
    const hash = randHex(8);
    const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
    setAuditLog(prev => [{
      n: prev.length + 1,
      ts: stamp,
      type: "Compliance Report",
      desc: `Cryptographic compliance report generated — milestones audit`,
      hash: randHex(12),
      prev: prev[0]?.hash || "—",
    }, ...prev]);
    setToast(`Cryptographic compliance report generated. Hash: ${hash}. Added to Audit Log.`);
  }, []);

  const exportAuditReport = useCallback(() => {
    setToast("Audit report prepared. In live version, this downloads a signed PDF.");
  }, []);

  // Derived RTPMV after override
  const rtpmvAfterOverride = useMemo(() => {
    const h = activeProject.buildHealth / 100;
    const land = activeProject.landValue;
    const struct = activeProject.structureValue;
    const base = overrideValue;
    const scale = base / activeProject.govtValue;
    return +(land * scale + struct * scale * h).toFixed(1);
  }, [overrideValue, activeProject]);

  // ── Layout styles ───────────────────────────────────────────────────────────
  const S = {
    app: {
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      background: C.bg, color: C.text, minHeight: "100vh",
      padding: "20px 24px", fontSize: 13,
    },
    headerLogo: { fontSize: 22, fontWeight: 700, color: C.gold, letterSpacing: 1.5 },
    headerSub:  { fontSize: 11, color: C.textDim, marginTop: 3, letterSpacing: 0.4 },
    kpiRow:     { display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 14, marginTop: 18 },
    kpiCard:    { background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 18px" },
    kpiLabel:   { fontSize: 10, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2, fontWeight: 600 },
    kpiValue:   { fontSize: 22, fontWeight: 700, color: C.text, marginTop: 6 },
    subBar:     {
      display: "flex", justifyContent: "space-between", alignItems: "center",
      marginTop: 14, padding: "8px 14px",
      background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6,
      fontSize: 11, color: C.textDim, letterSpacing: 0.3,
    },
    demoBadge:  {
      background: C.gold, color: C.bg, padding: "3px 10px",
      borderRadius: 4, fontSize: 10, fontWeight: 800, letterSpacing: 1.5,
    },
    selectorRow:{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginTop: 18 },
    selectorCard: (active) => ({
      background: active ? C.surfaceAlt : C.surface,
      border: `1px solid ${active ? C.gold : C.border}`,
      borderLeft: `3px solid ${active ? C.gold : C.border}`,
      borderRadius: 8, padding: "14px 16px", cursor: "pointer",
      transition: "all 0.2s",
    }),
    tabsBar:    { display: "flex", gap: 0, borderBottom: `1px solid ${C.border}`, marginTop: 22, marginBottom: 18, flexWrap: "wrap" },
    grid2:      { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 },
    grid3:      { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 },
    grid5:      { display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 },
    twinSplit:  { display: "grid", gridTemplateColumns: "60% 40%", gap: 14 },
  };

  const TABS = [
    { id: "overview",  label: "Project Overview" },
    { id: "twin",      label: "Digital Twin Viewer" },
    { id: "indicators",label: "Build Health Indicators" },
    { id: "rtpmv",     label: "RTPMV Projection" },
    { id: "milestone", label: "Milestone Tracker" },
    { id: "audit",     label: "Audit Log" },
    { id: "handover",  label: "Handover Readiness" },
    { id: "portfolio", label: "Portfolio Analytics" },
  ];

  const clockStr = now.toLocaleString("en-IN", {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={S.app}>
      {/* Animation keyframes for pulsing red sensor */}
      <style>{`
        @keyframes twinval-pulse {
          0%   { r: 7;  opacity: 1;   }
          50%  { r: 13; opacity: 0.4; }
          100% { r: 7;  opacity: 1;   }
        }
        @keyframes twinval-pulse-ring {
          0%   { r: 7;  opacity: 0.7; stroke-width: 2; }
          100% { r: 22; opacity: 0;   stroke-width: 1; }
        }
        .twinval-sensor-red {
          animation: twinval-pulse 1.6s ease-in-out infinite;
          transform-origin: center;
        }
        .twinval-sensor-ring {
          animation: twinval-pulse-ring 1.6s ease-out infinite;
          fill: none;
        }
      `}</style>

      {/* ── Persistent Header KPI Bar ──────────────────────────────────────── */}
      <div style={{ borderBottom: `1px solid ${C.border}`, paddingBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={S.headerLogo}>TwinVal PropDeveloper</div>
            <div style={S.headerSub}>Construction-Phase Intelligence · Real-Time Build Health & RTPMV</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 11, color: C.textDim }}>Patent: India App. No. 202641030498</div>
            <div style={{ fontSize: 10, color: C.textMuted, marginTop: 2 }}>Muqtadar Musawir Quraishi</div>
          </div>
        </div>

        <div style={S.kpiRow}>
          <div style={S.kpiCard}>
            <div style={S.kpiLabel}>Active Projects</div>
            <div style={S.kpiValue}>4</div>
          </div>
          <div style={S.kpiCard}>
            <div style={S.kpiLabel}>Portfolio Completion Value</div>
            <div style={{ ...S.kpiValue, color: C.gold }}>₹1,847 Cr</div>
          </div>
          <div style={S.kpiCard}>
            <div style={S.kpiLabel}>Avg. Build Health Score</div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 6 }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: C.text }}>78.4 <span style={{ fontSize: 12, color: C.textDim }}>/ 100</span></div>
              <HealthBadge value={78.4} />
            </div>
          </div>
          <div style={S.kpiCard}>
            <div style={S.kpiLabel}>Properties Ready for Handover</div>
            <div style={S.kpiValue}>1</div>
          </div>
          <div style={S.kpiCard}>
            <div style={S.kpiLabel}>Est. RTPMV at Completion</div>
            <div style={{ ...S.kpiValue, color: C.gold }}>₹2,104 Cr</div>
          </div>
        </div>

        {/* Sub-bar */}
        <div style={S.subBar}>
          <div>
            <span style={{ color: C.gold, fontWeight: 600 }}>TwinVal PropDeveloper</span>
            <span style={{ color: C.textMuted, margin: "0 8px" }}>|</span>
            <span>Simulated Demo Data</span>
            <span style={{ color: C.textMuted, margin: "0 8px" }}>|</span>
            <span>Last updated: {clockStr}</span>
          </div>
          <span style={S.demoBadge}>DEMO</span>
        </div>
      </div>

      {/* ── Project Selector Row ──────────────────────────────────────────── */}
      <div style={S.selectorRow}>
        {PROJECTS.map(p => {
          const active = p.id === activeProjectId;
          return (
            <div key={p.id} style={S.selectorCard(active)} onClick={() => setActiveProjectId(p.id)}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: active ? C.gold : C.text, lineHeight: 1.3 }}>{p.name}</div>
                  <div style={{ fontSize: 11, color: C.textDim, marginTop: 3 }}>{p.city}</div>
                </div>
                <HealthBadge value={p.buildHealth} />
              </div>
              <div style={{
                display: "inline-block", fontSize: 10, fontWeight: 600,
                color: C.textDim, background: C.surfaceAlt, padding: "2px 8px",
                borderRadius: 3, letterSpacing: 0.3, marginBottom: 8,
              }}>{p.type}</div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: C.textDim }}>Progress</span>
                <span style={{ fontSize: 10, color: C.text, fontWeight: 600 }}>{p.progress}%</span>
              </div>
              <ProgressBar value={p.progress} color={C.gold} height={5} />
            </div>
          );
        })}
      </div>

      {/* ── Tabs ────────────────────────────────────────────────────────── */}
      <div style={S.tabsBar}>
        {TABS.map(t => (
          <TabButton key={t.id} active={activeTab === t.id} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </TabButton>
        ))}
      </div>

      {/* ═══ TAB 1: PROJECT OVERVIEW ════════════════════════════════════ */}
      {activeTab === "overview" && (
        <div>
          <div style={S.grid2}>
            {/* LEFT — Project Summary */}
            <Card>
              <SectionHeader>Project Summary</SectionHeader>
              {[
                ["Property",                activeProject.name],
                ["Location",                activeProject.fullLocation],
                ["Type",                    activeProject.typeLong],
                ["Total Built-Up Area",     activeProject.builtUpArea],
                ["No. of Floors",           activeProject.floors],
                ["Developer",               activeProject.developer],
                ["Structural Engineer",     activeProject.structuralEng],
                ["Govt. Assessment Value",  `${fmt.cr(activeProject.govtValue)}  (${activeProject.govtRef})`],
                ["Construction Start",      activeProject.startDate],
                ["Est. Completion",         activeProject.completionDate],
                ["Current Phase",           activeProject.currentPhase],
              ].map(([label, val]) => (
                <div key={label} style={{
                  display: "grid", gridTemplateColumns: "180px 1fr",
                  gap: 12, padding: "8px 0", borderBottom: `1px solid ${C.border}`,
                }}>
                  <span style={{ fontSize: 11, color: C.textDim }}>{label}</span>
                  <span style={{ fontSize: 12, color: C.text, fontWeight: 500 }}>{val}</span>
                </div>
              ))}
            </Card>

            {/* RIGHT — Phase Timeline */}
            <Card>
              <SectionHeader>Phase Progress Timeline</SectionHeader>
              {(PHASES_BY_PROJECT[activeProject.id] || []).map((ph, i, arr) => {
                const icon = ph.status === "done" ? "✅" : ph.status === "active" ? "🔄" : "⏳";
                const col  = ph.status === "done" ? C.success : ph.status === "active" ? C.gold : C.textDim;
                const last = i === arr.length - 1;
                return (
                  <div key={ph.name} style={{ display: "flex", gap: 14, paddingBottom: 14, position: "relative" }}>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <div style={{
                        width: 28, height: 28, borderRadius: "50%",
                        background: C.surfaceAlt, border: `2px solid ${col}`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 13,
                      }}>{icon}</div>
                      {!last && <div style={{ width: 2, flex: 1, background: C.border, marginTop: 4 }} />}
                    </div>
                    <div style={{ flex: 1, paddingTop: 2 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: col }}>{ph.name}</div>
                      <div style={{ fontSize: 10, color: C.textDim, marginTop: 2 }}>{ph.range}</div>
                      {ph.status === "active" && (
                        <div style={{ marginTop: 6 }}>
                          <ProgressBar value={ph.pct || activeProject.progress} color={C.gold} height={4} />
                          <div style={{ fontSize: 10, color: C.gold, marginTop: 3 }}>{ph.pct || activeProject.progress}% complete</div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </Card>
          </div>

          {/* Sensor pills */}
          <div style={{ marginTop: 14 }}>
            <Card>
              <SectionHeader>Sensor Status Summary</SectionHeader>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {SENSOR_PILLS.map(p => {
                  const col = dotColor(p.status);
                  return (
                    <div key={p.name} style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "8px 14px", borderRadius: 20,
                      background: `${col}15`, border: `1px solid ${col}55`,
                    }}>
                      <span style={{
                        width: 10, height: 10, borderRadius: "50%",
                        background: col, display: "inline-block",
                      }} />
                      <span style={{ fontSize: 12, fontWeight: 600, color: C.text }}>{p.name}</span>
                      <span style={{ fontSize: 11, color: C.textDim }}>· {p.active} active</span>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ═══ TAB 2: DIGITAL TWIN VIEWER ════════════════════════════════════ */}
      {activeTab === "twin" && (
        <div>
          <div style={S.twinSplit}>
            {/* LEFT — Floor plan */}
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <SectionHeader style={{ marginBottom: 0 }}>Digital Twin Viewer — {activeProject.name}</SectionHeader>
              </div>

              {/* Floor selector */}
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 12, padding: "8px 0", borderBottom: `1px solid ${C.border}` }}>
                <span style={{ fontSize: 10, color: C.textDim, marginRight: 8, alignSelf: "center" }}>FLOOR:</span>
                {FLOOR_OPTIONS.map(f => (
                  <button key={f} onClick={() => setActiveFloor(f)} style={{
                    padding: "3px 8px", fontSize: 10, fontWeight: 600,
                    background: f === activeFloor ? C.gold : "transparent",
                    color:      f === activeFloor ? C.bg   : C.textDim,
                    border: `1px solid ${f === activeFloor ? C.gold : C.border}`,
                    borderRadius: 3, cursor: "pointer",
                  }}>{f}</button>
                ))}
              </div>

              {/* Filter toggles */}
              <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
                {["Structural", "Environmental", "Safety", "MEP"].map(t => {
                  const on = twinToggles[t];
                  return (
                    <button key={t} onClick={() => setTwinToggles(s => ({ ...s, [t]: !s[t] }))} style={{
                      padding: "5px 12px", fontSize: 10, fontWeight: 600,
                      background: on ? `${C.gold}25` : "transparent",
                      color:      on ? C.gold : C.textDim,
                      border: `1px solid ${on ? C.gold : C.border}`,
                      borderRadius: 14, cursor: "pointer", letterSpacing: 0.4,
                    }}>{t}</button>
                  );
                })}
              </div>

              {/* SVG floor plan */}
              <div style={{ position: "relative", background: C.surfaceAlt, borderRadius: 6, padding: 8 }}>
                <svg viewBox="0 0 600 420" width="100%" style={{ display: "block" }}>
                  {/* Outer floor plate */}
                  <rect x="20" y="20" width="560" height="380" fill="#0f0f0f" stroke={C.border} strokeWidth="2" />

                  {/* Reception / Lobby (top centre) */}
                  <rect x="240" y="20" width="120" height="60" fill="#1a1a14" stroke={C.goldDim} strokeWidth="1" />
                  <text x="300" y="56" textAnchor="middle" fill={C.textDim} fontSize="9" letterSpacing="1">RECEPTION / LOBBY</text>

                  {/* Central core */}
                  <rect x="240" y="130" width="120" height="160" fill="#181818" stroke={C.border} strokeWidth="1.5" />
                  <text x="300" y="200" textAnchor="middle" fill={C.textDim} fontSize="10" letterSpacing="1.2">CORE</text>
                  <text x="300" y="216" textAnchor="middle" fill={C.textMuted} fontSize="8">Lifts · Stairs · Toilets</text>
                  <line x1="240" y1="170" x2="360" y2="170" stroke={C.border} strokeWidth="1" />
                  <line x1="240" y1="220" x2="360" y2="220" stroke={C.border} strokeWidth="1" />
                  <line x1="240" y1="260" x2="360" y2="260" stroke={C.border} strokeWidth="1" />

                  {/* Wing zones — North */}
                  <rect x="20"  y="20"  width="220" height="110" fill="none" stroke={C.border} strokeWidth="0.5" strokeDasharray="3 3" />
                  <text x="130" y="80"  textAnchor="middle" fill={C.textMuted} fontSize="10" letterSpacing="1">NORTH WING</text>
                  {/* East */}
                  <rect x="360" y="20"  width="220" height="270" fill="none" stroke={C.border} strokeWidth="0.5" strokeDasharray="3 3" />
                  <text x="470" y="160" textAnchor="middle" fill={C.textMuted} fontSize="10" letterSpacing="1">EAST WING</text>
                  {/* West */}
                  <rect x="20"  y="130" width="220" height="270" fill="none" stroke={C.border} strokeWidth="0.5" strokeDasharray="3 3" />
                  <text x="130" y="280" textAnchor="middle" fill={C.textMuted} fontSize="10" letterSpacing="1">WEST WING</text>
                  {/* South */}
                  <rect x="240" y="290" width="340" height="110" fill="none" stroke={C.border} strokeWidth="0.5" strokeDasharray="3 3" />
                  <text x="410" y="370" textAnchor="middle" fill={C.textMuted} fontSize="10" letterSpacing="1">SOUTH WING</text>

                  {/* Sensor nodes */}
                  {SENSOR_NODES.map(n => {
                    const col = dotColor(n.status);
                    if (n.status === "red") {
                      return (
                        <g key={n.id} onMouseEnter={() => setHoverSensor(n)} onMouseLeave={() => setHoverSensor(null)} style={{ cursor: "pointer" }}>
                          <circle cx={n.x} cy={n.y} r={7} fill={col} className="twinval-sensor-red" />
                          <circle cx={n.x} cy={n.y} r={7} stroke={col} className="twinval-sensor-ring" />
                        </g>
                      );
                    }
                    return (
                      <circle key={n.id} cx={n.x} cy={n.y} r={6} fill={col}
                        stroke={C.bg} strokeWidth="1.5"
                        onMouseEnter={() => setHoverSensor(n)}
                        onMouseLeave={() => setHoverSensor(null)}
                        style={{ cursor: "pointer" }} />
                    );
                  })}
                </svg>

                {/* Hover tooltip */}
                {hoverSensor && (
                  <div style={{
                    position: "absolute", top: 12, right: 12,
                    background: C.bg, border: `1px solid ${dotColor(hoverSensor.status)}`,
                    borderRadius: 6, padding: "10px 14px", minWidth: 200,
                    fontSize: 11, color: C.text, boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
                  }}>
                    <div style={{ fontWeight: 700, color: dotColor(hoverSensor.status), marginBottom: 6 }}>
                      Sensor {hoverSensor.id}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 12px" }}>
                      <span style={{ color: C.textDim }}>Type:</span>
                      <span>{hoverSensor.type}</span>
                      <span style={{ color: C.textDim }}>Reading:</span>
                      <span style={{ fontWeight: 600 }}>{hoverSensor.reading.toFixed(2)}</span>
                      <span style={{ color: C.textDim }}>Threshold:</span>
                      <span>{hoverSensor.threshold}</span>
                      <span style={{ color: C.textDim }}>Status:</span>
                      <span style={{ color: dotColor(hoverSensor.status), fontWeight: 700 }}>
                        {hoverSensor.status === "red" ? "ALERT" : hoverSensor.status === "amber" ? "WATCH" : "OK"}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <div style={{ marginTop: 10, fontSize: 10, color: C.textMuted, fontStyle: "italic" }}>
                Demo mode: static floor plan. Live version accepts AutoCAD DWG / Revit BIM file upload. Sensor nodes auto-mapped from file coordinates.
              </div>
            </Card>

            {/* RIGHT — Live sensor feed */}
            <Card>
              <SectionHeader>Live Sensor Stream — {activeFloor}</SectionHeader>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                <thead>
                  <tr style={{ color: C.textDim, textAlign: "left" }}>
                    <th style={{ padding: "6px 4px", borderBottom: `1px solid ${C.border}` }}>Time</th>
                    <th style={{ padding: "6px 4px", borderBottom: `1px solid ${C.border}` }}>Sensor ID</th>
                    <th style={{ padding: "6px 4px", borderBottom: `1px solid ${C.border}` }}>Type</th>
                    <th style={{ padding: "6px 4px", textAlign: "right", borderBottom: `1px solid ${C.border}` }}>Reading</th>
                    <th style={{ padding: "6px 4px", textAlign: "right", borderBottom: `1px solid ${C.border}` }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {feed.map((row, i) => (
                    <tr key={`${row.time}-${i}`} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ padding: "5px 4px", color: C.textDim, fontFamily: "monospace" }}>{row.time}</td>
                      <td style={{ padding: "5px 4px", color: C.text, fontFamily: "monospace" }}>{row.id}</td>
                      <td style={{ padding: "5px 4px", color: C.textDim }}>{row.type}</td>
                      <td style={{ padding: "5px 4px", textAlign: "right", color: C.text, fontWeight: 600, fontFamily: "monospace" }}>
                        {row.reading.toFixed(2)}
                      </td>
                      <td style={{ padding: "5px 4px", textAlign: "right", color: statusColor(row.status), fontWeight: 700 }}>
                        {row.status}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div style={{ marginTop: 16 }}>
                <SectionHeader>Active Alerts</SectionHeader>
                <div style={{
                  background: `${C.danger}10`, border: `1px solid ${C.danger}55`,
                  borderRadius: 6, padding: "10px 12px", marginBottom: 8, fontSize: 11,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.danger, fontWeight: 700 }}>
                    🔴 STR-14-SW-02
                  </div>
                  <div style={{ color: C.text, marginTop: 4 }}>
                    Structural · Reading <strong>0.41</strong> · Below threshold 0.65 · Since <span style={{ color: C.textDim }}>09:14</span>
                  </div>
                </div>
                <div style={{
                  background: `${C.warning}10`, border: `1px solid ${C.warning}55`,
                  borderRadius: 6, padding: "10px 12px", fontSize: 11,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.warning, fontWeight: 700 }}>
                    🟡 ENV-14-NW-07
                  </div>
                  <div style={{ color: C.text, marginTop: 4 }}>
                    Humidity · Reading <strong>0.72</strong> · Approaching threshold · Since <span style={{ color: C.textDim }}>11:02</span>
                  </div>
                </div>
              </div>

              <div style={{
                marginTop: 14, padding: "8px 0", borderTop: `1px solid ${C.border}`,
                fontSize: 10, color: C.textMuted, textAlign: "center",
              }}>
                Digital Twin last synced: {clockStr}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ═══ TAB 3: BUILD HEALTH INDICATORS ════════════════════════════════ */}
      {activeTab === "indicators" && (
        <div>
          <div style={S.grid5}>
            {INDICATORS.map(ind => {
              const col = statusColor(ind.status);
              return (
                <Card key={ind.code} style={{ padding: "14px 14px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.gold, letterSpacing: 1 }}>{ind.code}</div>
                    <span style={{
                      fontSize: 9, fontWeight: 700, color: col,
                      background: `${col}20`, padding: "2px 8px", borderRadius: 10, letterSpacing: 1,
                    }}>{ind.status}</span>
                  </div>
                  <div style={{ fontSize: 10, color: C.textDim, marginBottom: 8, minHeight: 28 }}>{ind.name}</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: C.text, marginBottom: 8 }}>{ind.value.toFixed(2)}</div>
                  <Sparkline data={ind.trend} color={col} w={140} h={24} />
                  <div style={{ fontSize: 10, color: C.textMuted, marginTop: 10, lineHeight: 1.4 }}>{ind.desc}</div>
                </Card>
              );
            })}
          </div>

          {/* Composite formula + gauge */}
          <div style={{ ...S.grid2, marginTop: 14 }}>
            <Card>
              <SectionHeader>Composite Build Health Score</SectionHeader>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: C.textDim, lineHeight: 1.9 }}>
                <div>Build Health Score = SHF × ESF × (1 − USS) × PDP × CI</div>
                <div style={{ color: C.text }}>= 0.82 × 0.76 × 0.39 × 0.89 × 0.84</div>
                <div style={{ color: C.gold, fontWeight: 700 }}>= 0.182 (raw) → Scaled to 78.4 / 100</div>
              </div>
              <div style={{ fontSize: 10, color: C.textMuted, marginTop: 12, lineHeight: 1.5, fontStyle: "italic" }}>
                Note: The raw composite (0–1) is non-linearly scaled to a 0–100 reading using a logarithmic transform that emphasises the upper performance band. This keeps scores comparable across building types and life-cycle stages.
              </div>
            </Card>
            <Card>
              <SectionHeader>Build Health Gauge</SectionHeader>
              <Gauge value={78.4} />
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0 30px", marginTop: -8, fontSize: 10, color: C.textMuted }}>
                <span>0 (Critical)</span>
                <span>50 (Watch)</span>
                <span>100 (Optimal)</span>
              </div>
            </Card>
          </div>

          {/* 7-day trend */}
          <Card style={{ marginTop: 14 }}>
            <SectionHeader>Build Health Score — Last 7 Days</SectionHeader>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={HEALTH_TREND_7D} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="healthGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"  stopColor={C.gold} stopOpacity={0.6} />
                    <stop offset="100%" stopColor={C.gold} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: C.textDim }} />
                <YAxis domain={[70, 85]} tick={{ fontSize: 11, fill: C.textDim }} />
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.gold}`, fontSize: 11, color: C.text }} />
                <Area type="monotone" dataKey="score" stroke={C.gold} strokeWidth={2} fill="url(#healthGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ═══ TAB 4: RTPMV PROJECTION ════════════════════════════════════ */}
      {activeTab === "rtpmv" && (
        <div>
          <div style={S.grid3}>
            <Card>
              <div style={{ fontSize: 11, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>Baseline Govt. Value</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.text, marginTop: 8 }}>{fmt.cr(activeProject.govtValue)}</div>
              <div style={{ fontSize: 10, color: C.textMuted, marginTop: 4 }}>{activeProject.govtRef}</div>
            </Card>
            <Card>
              <div style={{ fontSize: 11, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>RTPMV at Current Build Health</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.gold, marginTop: 8 }}>{fmt.cr(activeProject.rtpmvCurrent)}</div>
              <div style={{ fontSize: 10, color: C.textMuted, marginTop: 4 }}>Build Health: {activeProject.buildHealth} / 100</div>
            </Card>
            <Card>
              <div style={{ fontSize: 11, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>Projected RTPMV at Completion</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.success, marginTop: 8 }}>{fmt.cr(activeProject.rtpmvCompletion)}</div>
              <div style={{ fontSize: 10, color: C.textMuted, marginTop: 4 }}>Assuming Build Health maintained ≥ 88</div>
            </Card>
          </div>

          <div style={{ ...S.grid2, marginTop: 14 }}>
            <Card>
              <SectionHeader>RTPMV Breakdown</SectionHeader>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: C.textDim, lineHeight: 2 }}>
                <div>RTPMV = Land Value + (Structure Value × Health Factor)</div>
                <div style={{ color: C.text }}>= {fmt.cr(activeProject.landValue)} + ({fmt.cr(activeProject.structureValue)} × {(activeProject.buildHealth / 100).toFixed(3)})</div>
                <div style={{ color: C.text }}>= {fmt.cr(activeProject.landValue)} + {fmt.cr(+(activeProject.structureValue * activeProject.buildHealth / 100).toFixed(1))}</div>
                <div style={{ color: C.gold, fontWeight: 700 }}>= {fmt.cr(+(activeProject.landValue + activeProject.structureValue * activeProject.buildHealth / 100).toFixed(1))} (current)</div>
              </div>
              <div style={{
                marginTop: 14, padding: "10px 12px",
                background: `${C.success}10`, border: `1px solid ${C.success}40`, borderRadius: 6,
                fontSize: 11, color: C.text,
              }}>
                At completion, assuming Build Health Score reaches <strong style={{ color: C.success }}>88</strong>, projected RTPMV = <strong style={{ color: C.success }}>{fmt.cr(activeProject.rtpmvCompletion)}</strong>
              </div>
            </Card>

            <Card>
              <SectionHeader>Developer Value Adjustment</SectionHeader>
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 11, color: C.textDim, display: "block", marginBottom: 4 }}>
                  Override Baseline Value (₹ Cr)
                </label>
                <input type="number" value={overrideInput}
                  onChange={(e) => setOverrideInput(e.target.value)}
                  style={{
                    width: "100%", padding: "8px 10px", fontSize: 12,
                    background: C.surfaceAlt, border: `1px solid ${C.border}`,
                    borderRadius: 4, color: C.text, fontFamily: "monospace",
                  }} />
              </div>
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 11, color: C.textDim, display: "block", marginBottom: 4 }}>
                  Adjustment Reason
                </label>
                <input type="text" value={overrideReason}
                  onChange={(e) => setOverrideReason(e.target.value)}
                  placeholder="e.g. recent comparable sale, upgraded specifications"
                  style={{
                    width: "100%", padding: "8px 10px", fontSize: 12,
                    background: C.surfaceAlt, border: `1px solid ${C.border}`,
                    borderRadius: 4, color: C.text,
                  }} />
              </div>
              <button onClick={applyOverride} style={{
                background: C.gold, color: C.bg, border: "none", borderRadius: 4,
                padding: "8px 18px", fontSize: 12, fontWeight: 700, cursor: "pointer",
                letterSpacing: 0.5,
              }}>Apply Override</button>
              <div style={{ marginTop: 14, fontSize: 11, color: C.textDim }}>
                {lastOverride ? (
                  <>
                    Last override: <strong style={{ color: C.gold }}>₹{lastOverride.value} Cr</strong> at {lastOverride.ts}
                    <div style={{ marginTop: 2, fontSize: 10, fontStyle: "italic", color: C.textMuted }}>Reason: {lastOverride.reason}</div>
                    <div style={{ marginTop: 6, fontSize: 11, color: C.text }}>
                      Recomputed RTPMV: <strong style={{ color: C.gold }}>{fmt.cr(rtpmvAfterOverride)}</strong>
                    </div>
                  </>
                ) : (
                  <span>No override applied — using Govt. Assessment Value ({fmt.cr(activeProject.govtValue)})</span>
                )}
              </div>
            </Card>
          </div>

          <Card style={{ marginTop: 14 }}>
            <SectionHeader>RTPMV Sensitivity to Build Health Score (₹ Cr)</SectionHeader>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={RTPMV_SENSITIVITY}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="score" tick={{ fontSize: 11, fill: C.textDim }} label={{ value: "Build Health Score", position: "insideBottom", offset: -4, fill: C.textDim, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11, fill: C.textDim }} tickFormatter={v => `₹${v}`} />
                <Tooltip formatter={(v) => [`₹${v} Cr`]} contentStyle={{ background: C.surface, border: `1px solid ${C.gold}`, fontSize: 11, color: C.text }} />
                <Bar dataKey="rtpmv" radius={[4, 4, 0, 0]}>
                  {RTPMV_SENSITIVITY.map((d, i) => (
                    <Cell key={i} fill={d.score === 78 ? C.gold : `${C.gold}88`} />
                  ))}
                </Bar>
                <ReferenceLine y={activeProject.rtpmvCurrent} stroke={C.success} strokeDasharray="4 4" label={{ value: `Current ₹${activeProject.rtpmvCurrent} Cr`, fill: C.success, fontSize: 10, position: "right" }} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ═══ TAB 5: MILESTONE TRACKER ════════════════════════════════════ */}
      {activeTab === "milestone" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 14 }}>
            <Card>
              <div style={{ fontSize: 10, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>Milestones Completed</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: C.text, marginTop: 6 }}>14 <span style={{ fontSize: 14, color: C.textDim }}>/ 22</span></div>
            </Card>
            <Card>
              <div style={{ fontSize: 10, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>Overdue</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: C.danger, marginTop: 6 }}>1</div>
            </Card>
            <Card>
              <div style={{ fontSize: 10, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>Sensor-Verified Sign-offs</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: C.success, marginTop: 6 }}>11</div>
            </Card>
            <Card>
              <div style={{ fontSize: 10, color: C.textDim, textTransform: "uppercase", letterSpacing: 1.2 }}>Next Milestone</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: C.gold, marginTop: 6 }}>MEP Commissioning — L15 to L20</div>
              <div style={{ fontSize: 10, color: C.textMuted, marginTop: 3 }}>Due: 30 May 2026</div>
            </Card>
          </div>

          <Card>
            <SectionHeader>Construction Milestones & Compliance Sign-offs</SectionHeader>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ color: C.textDim, textAlign: "left" }}>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>#</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Milestone</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Phase</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Planned Date</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Actual / Est.</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Status</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}`, textAlign: "center" }}>Sensor Verified</th>
                </tr>
              </thead>
              <tbody>
                {MILESTONES.map(m => {
                  const sIcon = m.status === "done" ? "✅ Done"
                              : m.status === "active" ? "🔄 In Progress"
                              : m.status === "overdue" ? "⚠ Overdue"
                              : "⏳ Pending";
                  const sCol = m.status === "done" ? C.success
                             : m.status === "active" ? C.gold
                             : m.status === "overdue" ? C.danger
                             : C.textDim;
                  const vIcon = m.verified === "yes" ? "✅ Yes" : "⏳ Pending";
                  const vCol = m.verified === "yes" ? C.success : C.textDim;
                  return (
                    <tr key={m.n} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ padding: "8px 6px", color: C.textDim, fontFamily: "monospace" }}>{m.n}</td>
                      <td style={{ padding: "8px 6px", color: C.text }}>{m.name}</td>
                      <td style={{ padding: "8px 6px", color: C.textDim }}>{m.phase}</td>
                      <td style={{ padding: "8px 6px", color: C.text, fontFamily: "monospace" }}>{m.planned}</td>
                      <td style={{ padding: "8px 6px", color: C.text, fontFamily: "monospace" }}>{m.actual}</td>
                      <td style={{ padding: "8px 6px", color: sCol, fontWeight: 600 }}>{sIcon}</td>
                      <td style={{ padding: "8px 6px", textAlign: "center", color: vCol, fontWeight: 600 }}>{vIcon}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16 }}>
              <button onClick={generateComplianceReport} style={{
                background: C.gold, color: C.bg, border: "none", borderRadius: 4,
                padding: "8px 18px", fontSize: 12, fontWeight: 700, cursor: "pointer",
                letterSpacing: 0.5,
              }}>Generate Compliance Report</button>
            </div>
          </Card>
        </div>
      )}

      {/* ═══ TAB 6: AUDIT LOG ════════════════════════════════════ */}
      {activeTab === "audit" && (
        <div>
          <Card>
            <SectionHeader>Tamper-Evident Construction Audit Chain</SectionHeader>
            <div style={{
              background: `${C.gold}10`, border: `1px solid ${C.gold}55`,
              borderRadius: 6, padding: "12px 14px", marginBottom: 16,
              fontSize: 12, color: C.text, lineHeight: 1.5,
            }}>
              Every sensor batch, milestone sign-off, and RTPMV computation is <strong style={{ color: C.gold }}>SHA-256 hashed and chained</strong>. This log is independently verifiable by regulators, lenders, and buyers.
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ color: C.textDim, textAlign: "left" }}>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>#</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Timestamp</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Event Type</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Description</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Hash (12)</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Prev. Hash</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Chain Status</th>
                </tr>
              </thead>
              <tbody>
                {auditLog.map(r => (
                  <tr key={`${r.n}-${r.hash}`} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: "7px 6px", color: C.textDim, fontFamily: "monospace" }}>{r.n}</td>
                    <td style={{ padding: "7px 6px", color: C.text, fontFamily: "monospace", fontSize: 10 }}>{r.ts}</td>
                    <td style={{ padding: "7px 6px", color: C.gold }}>{r.type}</td>
                    <td style={{ padding: "7px 6px", color: C.text }}>{r.desc}</td>
                    <td style={{ padding: "7px 6px", color: C.textDim, fontFamily: "monospace", fontSize: 10 }}>{r.hash}</td>
                    <td style={{ padding: "7px 6px", color: C.textMuted, fontFamily: "monospace", fontSize: 10 }}>{r.prev}</td>
                    <td style={{ padding: "7px 6px", color: C.success, fontWeight: 600 }}>✅ Valid</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div style={{
              marginTop: 16, padding: "10px 14px",
              background: `${C.success}10`, border: `1px solid ${C.success}55`,
              borderRadius: 6, color: C.success, fontWeight: 600, fontSize: 12,
            }}>
              Chain Integrity: ✅ All {auditLog.length} records valid. No tampering detected.
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16 }}>
              <button onClick={exportAuditReport} style={{
                background: "transparent", color: C.gold, border: `1px solid ${C.gold}`,
                borderRadius: 4, padding: "8px 18px", fontSize: 12, fontWeight: 700,
                cursor: "pointer", letterSpacing: 0.5,
              }}>Export Audit Report (PDF)</button>
            </div>
          </Card>
        </div>
      )}

      {/* ═══ TAB 7: HANDOVER READINESS ════════════════════════════════════ */}
      {activeTab === "handover" && (
        <div>
          <Card style={{ marginBottom: 14 }}>
            <SectionHeader>Handover Readiness Assessment — {activeProject.name}</SectionHeader>
            <div style={{ display: "flex", alignItems: "center", gap: 24, padding: "20px 0" }}>
              <div>
                <div style={{ fontSize: 64, fontWeight: 700, color: C.warning, lineHeight: 1 }}>74<span style={{ fontSize: 28 }}>%</span></div>
                <div style={{ fontSize: 12, color: C.warning, fontWeight: 600, marginTop: 6, letterSpacing: 0.5 }}>
                  NOT YET READY — 3 criteria outstanding
                </div>
              </div>
              <div style={{ flex: 1 }}>
                {READINESS_DIMS.map(d => {
                  const col = d.status === "ready" ? C.success : C.gold;
                  const ico = d.status === "ready" ? "✅ Ready" : "🔄 In Progress";
                  return (
                    <div key={d.name} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 12, color: C.text, fontWeight: 500 }}>{d.name}</span>
                        <span style={{ fontSize: 11, color: col, fontWeight: 600 }}>{d.pct}% — {ico}</span>
                      </div>
                      <ProgressBar value={d.pct} color={col} height={8} />
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>

          <div style={S.grid2}>
            <Card style={{
              border: `2px solid ${C.gold}`, position: "relative",
              background: `linear-gradient(135deg, ${C.surface} 0%, ${C.surfaceAlt} 100%)`,
            }}>
              <div style={{
                position: "absolute", top: 12, right: 16,
                fontSize: 9, color: C.gold, letterSpacing: 2, fontWeight: 700,
              }}>HANDOVER TOKEN — PREVIEW</div>
              <SectionHeader>Buyer Handover Token</SectionHeader>
              {[
                ["Property",                 activeProject.name],
                ["Developer",                activeProject.developer],
                ["Build Health at Handover", "[will be populated on handover]"],
                ["RTPMV at Handover",        "[will be computed on handover]"],
                ["Cryptographic Hash",       "[pending final audit]"],
              ].map(([l, v]) => (
                <div key={l} style={{
                  display: "grid", gridTemplateColumns: "180px 1fr",
                  padding: "8px 0", borderBottom: `1px dashed ${C.border}`,
                }}>
                  <span style={{ fontSize: 11, color: C.textDim }}>{l}</span>
                  <span style={{ fontSize: 11, color: C.text, fontFamily: l.includes("Hash") ? "monospace" : "inherit" }}>{v}</span>
                </div>
              ))}
              <div style={{
                marginTop: 14, padding: "8px 12px",
                background: `${C.warning}15`, border: `1px solid ${C.warning}55`,
                borderRadius: 4, fontSize: 11, color: C.warning, fontWeight: 600,
              }}>
                Status: PENDING — Complete all readiness criteria to generate token
              </div>
              <div style={{ marginTop: 14 }}>
                <button disabled title="Available when Handover Readiness Score reaches 90%" style={{
                  background: C.surfaceAlt, color: C.textMuted,
                  border: `1px solid ${C.border}`, borderRadius: 4,
                  padding: "8px 18px", fontSize: 12, fontWeight: 600,
                  cursor: "not-allowed", letterSpacing: 0.5,
                }}>Generate Handover Certificate</button>
              </div>
            </Card>

            <Card>
              <SectionHeader>Outstanding Readiness Items</SectionHeader>
              {[
                "Complete MEP commissioning L15–L26",
                "Fire safety system certification",
                "Obtain Occupancy Permit from GHMC",
              ].map(item => (
                <div key={item} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "10px 0", borderBottom: `1px solid ${C.border}`,
                }}>
                  <span style={{
                    width: 16, height: 16, border: `1.5px solid ${C.textDim}`,
                    borderRadius: 3, flexShrink: 0,
                  }} />
                  <span style={{ fontSize: 12, color: C.text }}>{item}</span>
                </div>
              ))}
              <div style={{ marginTop: 14, fontSize: 11, color: C.textMuted, fontStyle: "italic" }}>
                Once all items are sensor-verified and signed off, the handover token will be auto-generated and registered in the audit chain.
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ═══ TAB 8: PORTFOLIO ANALYTICS ════════════════════════════════════ */}
      {activeTab === "portfolio" && (
        <div>
          <SectionHeader>Portfolio Overview — All Active Projects</SectionHeader>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 14 }}>
            {PROJECTS.map(p => (
              <Card key={p.id} style={{ cursor: "pointer", borderColor: p.id === activeProjectId ? C.gold : C.border }}
                onClick={() => setActiveProjectId(p.id)}>
                <div style={{ fontSize: 13, fontWeight: 700, color: C.gold, marginBottom: 4 }}>{p.name}</div>
                <div style={{ fontSize: 11, color: C.textDim, marginBottom: 10 }}>{p.city}</div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                  <span style={{ color: C.textDim }}>Progress</span>
                  <span style={{ color: C.text, fontWeight: 600 }}>{p.progress}%</span>
                </div>
                <ProgressBar value={p.progress} color={C.gold} height={4} />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10, fontSize: 11 }}>
                  <span style={{ color: C.textDim }}>Build Health</span>
                  <HealthBadge value={p.buildHealth} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 11 }}>
                  <span style={{ color: C.textDim }}>Est. RTPMV (compl.)</span>
                  <span style={{ color: C.gold, fontWeight: 600 }}>{fmt.cr(p.rtpmvCompletion)}</span>
                </div>
              </Card>
            ))}
          </div>

          <div style={S.grid2}>
            <Card>
              <SectionHeader>Projected RTPMV at Completion (₹ Cr)</SectionHeader>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={PORTFOLIO_RTPMV} layout="vertical" margin={{ top: 10, right: 20, left: 60, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: C.textDim }} tickFormatter={v => `₹${v}`} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: C.text }} width={120} />
                  <Tooltip formatter={(v) => [`₹${v} Cr`]} contentStyle={{ background: C.surface, border: `1px solid ${C.gold}`, fontSize: 11, color: C.text }} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} fill={C.gold} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <SectionHeader>Portfolio Build Health Trend (Last 30 Days)</SectionHeader>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={PORTFOLIO_HEALTH_30D} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="day" tick={{ fontSize: 10, fill: C.textDim }} />
                  <YAxis domain={[50, 100]} tick={{ fontSize: 10, fill: C.textDim }} />
                  <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.gold}`, fontSize: 11, color: C.text }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Area type="monotone" dataKey="EMPRESS"    name="Empress"    stroke={C.gold}    fill={`${C.gold}33`}    strokeWidth={1.6} />
                  <Area type="monotone" dataKey="NAVIMUMBAI" name="Navi Mumbai" stroke={C.warning} fill={`${C.warning}22`} strokeWidth={1.6} />
                  <Area type="monotone" dataKey="WHITEFIELD" name="Whitefield" stroke={C.success} fill={`${C.success}22`} strokeWidth={1.6} />
                  <Area type="monotone" dataKey="PUNETECH"   name="Pune Tech"  stroke={C.danger}  fill={`${C.danger}22`}  strokeWidth={1.6} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card style={{ marginTop: 14 }}>
            <SectionHeader>Portfolio Active Alerts</SectionHeader>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ color: C.textDim, textAlign: "left" }}>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Project</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Floor</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Sensor ID</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Type</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}`, textAlign: "right" }}>Reading</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}`, textAlign: "right" }}>Threshold</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Status</th>
                  <th style={{ padding: "8px 6px", borderBottom: `1px solid ${C.border}` }}>Since</th>
                </tr>
              </thead>
              <tbody>
                {ALL_ALERTS.map((a, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: "7px 6px", color: C.text }}>{a.project}</td>
                    <td style={{ padding: "7px 6px", color: C.textDim, fontFamily: "monospace" }}>{a.floor}</td>
                    <td style={{ padding: "7px 6px", color: C.text, fontFamily: "monospace" }}>{a.id}</td>
                    <td style={{ padding: "7px 6px", color: C.textDim }}>{a.type}</td>
                    <td style={{ padding: "7px 6px", textAlign: "right", color: C.text, fontFamily: "monospace", fontWeight: 600 }}>{a.reading.toFixed(2)}</td>
                    <td style={{ padding: "7px 6px", textAlign: "right", color: C.textDim, fontFamily: "monospace" }}>{a.threshold}</td>
                    <td style={{ padding: "7px 6px", color: statusColor(a.status), fontWeight: 700 }}>{a.status}</td>
                    <td style={{ padding: "7px 6px", color: C.textDim, fontFamily: "monospace" }}>{a.since}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card style={{ marginTop: 14 }}>
            <SectionHeader>Portfolio Metrics</SectionHeader>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <tbody>
                {[
                  ["Total Active Projects",      "4"],
                  ["Total Built-Up Area",        "3,84,000 sq ft"],
                  ["Aggregate Baseline Value",   "₹1,124 Cr"],
                  ["Projected Portfolio RTPMV",  "₹2,104 Cr"],
                  ["Avg. Build Health Score",    "78.4"],
                  ["Properties with Active Alerts", "2"],
                  ["Est. Handover (nearest)",    "Sep 2026"],
                ].map(([metric, val]) => (
                  <tr key={metric} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: "10px 6px", color: C.textDim }}>{metric}</td>
                    <td style={{ padding: "10px 6px", color: C.text, fontWeight: 600, textAlign: "right" }}>{val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* Toast notifications */}
      <Toast msg={toast} onClose={() => setToast("")} />
    </div>
  );
}
