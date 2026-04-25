import { useState, useEffect, useRef } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  ReferenceLine,
} from "recharts";

// ── Simulation constants ──────────────────────────────────────────────────────
const PROPS = {
  villa: {
    id: "villa",
    name: "Villa Damansara",
    type: "Residential Bungalow",
    builtYear: 2008,
    landAreaSqft: 6500,
    builtUpSqft: 4200,
    landValue: 850_000,
    structureValue: 650_000,
    SHF: 0.88, ESF: 0.91, USS: 0.14, PDP: 0.93, CI: 0.87,
    sensorsActive: 8,
    baselineValue: 1_450_000,
    sensors: [
      { name: "Temperature",        reading: "28.4 °C",   category: "Environmental", status: "OK"      },
      { name: "Humidity",           reading: "67 %",      category: "Environmental", status: "OK"      },
      { name: "Vibration",          reading: "0.12 mm/s", category: "Structural",    status: "OK"      },
      { name: "Strain",             reading: "142 μɛ",    category: "Structural",    status: "Warning" },
      { name: "Occupancy",          reading: "4 persons", category: "Usage",         status: "OK"      },
      { name: "Electrical Load",    reading: "4.2 kW",    category: "Usage",         status: "OK"      },
      { name: "Water Consumption",  reading: "380 L/day", category: "Usage",         status: "OK"      },
      { name: "Structural Tilt",    reading: "0.08 °",    category: "Structural",    status: "OK"      },
    ],
    alerts: [
      { time: "2 hours ago", sensor: "Strain",          event: "Foundation strain above baseline", resolved: "No"  },
      { time: "3 days ago",  sensor: "Electrical Load", event: "Spike detected at 6 pm",           resolved: "Yes" },
      { time: "1 week ago",  sensor: "Humidity",        event: "Elevated overnight reading",       resolved: "Yes" },
    ],
  },
  shophouse: {
    id: "shophouse",
    name: "Shophouse Georgetown",
    type: "Commercial Shophouse",
    builtYear: 1975,
    landAreaSqft: 2800,
    builtUpSqft: 3600,
    landValue: 1_200_000,
    structureValue: 480_000,
    SHF: 0.72, ESF: 0.78, USS: 0.31, PDP: 0.71, CI: 0.74,
    sensorsActive: 6,
    baselineValue: 1_550_000,
    sensors: [
      { name: "Footfall Counter",        reading: "127 /hr",   category: "Usage",         status: "OK"      },
      { name: "HVAC Load",               reading: "8.7 kW",    category: "Usage",         status: "Warning" },
      { name: "Temperature (Shopfront)", reading: "31.2 °C",   category: "Environmental", status: "OK"      },
      { name: "Electrical Load",         reading: "11.4 kW",   category: "Usage",         status: "OK"      },
      { name: "Air Quality (PM2.5)",     reading: "38 μg/m³",  category: "Environmental", status: "Warning" },
      { name: "Structural Vibration",    reading: "0.31 mm/s", category: "Structural",    status: "OK"      },
    ],
    alerts: [
      { time: "2 hours ago", sensor: "HVAC Load",       event: "Reading above normal range", resolved: "No"  },
      { time: "1 day ago",   sensor: "Air Quality",     event: "PM2.5 elevated briefly",     resolved: "Yes" },
      { time: "3 days ago",  sensor: "Electrical Load", event: "Spike detected",             resolved: "Yes" },
    ],
  },
};

// ── Formula helpers ───────────────────────────────────────────────────────────
function hf(p) {
  return p.SHF * p.ESF * (1 - p.USS) * p.PDP * p.CI;
}
function rv(p) {
  return p.landValue + p.structureValue * hf(p);
}
function tradingStatusRaw(h, ci) {
  if (h >= 0.75 && ci >= 0.80) return "ACTIVE";
  if (h >= 0.55 || ci >= 0.65) return "RESTRICTED";
  return "HALTED";
}
function tradingStatusLabel(raw) {
  return { ACTIVE: "Trading Active", RESTRICTED: "Under Review", HALTED: "Trading Paused" }[raw];
}
function healthBand(h) {
  if (h >= 0.75) return { label: "Good",            key: "good",     color: "#2e7d32" };
  if (h >= 0.55) return { label: "Moderate",        key: "moderate", color: "#e65100" };
  return                { label: "Needs Attention", key: "needs",    color: "#c62828" };
}
function confidenceBand(ci) {
  if (ci >= 0.80) return { label: "High",     color: "#2e7d32" };
  if (ci >= 0.65) return { label: "Moderate", color: "#e65100" };
  return                { label: "Low",       color: "#c62828" };
}
function spreadPct(h, ci) {
  return Math.max(0.01, 0.02 + (1 - ci) * 0.06 + (1 - h) * 0.04);
}
function rm(v) {
  return "RM " + Math.round(v).toLocaleString("en-MY");
}

// ── Deterministic 90-day estimated-value series (seeded LCG) ─────────────────
function seededRng(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}
function rtpmvSeries(p, days = 90) {
  const rng  = seededRng(p.id === "villa" ? 42 : 43);
  const base = rv(p);
  const today = new Date();
  const rows  = [];
  let cumNoise = 0;
  for (let i = 0; i < days; i++) {
    cumNoise += (rng() - 0.5) * base * 0.006;
    const date = new Date(today);
    date.setDate(today.getDate() - (days - i - 1));
    const val = base + cumNoise;
    rows.push({
      date:      date.toLocaleDateString("en-MY", { month: "short", day: "numeric" }),
      rtpmv:     Math.round(val),
      land:      Math.round(p.landValue),
      structure: Math.round(val - p.landValue),
      baseline:  Math.round(p.baselineValue),
    });
  }
  return rows;
}

// ── Status / indicator colour helpers ─────────────────────────────────────────
const STATUS_COLOUR   = { ACTIVE: "#2e7d32", RESTRICTED: "#e65100", HALTED: "#c62828" };
const POSITIVE_COLOUR = v => v >= 0.75 ? "#2e7d32" : v >= 0.55 ? "#e65100" : "#c62828";
const LOAD_COLOUR     = v => v <= 0.25 ? "#2e7d32" : v <= 0.45 ? "#e65100" : "#c62828";

// ── Styles (inline) ───────────────────────────────────────────────────────────
const S = {
  app:          { minHeight: "100vh", background: "#0a1628", color: "#e8e0d4", fontFamily: "'Segoe UI', system-ui, sans-serif", fontSize: 14 },
  topbar:       { background: "#0d1f3c", borderBottom: "1px solid rgba(201,168,76,0.25)", padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 },
  logo:         { fontWeight: 700, fontSize: 20, color: "#C9A84C", letterSpacing: 2 },
  badge:        { background: "rgba(201,168,76,0.12)", border: "1px solid rgba(201,168,76,0.3)", color: "#C9A84C", fontSize: 11, padding: "3px 10px", borderRadius: 3, letterSpacing: 1 },
  demoBanner:   { background: "#1a2a1a", border: "1px solid #2e7d32", color: "#81c784", padding: "8px 32px", fontSize: 12, textAlign: "center" },
  section:      { padding: "28px 32px" },
  h2:           { fontFamily: "serif", fontSize: 20, color: "#C9A84C", letterSpacing: 1, marginBottom: 16, fontWeight: 500 },
  card:         { background: "#0d1f3c", border: "1px solid rgba(201,168,76,0.18)", borderRadius: 6, padding: "18px 22px" },
  metricLabel:  { fontSize: 11, color: "#8a8070", letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 },
  metricValue:  { fontSize: 22, fontWeight: 700, color: "#E2C97E" },
  metricSmall:  { fontSize: 13, color: "#a8a090", marginTop: 2 },
  grid2:        { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
  grid3:        { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 },
  grid4:        { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 },
  propCard:     { background: "#0d1f3c", border: "1px solid rgba(201,168,76,0.18)", borderRadius: 6, padding: "20px 24px", cursor: "pointer", transition: "border-color 0.2s" },
  propCardSel:  { background: "#0d1f3c", border: "2px solid #C9A84C", borderRadius: 6, padding: "20px 24px", cursor: "pointer" },
  tabBar:       { display: "flex", gap: 4, borderBottom: "1px solid rgba(201,168,76,0.2)", marginBottom: 24, flexWrap: "wrap" },
  tab:          { padding: "8px 18px", cursor: "pointer", fontSize: 13, color: "#a8a090", borderBottom: "2px solid transparent", background: "none", border: "none" },
  tabActive:    { padding: "8px 18px", cursor: "pointer", fontSize: 13, color: "#C9A84C", borderBottom: "2px solid #C9A84C", background: "none", border: "none" },
  pill:         { display: "inline-block", padding: "2px 10px", borderRadius: 3, fontSize: 11, fontWeight: 600, letterSpacing: 1 },
  table:        { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th:           { textAlign: "left", padding: "8px 12px", color: "#8a8070", borderBottom: "1px solid rgba(201,168,76,0.15)", fontSize: 11, letterSpacing: 1 },
  td:           { padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.05)", color: "#c8c0b0" },
  divider:      { borderTop: "1px solid rgba(201,168,76,0.12)", margin: "24px 0" },
  infoBox:      { background: "rgba(33,150,243,0.10)", border: "1px solid rgba(33,150,243,0.35)", borderRadius: 6, padding: "14px 18px", color: "#a8cfff", fontSize: 13, lineHeight: 1.55 },
  greyBox:      { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 6, padding: "12px 16px", color: "#a8a090", fontSize: 12, lineHeight: 1.5 },
  pausedBox:    { background: "#1a120a", border: "1px solid #e65100", borderRadius: 4, padding: "10px 14px", fontSize: 12, color: "#ffb878", lineHeight: 1.5, marginTop: 12 },
  healthBadge:  { display: "inline-block", padding: "4px 14px", borderRadius: 4, fontSize: 13, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase" },
  imagePh:      { background: "linear-gradient(135deg, #2a3a50 0%, #1a2840 100%)", border: "1px solid rgba(201,168,76,0.15)", borderRadius: 6, minHeight: 180, display: "flex", alignItems: "center", justifyContent: "center", color: "#7a8a9a", fontSize: 14, letterSpacing: 1, textAlign: "center", padding: "12px" },
};

// ── Help tooltip (hover) ──────────────────────────────────────────────────────
function Help({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-block", marginLeft: 6 }}>
      <span
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        style={{
          display: "inline-flex", justifyContent: "center", alignItems: "center",
          width: 16, height: 16, borderRadius: "50%",
          border: "1px solid #8a8070", color: "#8a8070",
          fontSize: 10, cursor: "help", userSelect: "none", fontWeight: 600,
        }}
      >?</span>
      {show && (
        <span style={{
          position: "absolute", bottom: "100%", left: "50%",
          transform: "translateX(-50%) translateY(-6px)",
          background: "#1a2a40", border: "1px solid rgba(201,168,76,0.4)",
          borderRadius: 4, padding: "8px 12px", fontSize: 11,
          color: "#e8e0d4", width: 220, zIndex: 20,
          whiteSpace: "normal", textAlign: "left", fontWeight: 400,
          letterSpacing: 0, textTransform: "none", lineHeight: 1.5,
          pointerEvents: "none",
          boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
        }}>{text}</span>
      )}
    </span>
  );
}

// ── House / shop SVG icon ─────────────────────────────────────────────────────
function HouseIcon({ size = 56, color = "#9bb0c4", type = "house" }) {
  if (type === "shop") {
    return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke={color} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M3 10l1.5-4h15L21 10" />
        <path d="M4 10v11h16V10" />
        <path d="M9 21v-6h6v6" />
        <path d="M3 10h18" />
        <path d="M7.5 13h2M14.5 13h2" />
      </svg>
    );
  }
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke={color} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 10.5L12 3l9 7.5V21H3V10.5z" />
      <path d="M9 21V13h6v8" />
      <path d="M15 7.5V4h2.5v5" />
    </svg>
  );
}

// ── Force chart to remount once after first paint (Recharts measure fix) ─────
function useChartRemount() {
  const [k, setK] = useState(0);
  useEffect(() => {
    const t1 = setTimeout(() => {
      setK(x => x + 1);
      if (typeof window !== "undefined") window.dispatchEvent(new Event("resize"));
    }, 60);
    const t2 = setTimeout(() => {
      if (typeof window !== "undefined") window.dispatchEvent(new Event("resize"));
    }, 250);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);
  return k;
}

// ── Hover tooltip wrapper for arbitrary trigger element ──────────────────────
function HoverTip({ text, children }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-flex", alignItems: "center" }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span style={{
          position: "absolute", top: "100%", right: 0,
          transform: "translateY(8px)",
          background: "#1a2a40", border: "1px solid rgba(201,168,76,0.4)",
          borderRadius: 4, padding: "10px 14px", fontSize: 12,
          color: "#e8e0d4", width: 280, zIndex: 30,
          whiteSpace: "normal", textAlign: "left", fontWeight: 400,
          letterSpacing: 0, textTransform: "none", lineHeight: 1.5,
          pointerEvents: "none",
          boxShadow: "0 6px 16px rgba(0,0,0,0.45)",
        }}>{text}</span>
      )}
    </span>
  );
}

// ── Labelled progress bar ────────────────────────────────────────────────────
function ProgressBar({ value, label, help, color = "#C9A84C", showNumber = true }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
        <span style={{ color: "#c8c0b0" }}>
          {label}{help && <Help text={help} />}
        </span>
        {showNumber && <span style={{ color, fontWeight: 600 }}>{pct.toFixed(2)}</span>}
      </div>
      <div style={{ width: "100%", height: 10, background: "rgba(255,255,255,0.08)", borderRadius: 5, overflow: "hidden" }}>
        <div style={{ width: `${pct * 100}%`, height: "100%", background: color, transition: "width 0.3s" }} />
      </div>
    </div>
  );
}

// ── Sparkline tick ────────────────────────────────────────────────────────────
function Sparkline({ data, color = "#C9A84C" }) {
  const recent = data.slice(-30);
  return (
    <ResponsiveContainer width="100%" height={60}>
      <AreaChart data={recent} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={`sg-${color.replace("#","")}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="rtpmv"
          stroke={color} strokeWidth={1.5}
          fill={`url(#sg-${color.replace("#","")})`} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Metric card ───────────────────────────────────────────────────────────────
function Metric({ label, value, sub, help }) {
  return (
    <div style={S.card}>
      <div style={S.metricLabel}>{label}{help && <Help text={help} />}</div>
      <div style={S.metricValue}>{value}</div>
      {sub && <div style={S.metricSmall}>{sub}</div>}
    </div>
  );
}

// ── Health badge ─────────────────────────────────────────────────────────────
function HealthBadge({ h, size = "md" }) {
  const band = healthBand(h);
  const sizing = size === "lg"
    ? { fontSize: 16, padding: "6px 18px" }
    : { fontSize: 12, padding: "4px 12px" };
  return (
    <span style={{
      ...S.healthBadge, ...sizing,
      background: band.color + "22", color: band.color,
      border: `1px solid ${band.color}66`,
    }}>{band.label}</span>
  );
}

// ── Tab: Portfolio Overview ───────────────────────────────────────────────────
function TabPortfolio({ selected, setSelected }) {
  const chartKey = useChartRemount();
  const allRv = Object.values(PROPS).map(p => rv(p));
  const total = allRv.reduce((a, b) => a + b, 0);
  const allHf = Object.values(PROPS).map(p => hf(p));
  const avgHf = allHf.reduce((a, b) => a + b, 0) / allHf.length;
  const avgBand = healthBand(avgHf);
  const totSensors = Object.values(PROPS).reduce((a, p) => a + p.sensorsActive, 0);
  const seriesAll  = Object.values(PROPS).map(p => rtpmvSeries(p, 90));
  const propCount  = Object.keys(PROPS).length;

  // Combined 30-day trend
  const combined = seriesAll[0].slice(-30).map((row, i) => ({
    date: row.date,
    Villa:     seriesAll[0].slice(-30)[i].rtpmv,
    Shophouse: seriesAll[1].slice(-30)[i].rtpmv,
  }));
  // Y-axis zoom fix
  const allVals = combined.flatMap(d => [d.Villa, d.Shophouse]);
  const yMin = Math.min(...allVals) * 0.95;
  const yMax = Math.max(...allVals) * 1.05;

  // Summary banner sentences
  const bandVerb = b => b === "good" ? "is in good condition"
                     : b === "moderate" ? "is in moderate condition"
                     : "needs attention";
  const summarySentences = [
    `Your ${propCount} properties are worth a combined ${rm(total)} today.`,
    ...Object.values(PROPS).map(p => {
      const b = healthBand(hf(p)).key;
      const warns = p.sensors.filter(s => s.status === "Warning").length;
      const warnText = warns > 0 ? ` — ${warns} sensor warning${warns > 1 ? "s" : ""} active` : "";
      return `${p.name} ${bandVerb(b)}${warnText}.`;
    }),
  ].join(" ");

  return (
    <div>
      {/* Top metrics */}
      <div style={S.grid4}>
        <Metric
          label="Total Property Value"
          value={rm(total)}
          sub={<span>average across your {propCount} properties</span>}
          help="The combined estimated value of every property you own on TwinVal."
        />
        <div style={S.card}>
          <div style={S.metricLabel}>
            Portfolio Health
            <Help text="Property Health Score — A score from 0 to 1 based on your sensors. 1.0 = perfect condition. Below 0.60 = action recommended." />
          </div>
          <div style={{ marginTop: 4, marginBottom: 6 }}>
            <span style={{
              ...S.healthBadge, fontSize: 14, padding: "5px 14px",
              background: avgBand.color + "22", color: avgBand.color,
              border: `1px solid ${avgBand.color}66`,
            }}>{avgBand.label}</span>
          </div>
          <div style={{ fontSize: 12, color: "#8a8070" }}>
            (Score: {avgHf.toFixed(2)} / 1.0)
          </div>
          <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>
            Based on sensor readings across your {propCount} properties
          </div>
        </div>
        <Metric label="Properties" value={propCount} />
        <Metric label="Active Sensors" value={totSensors} />
      </div>

      {/* Summary banner */}
      <div style={{ marginTop: 20, ...S.infoBox }}>
        {summarySentences}
      </div>

      {/* 30-day trend */}
      <div style={{ marginTop: 24 }}>
        <div style={{ fontSize: 12, color: "#8a8070", letterSpacing: 1, marginBottom: 8, textTransform: "uppercase" }}>
          Total Property Value — Last 30 Days
        </div>
        <div style={{ ...S.card, padding: "16px 12px", minHeight: 240 }}>
          <ResponsiveContainer key={`port-${chartKey}`} width="100%" height={220}>
            <LineChart data={combined}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={5} />
              <YAxis
                tick={{ fontSize: 10, fill: "#666" }}
                tickFormatter={v => `RM ${(v/1e6).toFixed(2)}M`}
                domain={[yMin, yMax]}
              />
              <Tooltip formatter={(v, name) => [rm(v), name]} contentStyle={{ background: "#0d1f3c", border: "1px solid #C9A84C55", borderRadius: 4 }} />
              <Legend />
              <Line type="monotone" dataKey="Villa"     stroke="#C9A84C" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="Shophouse" stroke="#2196f3" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Property cards */}
      <div style={{ marginTop: 24, ...S.grid2 }}>
        {Object.values(PROPS).map((p) => {
          const h = hf(p), r = rv(p);
          const statusRaw = tradingStatusRaw(h, p.CI);
          const statusLbl = tradingStatusLabel(statusRaw);
          const iconType  = p.type.toLowerCase().includes("shop") ? "shop" : "house";
          return (
            <div key={p.id}
              style={selected === p.id ? S.propCardSel : S.propCard}
              onClick={() => setSelected(p.id)}>
              <div style={{ fontSize: 11, color: "#8a8070", marginBottom: 4 }}>{p.type} · Built {p.builtYear}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#E2C97E", marginBottom: 10 }}>{p.name}</div>
              <div style={{ ...S.imagePh, minHeight: 110, gap: 10, flexDirection: "row" }}>
                <HouseIcon size={56} type={iconType} />
                <div style={{ textAlign: "left" }}>
                  <div style={{ fontSize: 12, color: "#9bb0c4" }}>Property image</div>
                  <div style={{ fontSize: 11, color: "#6a7a8a", marginTop: 2 }}>{p.name}</div>
                </div>
              </div>
              <div style={{ marginTop: 12, fontSize: 20, fontWeight: 700, color: "#C9A84C" }}>{rm(r)}</div>
              <div style={{ display: "flex", gap: 10, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
                <HealthBadge h={h} />
                <span style={{ ...S.pill, background: STATUS_COLOUR[statusRaw] + "22", color: STATUS_COLOUR[statusRaw], border: `1px solid ${STATUS_COLOUR[statusRaw]}55` }}>
                  {statusLbl}
                </span>
                <span style={{ color: "#888", fontSize: 12 }}>{p.sensorsActive} sensors</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Tab: My Properties (Property Detail) ─────────────────────────────────────
function TabProperty({ propId }) {
  const p = PROPS[propId];
  if (!p) return <div style={{ color: "#c62828" }}>Property not found.</div>;

  const h = hf(p);
  const r = rv(p);
  const band = healthBand(h);
  const now = new Date().toLocaleTimeString("en-MY");

  return (
    <div>
      <div style={{ fontSize: 20, fontWeight: 600, color: "#E2C97E", marginBottom: 16 }}>
        {p.name}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.3fr", gap: 20 }}>
        {/* Left: Image placeholder + metadata */}
        <div>
          <div style={{ ...S.imagePh, minHeight: 200, flexDirection: "column", gap: 12 }}>
            <HouseIcon size={88} type={p.type.toLowerCase().includes("shop") ? "shop" : "house"} />
            <div>
              <div style={{ fontSize: 16, color: "#b8c8d8", marginBottom: 4 }}>{p.name}</div>
              <div style={{ fontSize: 11, color: "#7a8a9a" }}>Property image</div>
            </div>
          </div>

          <div style={{ ...S.card, marginTop: 14 }}>
            <div style={S.metricLabel}>Property Details</div>
            <table style={{ width: "100%", fontSize: 13, marginTop: 6 }}>
              <tbody>
                <tr><td style={{ padding: "4px 0", color: "#8a8070" }}>Type</td><td style={{ color: "#E2C97E", textAlign: "right" }}>{p.type}</td></tr>
                <tr><td style={{ padding: "4px 0", color: "#8a8070" }}>Built Year</td><td style={{ color: "#E2C97E", textAlign: "right" }}>{p.builtYear}</td></tr>
                <tr><td style={{ padding: "4px 0", color: "#8a8070" }}>Land Area</td><td style={{ color: "#E2C97E", textAlign: "right" }}>{p.landAreaSqft.toLocaleString("en-MY")} sqft</td></tr>
                <tr><td style={{ padding: "4px 0", color: "#8a8070" }}>Built-up</td><td style={{ color: "#E2C97E", textAlign: "right" }}>{p.builtUpSqft.toLocaleString("en-MY")} sqft</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Value + Health + Indicators */}
        <div>
          {/* Estimated value */}
          <div style={{ ...S.card, padding: "22px 26px" }}>
            <div style={S.metricLabel}>
              Estimated Value
              <Help text="The current estimated market value of your property, updated daily based on sensor readings and market data." />
            </div>
            <div style={{ fontSize: 32, fontWeight: 700, color: "#C9A84C", marginTop: 4 }}>
              {rm(r)}
            </div>
            <div style={{ fontSize: 12, color: "#8a8070", marginTop: 2 }}>
              Land {rm(p.landValue)} + Building {rm(p.structureValue * h)}
            </div>
          </div>

          {/* Health score bar */}
          <div style={{ ...S.card, marginTop: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={S.metricLabel}>
                Property Health Score
                <Help text="Property Health Score — A score from 0 to 1 based on your sensors. 1.0 = perfect condition. Below 0.60 = action recommended." />
              </div>
              <HealthBadge h={h} size="lg" />
            </div>
            <div style={{ width: "100%", height: 14, background: "rgba(255,255,255,0.08)", borderRadius: 7, overflow: "hidden" }}>
              <div style={{ width: `${h * 100}%`, height: "100%", background: band.color, transition: "width 0.4s" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 11, color: "#8a8070" }}>
              <span>0.00</span>
              <span style={{ color: band.color, fontWeight: 600 }}>Score: {h.toFixed(2)}</span>
              <span>1.00</span>
            </div>
          </div>

          {/* Land / Structure value split */}
          <div style={{ ...S.grid2, marginTop: 14 }}>
            <div style={S.card}>
              <div style={S.metricLabel}>Land Value<Help text="The value of the land itself — this does not change with your property's condition." /></div>
              <div style={S.metricValue}>{rm(p.landValue)}</div>
            </div>
            <div style={S.card}>
              <div style={S.metricLabel}>Structure Value<Help text="The value of the building, adjusted for its current condition." /></div>
              <div style={S.metricValue}>{rm(p.structureValue * h)}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Five indicator bars */}
      <div style={{ marginTop: 22, ...S.card }}>
        <div style={{ ...S.metricLabel, marginBottom: 14 }}>Condition Indicators</div>
        <div style={S.grid2}>
          <div>
            <ProgressBar
              value={p.SHF}
              label="Structure Strength"
              help="How strong your building's frame and foundations are."
              color={POSITIVE_COLOUR(p.SHF)}
            />
            <ProgressBar
              value={p.ESF}
              label="Environment Stability"
              help="How stable the surroundings are — ground, drainage, climate."
              color={POSITIVE_COLOUR(p.ESF)}
            />
            <ProgressBar
              value={p.CI}
              label="Data Confidence"
              help="How reliable the sensor data is. Higher = more accurate valuation."
              color={POSITIVE_COLOUR(p.CI)}
            />
          </div>
          <div>
            <ProgressBar
              value={p.USS}
              label="Usage Load"
              help="How heavily your property is being used. Lower = less wear."
              color={LOAD_COLOUR(p.USS)}
            />
            <ProgressBar
              value={p.PDP}
              label="Age Deterioration"
              help="How well your property has aged. Higher = holding up better."
              color={POSITIVE_COLOUR(p.PDP)}
            />
          </div>
        </div>
      </div>

      {/* Sensors row */}
      <div style={{ ...S.grid2, marginTop: 14 }}>
        <div style={S.card}>
          <div style={S.metricLabel}>Sensors Active</div>
          <div style={S.metricValue}>{p.sensorsActive}</div>
          <div style={S.metricSmall}>Installed on this property</div>
        </div>
        <div style={S.card}>
          <div style={S.metricLabel}>Last Updated</div>
          <div style={{ ...S.metricValue, fontSize: 18 }}>{now}</div>
          <div style={S.metricSmall}>Updates automatically every 15 seconds</div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Value History (was Timeline) ────────────────────────────────────────
function TabTimeline({ propId }) {
  const p = PROPS[propId];
  const [range, setRange] = useState("90d");
  const [showLand,   setShowLand]   = useState(true);
  const [showStruct, setShowStruct] = useState(false);
  const chartKey = useChartRemount();

  const allData = rtpmvSeries(p);
  const days    = range === "30d" ? 30 : 90;
  const data    = allData.slice(-days);
  const vals    = data.map(d => d.rtpmv);
  const last    = vals[vals.length - 1];
  const prev    = vals[Math.max(0, vals.length - 30)];
  const chg30   = ((last - prev) / prev * 100).toFixed(2);
  const maxV    = Math.max(...vals);
  const minV    = Math.min(...vals);
  const yMin    = Math.min(...vals, p.baselineValue) * 0.97;
  const yMax    = Math.max(...vals, p.baselineValue) * 1.03;

  return (
    <div>
      <div style={{ fontSize: 15, color: "#E2C97E", marginBottom: 6, fontWeight: 600 }}>
        How your property's estimated value has changed over time
      </div>
      <div style={{ fontSize: 12, color: "#8a8070", marginBottom: 16 }}>
        {p.name} · last {days} days
      </div>

      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 6 }}>
          {["30d","90d"].map(r => (
            <button key={r} onClick={() => setRange(r)} style={{
              padding: "5px 14px", borderRadius: 3, cursor: "pointer", fontSize: 12,
              background: range === r ? "#C9A84C" : "transparent",
              color: range === r ? "#0a1628" : "#a8a090",
              border: `1px solid ${range === r ? "#C9A84C" : "rgba(201,168,76,0.3)"}`,
            }}>{r}</button>
          ))}
        </div>
        <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, color: "#a8a090", cursor: "pointer" }}>
          <input type="checkbox" checked={showLand} onChange={e => setShowLand(e.target.checked)} />
          Land Value (does not change with condition)
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, color: "#a8a090", cursor: "pointer" }}>
          <input type="checkbox" checked={showStruct} onChange={e => setShowStruct(e.target.checked)} />
          Building Value (changes with property health)
        </label>
      </div>

      <div style={{ ...S.card, padding: "16px 12px", minHeight: 320 }}>
        <ResponsiveContainer key={`tl-${chartKey}-${range}-${showLand}-${showStruct}`} width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={Math.floor(days/8)} />
            <YAxis
              tick={{ fontSize: 10, fill: "#666" }}
              tickFormatter={v => `RM ${(v/1e6).toFixed(2)}M`}
              domain={[yMin, yMax]}
            />
            <Tooltip formatter={(v, n) => [rm(v), n]} contentStyle={{ background: "#0d1f3c", border: "1px solid #C9A84C55", borderRadius: 4 }} />
            <Legend />
            <Line
              type="monotone" dataKey="baseline"
              name="Baseline (government assessment)"
              stroke="#2e7d32" strokeWidth={1.5} strokeDasharray="6 4" dot={false} isAnimationActive={false}
            />
            <Line type="monotone" dataKey="rtpmv" name="Estimated Property Value" stroke="#C9A84C" strokeWidth={2} dot={false} isAnimationActive={false} />
            {showLand   && <Line type="monotone" dataKey="land"      name="Land Value"     stroke="#4fc3f7" strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />}
            {showStruct && <Line type="monotone" dataKey="structure" name="Building Value" stroke="#e65100" strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: 20, ...S.grid4 }}>
        <Metric label="Today's Estimated Value" value={rm(last)} />
        <Metric label={`Highest Value (${days} days)`} value={rm(maxV)} />
        <Metric label={`Lowest Value (${days} days)`}  value={rm(minV)} />
        <Metric
          label="Change vs 30 Days Ago"
          value={`${chg30 >= 0 ? "▲" : "▼"} ${Math.abs(chg30)}%`}
          sub={<span style={{ color: chg30 >= 0 ? "#2e7d32" : "#c62828" }}>
            {chg30 >= 0 ? "Up" : "Down"} from a month ago
          </span>}
        />
      </div>
    </div>
  );
}

// ── Tab: Sensors ──────────────────────────────────────────────────────────────
function TabSensors({ propId }) {
  const p = PROPS[propId];
  const warnings = p.sensors.filter(s => s.status === "Warning");

  const ciGaugePct = Math.round(p.CI * 100);
  const ciColour   = POSITIVE_COLOUR(p.CI);
  const ciBand     = confidenceBand(p.CI);

  const CAT_LABEL = {
    Structural:    "Building Structure",
    Environmental: "Air & Temperature",
    Usage:         "Occupancy & Power",
    Safety:        "Safety Systems",
  };

  return (
    <div>
      <div style={{ fontSize: 15, color: "#E2C97E", marginBottom: 12, fontWeight: 600 }}>
        {p.name} · Sensor Network
      </div>

      {warnings.length > 0 && (
        <div style={{ background: "#1a0e00", border: "1px solid #e65100", borderRadius: 4, padding: "8px 16px", marginBottom: 16, fontSize: 12, color: "#ff8a50" }}>
          ⚠️ {warnings.length} sensor{warnings.length > 1 ? "s" : ""} in Warning state: {warnings.map(s => s.name).join(", ")}
        </div>
      )}

      <div style={S.grid2}>
        {/* Data Reliability Gauge */}
        <div style={S.card}>
          <div style={{ fontSize: 12, color: "#8a8070", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
            Data Reliability Score
            <Help text="How reliable your sensor data is. A higher score means more accurate property valuations." />
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
            <div style={{ position: "relative", width: 140, height: 70, overflow: "hidden" }}>
              <div style={{ position: "absolute", width: 140, height: 140, borderRadius: "50%", border: "14px solid rgba(255,255,255,0.06)", top: 0, left: 0 }} />
              <div style={{
                position: "absolute", width: 140, height: 140, borderRadius: "50%",
                border: `14px solid ${ciColour}`,
                top: 0, left: 0,
                clipPath: "inset(0 0 50% 0)",
                transform: `rotate(${ciGaugePct * 1.8 - 90}deg)`,
                transformOrigin: "70px 70px",
              }} />
              <div style={{ position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)", fontSize: 22, fontWeight: 700, color: ciColour }}>
                {p.CI.toFixed(2)}
              </div>
            </div>
            <div style={{ fontSize: 12, color: ciColour, fontWeight: 600 }}>
              {ciBand.label} confidence
            </div>
            <div style={{ fontSize: 11, color: "#8a8070", textAlign: "center", lineHeight: 1.4, maxWidth: 280 }}>
              This score reflects how reliable your sensor data is. A higher score means more accurate property valuations.
            </div>
          </div>

          <div style={S.divider} />
          <div style={{ fontSize: 12, color: "#8a8070", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>By Category</div>
          {["Structural","Environmental","Usage","Safety"].map(cat => {
            const catSensors = p.sensors.filter(s => s.category === cat);
            if (catSensors.length === 0) {
              return (
                <div key={cat} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 12, color: "#666" }}>
                  <span>— {CAT_LABEL[cat]}</span>
                  <span>no sensors</span>
                </div>
              );
            }
            const catWarn = catSensors.filter(s => s.status === "Warning").length;
            return (
              <div key={cat} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 12, color: "#a8a090" }}>
                <span>{catWarn ? "⚠️" : "✅"} {CAT_LABEL[cat]}</span>
                <span>{catSensors.length - catWarn} OK{catWarn ? `, ${catWarn} Warn` : ""}</span>
              </div>
            );
          })}
        </div>

        {/* Sensor table */}
        <div style={S.card}>
          <div style={{ fontSize: 12, color: "#8a8070", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>Live Sensor Feed</div>
          <table style={S.table}>
            <thead>
              <tr>
                <th style={S.th}>Sensor</th>
                <th style={S.th}>Reading</th>
                <th style={S.th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {p.sensors.map(s => (
                <tr key={s.name}>
                  <td style={S.td}>{s.name}</td>
                  <td style={{ ...S.td, fontFamily: "monospace", color: "#C9A84C" }}>{s.reading}</td>
                  <td style={{ ...S.td, color: s.status === "Warning" ? "#e65100" : "#2e7d32", fontWeight: 600 }}>
                    {s.status === "Warning" ? "⚠️ " : "✅ "}{s.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Alerts */}
      <div style={{ ...S.card, marginTop: 16 }}>
        <div style={{ fontSize: 12, color: "#8a8070", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>Recent Alerts</div>
        <table style={S.table}>
          <thead>
            <tr>
              <th style={S.th}>Time</th>
              <th style={S.th}>Sensor</th>
              <th style={S.th}>Event</th>
              <th style={S.th}>Resolved?</th>
            </tr>
          </thead>
          <tbody>
            {p.alerts.map((a, i) => (
              <tr key={i}>
                <td style={S.td}>{a.time}</td>
                <td style={{ ...S.td, color: "#C9A84C" }}>{a.sensor}</td>
                <td style={S.td}>{a.event}</td>
                <td style={{ ...S.td, color: a.resolved === "Yes" ? "#2e7d32" : "#e65100", fontWeight: 600 }}>
                  {a.resolved === "Yes" ? "✅ Yes" : "⚠️ No"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Tab: Market Listing (was Exchange) ───────────────────────────────────────
function TabExchange() {
  const [tick, setTick] = useState(0);
  useEffect(() => { const id = setInterval(() => setTick(t => t + 1), 13_000); return () => clearInterval(id); }, []);

  return (
    <div>
      {/* Explainer */}
      <div style={{ ...S.greyBox, marginBottom: 18 }}>
        Your properties are listed on the TwinVal exchange. <strong style={{ color: "#c8c0b0" }}>Buy Offer</strong> is the price a buyer might offer.
        &nbsp;<strong style={{ color: "#c8c0b0" }}>Sell Offer</strong> is your asking price. These update automatically based on your property's condition.
      </div>

      <div style={S.grid2}>
        {Object.values(PROPS).map((p, idx) => {
          const h = hf(p), r = rv(p);
          const sp   = spreadPct(h, p.CI);
          const bid  = r * (1 - sp / 2);
          const ask  = r * (1 + sp / 2);
          const statusRaw = tradingStatusRaw(h, p.CI);
          const statusLbl = tradingStatusLabel(statusRaw);
          const vol24  = [4.3, 6.2][idx];
          const valuationPaused = vol24 > 5;
          const band = healthBand(h);
          const ciB  = confidenceBand(p.CI);
          return (
            <div key={p.id} style={{ ...S.card, borderTop: `2px solid ${STATUS_COLOUR[statusRaw]}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#E2C97E" }}>{p.name}</div>
                <span style={{ ...S.pill, background: STATUS_COLOUR[statusRaw] + "22", color: STATUS_COLOUR[statusRaw], border: `1px solid ${STATUS_COLOUR[statusRaw]}55` }}>
                  {statusLbl}
                </span>
              </div>

              <div style={S.grid3}>
                <div>
                  <div style={S.metricLabel}>Estimated Value<Help text="The current estimated market value of your property." /></div>
                  <div style={{ color: "#C9A84C", fontWeight: 700, fontSize: 16 }}>{rm(r)}</div>
                </div>
                <div>
                  <div style={S.metricLabel}>Buy Offer<Help text="The price a buyer might offer for your property right now." /></div>
                  <div style={{ color: "#2e7d32", fontWeight: 700, fontSize: 16 }}>{rm(bid)}</div>
                </div>
                <div>
                  <div style={S.metricLabel}>Sell Offer<Help text="Your asking price, based on current property condition." /></div>
                  <div style={{ color: "#c62828", fontWeight: 700, fontSize: 16 }}>{rm(ask)}</div>
                </div>
              </div>

              <div style={{ marginTop: 14, display: "flex", gap: 20, fontSize: 12, color: "#a8a090", flexWrap: "wrap" }}>
                <span>
                  Property Health: <strong style={{ color: band.color }}>{band.label}</strong>
                  <span style={{ color: "#8a8070" }}> ({h.toFixed(2)})</span>
                </span>
                <span>
                  Data Reliability: <strong style={{ color: ciB.color }}>{ciB.label}</strong>
                  <span style={{ color: "#8a8070" }}> ({p.CI.toFixed(2)})</span>
                </span>
              </div>

              <div style={{ marginTop: 10, fontSize: 11, color: "#666" }}>
                Last updated: {tick % 13 || 7} seconds ago
              </div>

              {valuationPaused && (
                <div style={S.pausedBox}>
                  <div style={{ fontWeight: 700, color: "#ffb878", marginBottom: 4 }}>⚠️ Valuation Paused</div>
                  Unusual price movement detected. Our system is reviewing this property's valuation. This is automatic and temporary.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
const TABS = ["Portfolio", "My Properties", "Value History", "Sensors", "Market Listing"];

export default function TwinValIndividual() {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProp, setSelectedProp] = useState("villa");
  const [showSwitchHint, setShowSwitchHint] = useState(true);

  // Force chart re-layout after tab switch (fixes blank chart on first paint)
  useEffect(() => {
    const t = setTimeout(() => {
      if (typeof window !== "undefined") window.dispatchEvent(new Event("resize"));
    }, 40);
    return () => clearTimeout(t);
  }, [activeTab]);

  // Hide "switch property" hint after first click
  function handleSelectProp(id) {
    setSelectedProp(id);
    setShowSwitchHint(false);
  }

  const allRv  = Object.values(PROPS).map(p => rv(p));
  const total  = allRv.reduce((a, b) => a + b, 0);
  const avgHf  = Object.values(PROPS).map(p => hf(p)).reduce((a,b)=>a+b,0) / Object.keys(PROPS).length;
  const avgBand = healthBand(avgHf);

  // Navbar Health tooltip text — names properties that need attention
  const needsAttn = Object.values(PROPS).filter(p => healthBand(hf(p)).key === "needs");
  const attnSentence = needsAttn.length === 0
    ? "All your properties are within healthy ranges."
    : `${needsAttn.map(p => p.name).join(" and ")} ${needsAttn.length > 1 ? "need" : "needs"} a sensor inspection.`;
  const navHealthTip = `Your properties' combined health score is ${avgHf.toFixed(2)} / 1.0. ${attnSentence}`;

  const showSelector = activeTab === 1 || activeTab === 2 || activeTab === 3;

  return (
    <div style={S.app}>
      {/* Top bar */}
      <div style={S.topbar}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={S.logo}>TwinVal</div>
          <span style={S.badge}>INDIVIDUAL CUSTOMER</span>
        </div>
        <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ fontSize: 13, color: "#8a8070", display: "flex", alignItems: "center", gap: 14 }}>
            <span>Total&nbsp;<strong style={{ color: "#C9A84C" }}>{rm(total)}</strong></span>
            <HoverTip text={navHealthTip}>
              <span style={{ display: "flex", alignItems: "center", gap: 6, cursor: "help", padding: "2px 4px", borderBottom: "1px dotted rgba(201,168,76,0.35)" }}>
                Health:
                <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: avgBand.color }} />
                <strong style={{ color: avgBand.color }}>{avgBand.label}</strong>
                <span style={{
                  display: "inline-flex", justifyContent: "center", alignItems: "center",
                  width: 14, height: 14, borderRadius: "50%",
                  border: "1px solid #8a8070", color: "#8a8070",
                  fontSize: 9, fontWeight: 600, marginLeft: 2,
                }}>?</span>
              </span>
            </HoverTip>
          </div>
          <div style={{ fontSize: 10, color: "#4a6070", letterSpacing: 1 }}>
            {new Date().toLocaleTimeString("en-MY")}
          </div>
        </div>
      </div>

      {/* Demo banner */}
      <div style={S.demoBanner}>
        ⚠️ Demo Mode — All sensor readings and valuations are simulated for illustration purposes.
        &nbsp;·&nbsp; Patent Pending IN 202641030498
      </div>

      {/* Prominent property selector */}
      {showSelector && (
        <div style={{
          background: "#0a1628",
          borderBottom: "1px solid rgba(201,168,76,0.15)",
          padding: "14px 32px",
          display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap"
        }}>
          <span style={{ fontSize: 15, color: "#E2C97E", fontWeight: 600 }}>Viewing:</span>
          {Object.values(PROPS).map(p => {
            const sel = selectedProp === p.id;
            return (
              <button key={p.id} onClick={() => handleSelectProp(p.id)} style={{
                padding: "8px 20px", borderRadius: 4, cursor: "pointer",
                fontSize: 14, fontWeight: 600,
                background: sel ? "#C9A84C" : "transparent",
                color: sel ? "#0a1628" : "#C9A84C",
                border: `1.5px solid #C9A84C`,
                transition: "all 0.15s",
              }}>{p.name}</button>
            );
          })}
          {showSwitchHint && (
            <span style={{ fontSize: 11, color: "#8a8070", fontStyle: "italic", marginLeft: 4 }}>
              ← tap to switch property
            </span>
          )}
        </div>
      )}
      {!showSelector && (
        <div style={{
          background: "#0a1628",
          borderBottom: "1px solid rgba(201,168,76,0.12)",
          padding: "10px 32px", fontSize: 12, color: "#666"
        }}>
          Viewing all properties
        </div>
      )}

      {/* Tab bar */}
      <div style={{ ...S.section, paddingBottom: 0 }}>
        <div style={S.tabBar}>
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setActiveTab(i)}
              style={activeTab === i ? S.tabActive : S.tab}>
              {t}
            </button>
          ))}
        </div>

        {/* Tab content — key forces clean remount so Recharts measures correctly */}
        <div style={{ paddingTop: 16 }} key={`tab-${activeTab}-${selectedProp}`}>
          {activeTab === 0 && <TabPortfolio selected={selectedProp} setSelected={id => { handleSelectProp(id); setActiveTab(1); }} />}
          {activeTab === 1 && <TabProperty  propId={selectedProp} />}
          {activeTab === 2 && <TabTimeline  propId={selectedProp} />}
          {activeTab === 3 && <TabSensors   propId={selectedProp} />}
          {activeTab === 4 && <TabExchange  />}
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: "20px 32px", borderTop: "1px solid rgba(201,168,76,0.12)", fontSize: 11, color: "#4a5a6a", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <span>Aethel Twin Sdn. Bhd. · Reg. No. 202601012908 / 1675006-X · Kuala Lumpur</span>
        <span>TwinVal Individual Customer Dashboard · Demo Mode</span>
      </div>
    </div>
  );
}
