import { useState, useEffect } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

// ── Simulation constants (matches individual_customer.py exactly) ─────────────
const PROPS = {
  villa: {
    id: "villa",
    name: "Villa Damansara",
    type: "Residential Bungalow",
    builtYear: 2008,
    landValue: 850_000,
    structureValue: 650_000,
    SHF: 0.88, ESF: 0.91, USS: 0.14, PDP: 0.93, CI: 0.87,
    sensorsActive: 8,
    baselineValue: 1_450_000,
    sensors: [
      { name: "Temperature (Ground Fl)", reading: "28.4 °C",   category: "Environmental", status: "OK"      },
      { name: "Humidity",                reading: "67 %",       category: "Environmental", status: "OK"      },
      { name: "Structural Vibration",    reading: "0.12 mm/s",  category: "Structural",    status: "OK"      },
      { name: "Foundation Strain",       reading: "142 μɛ",     category: "Structural",    status: "Warning" },
      { name: "Occupancy Counter",       reading: "4 persons",  category: "Usage",         status: "OK"      },
      { name: "Electrical Load",         reading: "4.2 kW",     category: "Usage",         status: "OK"      },
      { name: "Roof Moisture",           reading: "12.1 %",     category: "Safety",        status: "OK"      },
      { name: "Air Quality (CO₂)",       reading: "612 ppm",    category: "Environmental", status: "OK"      },
    ],
  },
  shophouse: {
    id: "shophouse",
    name: "Shophouse Georgetown",
    type: "Commercial Shophouse",
    builtYear: 1975,
    landValue: 1_200_000,
    structureValue: 480_000,
    SHF: 0.72, ESF: 0.78, USS: 0.31, PDP: 0.71, CI: 0.74,
    sensorsActive: 6,
    baselineValue: 1_550_000,
    sensors: [
      { name: "Footfall Counter",        reading: "127 /hr",    category: "Usage",         status: "OK"      },
      { name: "HVAC Load",               reading: "8.7 kW",     category: "Usage",         status: "Warning" },
      { name: "Temperature (Shopfront)", reading: "31.2 °C",    category: "Environmental", status: "OK"      },
      { name: "Electrical Load",         reading: "11.4 kW",    category: "Usage",         status: "OK"      },
      { name: "Air Quality (PM2.5)",     reading: "38 μg/m³",   category: "Environmental", status: "Warning" },
      { name: "Structural Vibration",    reading: "0.31 mm/s",  category: "Structural",    status: "OK"      },
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
function tradingStatus(h, ci) {
  if (h >= 0.75 && ci >= 0.80) return "ACTIVE";
  if (h >= 0.55 || ci >= 0.65) return "RESTRICTED";
  return "HALTED";
}
function spreadPct(h, ci) {
  return Math.max(0.01, 0.02 + (1 - ci) * 0.06 + (1 - h) * 0.04);
}
function rm(v) {
  return "RM " + Math.round(v).toLocaleString("en-MY");
}

// ── Deterministic 90-day RTPMV series (seeded LCG) ───────────────────────────
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
    });
  }
  return rows;
}

// ── Status colour helpers ─────────────────────────────────────────────────────
const STATUS_COLOUR = { ACTIVE: "#2e7d32", RESTRICTED: "#e65100", HALTED: "#c62828" };
const HF_COLOUR     = h => h >= 0.80 ? "#2e7d32" : h >= 0.60 ? "#e65100" : "#c62828";

// ── Styles (inline) ───────────────────────────────────────────────────────────
const S = {
  app:          { minHeight: "100vh", background: "#0a1628", color: "#e8e0d4", fontFamily: "'Segoe UI', system-ui, sans-serif", fontSize: 14 },
  topbar:       { background: "#0d1f3c", borderBottom: "1px solid rgba(201,168,76,0.25)", padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" },
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
  tabBar:       { display: "flex", gap: 4, borderBottom: "1px solid rgba(201,168,76,0.2)", marginBottom: 24 },
  tab:          { padding: "8px 18px", cursor: "pointer", fontSize: 13, color: "#a8a090", borderBottom: "2px solid transparent", background: "none", border: "none" },
  tabActive:    { padding: "8px 18px", cursor: "pointer", fontSize: 13, color: "#C9A84C", borderBottom: "2px solid #C9A84C", background: "none", border: "none" },
  pill:         { display: "inline-block", padding: "2px 10px", borderRadius: 3, fontSize: 11, fontWeight: 600, letterSpacing: 1 },
  table:        { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th:           { textAlign: "left", padding: "8px 12px", color: "#8a8070", borderBottom: "1px solid rgba(201,168,76,0.15)", fontSize: 11, letterSpacing: 1 },
  td:           { padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.05)", color: "#c8c0b0" },
  formulaBox:   { background: "#060f1e", border: "1px solid rgba(201,168,76,0.2)", borderRadius: 4, padding: "12px 16px", fontFamily: "monospace", fontSize: 12, color: "#a8a090", marginTop: 12 },
  divider:      { borderTop: "1px solid rgba(201,168,76,0.12)", margin: "24px 0" },
};

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
function Metric({ label, value, sub }) {
  return (
    <div style={S.card}>
      <div style={S.metricLabel}>{label}</div>
      <div style={S.metricValue}>{value}</div>
      {sub && <div style={S.metricSmall}>{sub}</div>}
    </div>
  );
}

// ── Tab: Portfolio Overview ───────────────────────────────────────────────────
function TabPortfolio({ selected, setSelected }) {
  const allRv = Object.values(PROPS).map(p => rv(p));
  const total = allRv.reduce((a, b) => a + b, 0);
  const allHf = Object.values(PROPS).map(p => hf(p));
  const avgHf = allHf.reduce((a, b) => a + b, 0) / allHf.length;
  const totSensors = Object.values(PROPS).reduce((a, p) => a + p.sensorsActive, 0);
  const seriesAll  = Object.values(PROPS).map(p => rtpmvSeries(p, 90));

  // Build combined 30-day trend
  const combined = seriesAll[0].slice(-30).map((row, i) => ({
    date: row.date,
    Villa:     seriesAll[0].slice(-30)[i].rtpmv,
    Shophouse: seriesAll[1].slice(-30)[i].rtpmv,
  }));

  return (
    <div>
      {/* Top metrics */}
      <div style={S.grid4}>
        <Metric label="Total Portfolio" value={rm(total)} />
        <Metric label="Portfolio Health" value={avgHf.toFixed(2)}
          sub={<span style={{ color: HF_COLOUR(avgHf) }}>weighted avg</span>} />
        <Metric label="Properties" value={Object.keys(PROPS).length} />
        <Metric label="Active Sensors" value={totSensors} />
      </div>

      {/* 30-day trend */}
      <div style={{ marginTop: 24 }}>
        <div style={{ fontSize: 12, color: "#8a8070", letterSpacing: 1, marginBottom: 8, textTransform: "uppercase" }}>Portfolio RTPMV — Last 30 Days</div>
        <div style={{ ...S.card, padding: "16px 12px" }}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={combined}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={5} />
              <YAxis tick={{ fontSize: 10, fill: "#666" }} tickFormatter={v => `RM ${(v/1e6).toFixed(2)}M`} />
              <Tooltip formatter={(v, name) => [rm(v), name]} contentStyle={{ background: "#0d1f3c", border: "1px solid #C9A84C55", borderRadius: 4 }} />
              <Legend />
              <Line type="monotone" dataKey="Villa"     stroke="#C9A84C" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Shophouse" stroke="#2196f3" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Property cards */}
      <div style={{ marginTop: 24, ...S.grid2 }}>
        {Object.values(PROPS).map((p, idx) => {
          const h = hf(p), r = rv(p);
          const status = tradingStatus(h, p.CI);
          const series = seriesAll[idx];
          const colours = ["#C9A84C", "#2196f3"];
          return (
            <div key={p.id}
              style={selected === p.id ? S.propCardSel : S.propCard}
              onClick={() => setSelected(p.id)}>
              <div style={{ fontSize: 11, color: "#8a8070", marginBottom: 4 }}>{p.type} · Built {p.builtYear}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#E2C97E", marginBottom: 10 }}>{p.name}</div>
              <Sparkline data={series} color={colours[idx]} />
              <div style={{ marginTop: 10, fontSize: 20, fontWeight: 700, color: "#C9A84C" }}>{rm(r)}</div>
              <div style={{ display: "flex", gap: 12, marginTop: 8, alignItems: "center" }}>
                <span style={{ color: HF_COLOUR(h), fontSize: 13 }}>HF {h.toFixed(2)}</span>
                <span style={{ ...S.pill, background: STATUS_COLOUR[status] + "22", color: STATUS_COLOUR[status], border: `1px solid ${STATUS_COLOUR[status]}55` }}>
                  {status}
                </span>
                <span style={{ color: "#666", fontSize: 12 }}>{p.sensorsActive} sensors</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Tab: Property Detail ──────────────────────────────────────────────────────
function TabProperty({ propId }) {
  const p    = PROPS[propId];
  const h    = hf(p);
  const r    = rv(p);
  const age  = new Date().getFullYear() - p.builtYear;
  const effAge = age * (1 - h * 0.4);
  const delta  = r - p.baselineValue;
  const deltaPct = (delta / p.baselineValue * 100).toFixed(1);

  const radarData = [
    { ind: "SHF",   val: p.SHF },
    { ind: "ESF",   val: p.ESF },
    { ind: "1−USS", val: 1 - p.USS },
    { ind: "PDP",   val: p.PDP },
    { ind: "CI",    val: p.CI },
  ];

  const ageData = [
    { label: "Chronological", years: age },
    { label: "Effective",     years: parseFloat(effAge.toFixed(1)) },
  ];

  return (
    <div>
      <div style={{ fontSize: 18, fontWeight: 600, color: "#E2C97E", marginBottom: 16 }}>{p.name}</div>

      {/* Metadata */}
      <div style={S.grid4}>
        <Metric label="Type"         value={p.type} />
        <Metric label="Built Year"   value={p.builtYear} />
        <Metric label="Land Area"    value="—" sub="See records" />
        <Metric label="Sensors"      value={p.sensorsActive + " active"} />
      </div>

      <div style={S.divider} />

      {/* RTPMV breakdown */}
      <div style={S.grid3}>
        <Metric label="Land Value"          value={rm(p.landValue)} />
        <Metric label="Structure (adj.)"    value={rm(p.structureValue * h)} sub={`× HF ${h.toFixed(4)}`} />
        <Metric label="RTPMV"               value={rm(r)}
          sub={<span style={{ color: delta >= 0 ? "#2e7d32" : "#c62828" }}>
            {delta >= 0 ? "▲" : "▼"} {Math.abs(deltaPct)}% vs baseline
          </span>} />
      </div>

      <div style={S.formulaBox}>
        RTPMV = Land + Structure × Health Factor<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {rm(p.landValue)} + {rm(p.structureValue)} × {h.toFixed(4)}<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong style={{ color: "#C9A84C" }}>{rm(r)}</strong>
      </div>

      <div style={S.divider} />

      {/* Radar + Age */}
      <div style={S.grid2}>
        <div style={S.card}>
          <div style={{ fontSize: 12, color: "#8a8070", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Health Factor Breakdown</div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(201,168,76,0.2)" />
              <PolarAngleAxis dataKey="ind" tick={{ fill: "#a8a090", fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 1]} tick={{ fill: "#666", fontSize: 9 }} />
              <Radar dataKey="val" stroke="#C9A84C" fill="#C9A84C" fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ textAlign: "center", color: HF_COLOUR(h), fontWeight: 700, fontSize: 18 }}>HF = {h.toFixed(4)}</div>
        </div>
        <div style={S.card}>
          <div style={{ fontSize: 12, color: "#8a8070", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Age Analysis</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={ageData} barSize={40}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="label" tick={{ fill: "#a8a090", fontSize: 11 }} />
              <YAxis tick={{ fill: "#666", fontSize: 11 }} unit=" yr" />
              <Tooltip contentStyle={{ background: "#0d1f3c", border: "1px solid #C9A84C55" }} />
              <Bar dataKey="years" fill="#1565c0" radius={[3, 3, 0, 0]}
                label={{ position: "top", fill: "#a8a090", fontSize: 11, formatter: v => `${v} yr` }} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: 8, fontSize: 12, color: "#8a8070" }}>
            Effective age derived from Health Factor. Lower HF → higher effective age relative to chronological.
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Valuation Timeline ───────────────────────────────────────────────────
function TabTimeline({ propId }) {
  const p        = PROPS[propId];
  const [range, setRange] = useState("90d");
  const [showLand, setShowLand]   = useState(true);
  const [showStruct, setShowStruct] = useState(false);
  const allData  = rtpmvSeries(p);
  const days     = range === "30d" ? 30 : 90;
  const data     = allData.slice(-days);
  const vals     = data.map(d => d.rtpmv);
  const chg30    = ((vals[vals.length-1] - vals[Math.max(0, vals.length-30)]) / vals[Math.max(0, vals.length-30)] * 100).toFixed(2);
  const maxV     = Math.max(...vals), minV = Math.min(...vals);
  const vol      = (Math.sqrt(vals.reduce((s, v) => s + Math.pow(v - vals.reduce((a,b) => a+b,0)/vals.length, 2), 0) / vals.length) / (vals.reduce((a,b)=>a+b,0)/vals.length) * 100).toFixed(2);

  return (
    <div>
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
          Land component
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, color: "#a8a090", cursor: "pointer" }}>
          <input type="checkbox" checked={showStruct} onChange={e => setShowStruct(e.target.checked)} />
          Structure component
        </label>
      </div>

      <div style={{ ...S.card, padding: "16px 12px" }}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={Math.floor(days/8)} />
            <YAxis tick={{ fontSize: 10, fill: "#666" }} tickFormatter={v => `RM ${(v/1e6).toFixed(2)}M`} />
            <Tooltip formatter={(v, n) => [rm(v), n]} contentStyle={{ background: "#0d1f3c", border: "1px solid #C9A84C55", borderRadius: 4 }} />
            <Legend />
            <Line type="monotone" dataKey="rtpmv" name="RTPMV" stroke="#C9A84C" strokeWidth={2} dot={false} />
            {showLand   && <Line type="monotone" dataKey="land"      name="Land Value"      stroke="#2e7d32" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />}
            {showStruct && <Line type="monotone" dataKey="structure" name="Structure (adj.)" stroke="#e65100" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: 20, ...S.grid4 }}>
        <Metric label="Current RTPMV" value={rm(vals[vals.length-1])} />
        <Metric label="Highest"       value={rm(maxV)} />
        <Metric label="Lowest"        value={rm(minV)} />
        <Metric label="30-Day Change" value={`${chg30 >= 0 ? "▲" : "▼"} ${Math.abs(chg30)}%`}
          sub={<span style={{ color: chg30 >= 0 ? "#2e7d32" : "#c62828" }}>vs 30 days ago</span>} />
      </div>
    </div>
  );
}

// ── Tab: Sensors & Health ─────────────────────────────────────────────────────
function TabSensors({ propId }) {
  const p = PROPS[propId];
  const warnings = p.sensors.filter(s => s.status === "Warning");

  const ciGaugePct = Math.round(p.CI * 100);
  const ciColour   = HF_COLOUR(p.CI);

  return (
    <div>
      {warnings.length > 0 && (
        <div style={{ background: "#1a0e00", border: "1px solid #e65100", borderRadius: 4, padding: "8px 16px", marginBottom: 16, fontSize: 12, color: "#ff8a50" }}>
          ⚠️ {warnings.length} sensor{warnings.length > 1 ? "s" : ""} in Warning state: {warnings.map(s => s.name).join(", ")}
        </div>
      )}

      <div style={S.grid2}>
        {/* CI Gauge (CSS arc) */}
        <div style={S.card}>
          <div style={{ fontSize: 12, color: "#8a8070", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>Confidence Index</div>
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
            <div style={{ fontSize: 11, color: "#666" }}>
              {p.CI >= 0.80 ? "High confidence" : p.CI >= 0.65 ? "Moderate confidence" : "Low confidence"}
            </div>
          </div>

          <div style={S.divider} />
          <div style={{ fontSize: 12, color: "#8a8070", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>By Category</div>
          {["Structural","Environmental","Usage","Safety"].map(cat => {
            const catSensors = p.sensors.filter(s => s.category === cat);
            const catWarn    = catSensors.filter(s => s.status === "Warning").length;
            return (
              <div key={cat} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 12, color: "#a8a090" }}>
                <span>{catWarn ? "⚠️" : "✅"} {cat}</span>
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
    </div>
  );
}

// ── Tab: Exchange ─────────────────────────────────────────────────────────────
function TabExchange() {
  const [tick, setTick] = useState(0);
  useEffect(() => { const id = setInterval(() => setTick(t => t + 1), 13_000); return () => clearInterval(id); }, []);

  return (
    <div>
      <div style={S.grid2}>
        {Object.values(PROPS).map((p, idx) => {
          const h = hf(p), r = rv(p);
          const sp   = spreadPct(h, p.CI);
          const bid  = r * (1 - sp / 2);
          const ask  = r * (1 + sp / 2);
          const status = tradingStatus(h, p.CI);
          const vol24  = [4.3, 6.2][idx];   // shophouse triggers circuit breaker
          const circuitBreaker = vol24 > 5;
          return (
            <div key={p.id} style={{ ...S.card, borderTopColor: STATUS_COLOUR[status], borderTopWidth: 2 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                <div style={{ fontSize: 15, fontWeight: 600, color: "#E2C97E" }}>{p.name}</div>
                <span style={{ ...S.pill, background: STATUS_COLOUR[status] + "22", color: STATUS_COLOUR[status], border: `1px solid ${STATUS_COLOUR[status]}55` }}>
                  {status}
                </span>
              </div>
              <div style={S.grid3}>
                <div>
                  <div style={S.metricLabel}>RTPMV</div>
                  <div style={{ color: "#C9A84C", fontWeight: 700, fontSize: 15 }}>{rm(r)}</div>
                </div>
                <div>
                  <div style={S.metricLabel}>Bid</div>
                  <div style={{ color: "#2e7d32", fontWeight: 700, fontSize: 15 }}>{rm(bid)}</div>
                </div>
                <div>
                  <div style={S.metricLabel}>Ask</div>
                  <div style={{ color: "#c62828", fontWeight: 700, fontSize: 15 }}>{rm(ask)}</div>
                </div>
              </div>
              <div style={{ marginTop: 10, display: "flex", gap: 16, fontSize: 12, color: "#8a8070" }}>
                <span>Spread: {(sp * 100).toFixed(1)}%</span>
                <span>HF: {h.toFixed(2)}</span>
                <span>CI: {p.CI.toFixed(2)}</span>
                <span style={{ color: circuitBreaker ? "#c62828" : "#666" }}>
                  24h Vol: {vol24.toFixed(1)}%{circuitBreaker ? " ⚡" : ""}
                </span>
              </div>
              {circuitBreaker && (
                <div style={{ marginTop: 8, background: "#1a0000", border: "1px solid #c62828", borderRadius: 3, padding: "5px 10px", fontSize: 11, color: "#ef5350" }}>
                  ⚡ CIRCUIT BREAKER — 24h volatility exceeded 5%. Trading restricted.
                </div>
              )}
              <div style={{ marginTop: 10, fontSize: 11, color: "#666", fontFamily: "monospace" }}>
                verified {tick % 13}s ago
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 16, ...S.card, fontSize: 12, color: "#8a8070" }}>
        <strong style={{ color: "#C9A84C" }}>Spread formula:</strong>&nbsp;
        spread = max(1%, 2% + (1−CI)×6% + (1−HF)×4%) &nbsp;·&nbsp;
        bid = RTPMV × (1 − spread/2) &nbsp;·&nbsp;
        ask = RTPMV × (1 + spread/2)
      </div>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
const TABS = ["Portfolio", "My Properties", "Timeline", "Sensors", "Exchange"];

export default function TwinValIndividual() {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProp, setSelectedProp] = useState("villa");

  const allRv  = Object.values(PROPS).map(p => rv(p));
  const total  = allRv.reduce((a, b) => a + b, 0);
  const avgHf  = Object.values(PROPS).map(p => hf(p)).reduce((a,b)=>a+b,0) / Object.keys(PROPS).length;

  return (
    <div style={S.app}>
      {/* Top bar */}
      <div style={S.topbar}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={S.logo}>TwinVal</div>
          <span style={S.badge}>INDIVIDUAL CUSTOMER</span>
        </div>
        <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
          <div style={{ fontSize: 12, color: "#8a8070" }}>
            Portfolio &nbsp;<strong style={{ color: "#C9A84C" }}>{rm(total)}</strong>
            &nbsp;·&nbsp; Health&nbsp;<strong style={{ color: HF_COLOUR(avgHf) }}>{avgHf.toFixed(2)}</strong>
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

      {/* Property selector strip */}
      <div style={{ background: "#0a1628", borderBottom: "1px solid rgba(201,168,76,0.12)", padding: "10px 32px", display: "flex", gap: 12, alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#8a8070", letterSpacing: 1, textTransform: "uppercase" }}>Viewing:</span>
        {activeTab > 0 && activeTab < 4 && Object.values(PROPS).map(p => {
          const sel = selectedProp === p.id;
          return (
            <button key={p.id} onClick={() => setSelectedProp(p.id)} style={{
              padding: "4px 14px", borderRadius: 3, cursor: "pointer", fontSize: 12,
              background: sel ? "rgba(201,168,76,0.15)" : "transparent",
              color: sel ? "#C9A84C" : "#8a8070",
              border: `1px solid ${sel ? "#C9A84C55" : "transparent"}`,
            }}>{p.name}</button>
          );
        })}
        {(activeTab === 0 || activeTab === 4) && (
          <span style={{ fontSize: 12, color: "#666" }}>All properties</span>
        )}
      </div>

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

        {/* Tab content */}
        <div style={{ paddingTop: 16 }}>
          {activeTab === 0 && <TabPortfolio selected={selectedProp} setSelected={id => { setSelectedProp(id); setActiveTab(1); }} />}
          {activeTab === 1 && <TabProperty  propId={selectedProp} />}
          {activeTab === 2 && <TabTimeline  propId={selectedProp} />}
          {activeTab === 3 && <TabSensors   propId={selectedProp} />}
          {activeTab === 4 && <TabExchange  />}
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: "20px 32px", borderTop: "1px solid rgba(201,168,76,0.12)", fontSize: 11, color: "#4a5a6a", display: "flex", justifyContent: "space-between" }}>
        <span>Aethel Twin Sdn. Bhd. · Reg. No. 202601012908 / 1675006-X · Kuala Lumpur</span>
        <span>TwinVal Individual Customer Dashboard · Demo Mode</span>
      </div>
    </div>
  );
}
