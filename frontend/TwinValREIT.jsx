import { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PieChart, Pie, Cell, Legend, ReferenceLine,
} from "recharts";

// ── Properties ─────────────────────────────────────────────────────────────────
const PROPERTIES = [
  { id: "KLCC_TOWER",   name: "KLCC Tower",           type: "Grade A Office",          location: "Kuala Lumpur City Centre", area: 125000, govtVal: 185_000_000, lat: 3.1578,  lng: 101.7123 },
  { id: "AXIS_SHAH",    name: "Axis Shah Alam DC",     type: "Industrial / Data Centre", location: "Shah Alam, Selangor",       area:  48000, govtVal:  62_000_000, lat: 3.0851,  lng: 101.5325 },
  { id: "ALAQAR_MED",   name: "Al-Aqar Medical Hub",   type: "Healthcare",               location: "Damansara, KL",             area:  32000, govtVal:  41_000_000, lat: 3.1502,  lng: 101.6235 },
  { id: "PAVILION_RTL", name: "Pavilion Retail Arcade", type: "Retail",                  location: "Bukit Bintang, KL",         area:  89000, govtVal: 137_000_000, lat: 3.1489,  lng: 101.7132 },
  { id: "SUNWAY_LOG",   name: "Sunway Logistics Park",  type: "Logistics / Industrial",   location: "Subang Jaya",               area:  71000, govtVal:  88_000_000, lat: 3.0588,  lng: 101.5841 },
];

// ── REIT Capital Structure ──────────────────────────────────────────────────────
const CAPITAL_STRUCTURE = {
  unitsOutstanding: 850_000_000,
  marketUnitPrice:  0.62,
  totalDebt:        183_000_000,
  avgCostOfDebt:    0.0425,
  managementFeeRate: 0.0040,
  trusteeFeeRate:    0.0005,
  gearingLimit:     0.50,
  debtMaturity: [
    { year: "2025", amount: 18_000_000 },
    { year: "2026", amount: 35_000_000 },
    { year: "2027", amount: 42_000_000 },
    { year: "2028", amount: 55_000_000 },
    { year: "2029", amount: 33_000_000 },
  ],
};

// ── Income / NPI Data ──────────────────────────────────────────────────────────
const INCOME_DATA = {
  KLCC_TOWER:   { monthlyRatePerSqft: 10.50, opexRatio: 0.26, occupancyRate: 0.92 },
  AXIS_SHAH:    { monthlyRatePerSqft:  5.20, opexRatio: 0.18, occupancyRate: 0.98 },
  ALAQAR_MED:   { monthlyRatePerSqft:  7.80, opexRatio: 0.22, occupancyRate: 0.96 },
  PAVILION_RTL: { monthlyRatePerSqft: 12.40, opexRatio: 0.34, occupancyRate: 0.88 },
  SUNWAY_LOG:   { monthlyRatePerSqft:  3.60, opexRatio: 0.15, occupancyRate: 0.94 },
};

// ── Lease Profiles ─────────────────────────────────────────────────────────────
const LEASE_PROFILES = {
  KLCC_TOWER: {
    wale: 3.8,
    expiry: [
      { year: "2025", pct: 8  },
      { year: "2026", pct: 14 },
      { year: "2027", pct: 22 },
      { year: "2028", pct: 31 },
      { year: "2029+", pct: 25 },
    ],
    tenants: [
      { name: "Petronas",       pct: 28, sector: "Oil & Gas"    },
      { name: "HSBC Malaysia",  pct: 19, sector: "Banking"      },
      { name: "McKinsey & Co",  pct: 12, sector: "Consulting"   },
      { name: "Others",         pct: 41, sector: "Diversified"  },
    ],
  },
  AXIS_SHAH: {
    wale: 6.2,
    expiry: [
      { year: "2025", pct: 0  },
      { year: "2026", pct: 5  },
      { year: "2027", pct: 10 },
      { year: "2028", pct: 15 },
      { year: "2029+", pct: 70 },
    ],
    tenants: [
      { name: "AWS Malaysia",   pct: 45, sector: "Technology"   },
      { name: "Telekom Malaysia",pct: 30, sector: "Telco"       },
      { name: "Others",         pct: 25, sector: "Diversified"  },
    ],
  },
  ALAQAR_MED: {
    wale: 8.1,
    expiry: [
      { year: "2025", pct: 0 },
      { year: "2026", pct: 0 },
      { year: "2027", pct: 5 },
      { year: "2028", pct: 10},
      { year: "2029+", pct: 85},
    ],
    tenants: [
      { name: "KPJ Healthcare", pct: 72, sector: "Healthcare"   },
      { name: "Ramsay Sime",    pct: 18, sector: "Healthcare"   },
      { name: "Others",         pct: 10, sector: "Diversified"  },
    ],
  },
  PAVILION_RTL: {
    wale: 2.9,
    expiry: [
      { year: "2025", pct: 18 },
      { year: "2026", pct: 26 },
      { year: "2027", pct: 28 },
      { year: "2028", pct: 20 },
      { year: "2029+", pct: 8  },
    ],
    tenants: [
      { name: "Parkson",        pct: 22, sector: "Department Store" },
      { name: "GSC Cinemas",    pct: 11, sector: "Entertainment"    },
      { name: "F&B Tenants",    pct: 30, sector: "Food & Beverage"  },
      { name: "Fashion / Misc", pct: 37, sector: "Retail Mix"       },
    ],
  },
  SUNWAY_LOG: {
    wale: 4.5,
    expiry: [
      { year: "2025", pct: 5  },
      { year: "2026", pct: 10 },
      { year: "2027", pct: 20 },
      { year: "2028", pct: 35 },
      { year: "2029+", pct: 30 },
    ],
    tenants: [
      { name: "DHL Supply Chain",pct: 38, sector: "Logistics"      },
      { name: "Pos Malaysia",    pct: 22, sector: "Postal/Courier" },
      { name: "Others",          pct: 40, sector: "Diversified"    },
    ],
  },
};

// ── Capex Budget ────────────────────────────────────────────────────────────────
const CAPEX_BUDGET = {
  KLCC_TOWER:   { sustainingBudget: 2_800_000, enhancementBudget: 1_500_000, ytdSpend: 1_950_000, pipeline: ["M&E Upgrade Q3", "Lobby Refurb Q4"], sensorFlags: [] },
  AXIS_SHAH:    { sustainingBudget: 1_200_000, enhancementBudget:   800_000, ytdSpend:   640_000, pipeline: ["UPS Replacement Q2", "Cooling Tower Q4"], sensorFlags: ["HVAC efficiency below target"] },
  ALAQAR_MED:   { sustainingBudget:   900_000, enhancementBudget:   400_000, ytdSpend:   510_000, pipeline: ["Medical Gas Lines Q3"], sensorFlags: [] },
  PAVILION_RTL: { sustainingBudget: 3_400_000, enhancementBudget: 2_200_000, ytdSpend: 2_780_000, pipeline: ["Food Court Reno Q3", "New Atrium Escalators"], sensorFlags: ["Structural check flagged"] },
  SUNWAY_LOG:   { sustainingBudget: 1_600_000, enhancementBudget:   600_000, ytdSpend:   820_000, pipeline: ["Dock Leveler Replacement Q2"], sensorFlags: [] },
};

// ── Market Comparables ─────────────────────────────────────────────────────────
const MARKET_COMPS = {
  KLCC_TOWER:   { compPsf: 1450, yieldBenchmark: 0.048, recentTxn: "Menara TM @ RM 1,380/sqft (Dec 2024)" },
  AXIS_SHAH:    { compPsf:  680, yieldBenchmark: 0.055, recentTxn: "Citaglobal IDC @ RM 650/sqft (Nov 2024)" },
  ALAQAR_MED:   { compPsf:  900, yieldBenchmark: 0.052, recentTxn: "Pantai Hospital Assets @ RM 870/sqft (Q3 2024)" },
  PAVILION_RTL: { compPsf: 1700, yieldBenchmark: 0.042, recentTxn: "Mid Valley Megamall tranche @ RM 1,720/sqft (Jan 2025)" },
  SUNWAY_LOG:   { compPsf:  420, yieldBenchmark: 0.062, recentTxn: "Mapletree Shah Alam @ RM 410/sqft (Q4 2024)" },
};

// ── Sector Palette ─────────────────────────────────────────────────────────────
const SECTOR_COLORS = {
  "Grade A Office":          "#0ea5e9",
  "Industrial / Data Centre":"#6366f1",
  "Healthcare":              "#10b981",
  "Retail":                  "#f59e0b",
  "Logistics / Industrial":  "#f97316",
};

// ── Sensor Simulation ───────────────────────────────────────────────────────────
function generateSensorData(seed) {
  return {
    structural:   Math.max(0.45, Math.min(1, 0.78 + Math.sin(seed * 0.07) * 0.12)),
    environmental:Math.max(0.50, Math.min(1, 0.82 + Math.cos(seed * 0.05) * 0.09)),
    occupancy:    Math.max(0.30, Math.min(1, 0.71 + Math.sin(seed * 0.11) * 0.15)),
    electrical:   Math.max(0.55, Math.min(1, 0.88 + Math.cos(seed * 0.09) * 0.07)),
    hvac:         Math.max(0.40, Math.min(1, 0.75 + Math.sin(seed * 0.13) * 0.13)),
  };
}

function computeHealthFactor(sensors) {
  const { structural, environmental, occupancy, electrical, hvac } = sensors;
  return (structural * 0.30 + environmental * 0.20 + occupancy * 0.20 + electrical * 0.15 + hvac * 0.15);
}

function computeCI(sensors) {
  const vals = Object.values(sensors);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
  return Math.max(0.5, 1 - variance * 8);
}

function computeRTPMV(govtVal, health, ci, override) {
  const base = override > 0 ? override : govtVal;
  return base * health * (0.85 + ci * 0.15);
}

// ── REIT Helpers ────────────────────────────────────────────────────────────────
function computeGRI(prop) {
  const inc = INCOME_DATA[prop.id];
  return inc.monthlyRatePerSqft * prop.area * inc.occupancyRate * 12;
}

function computeNPI(prop) {
  const inc = INCOME_DATA[prop.id];
  const gri = computeGRI(prop);
  return gri * (1 - inc.opexRatio);
}

function gearColor(v) {
  if (v < 0.35) return "#10b981";
  if (v < 0.45) return "#f59e0b";
  return "#ef4444";
}

function waleColor(v) {
  if (v >= 5)   return "#10b981";
  if (v >= 3)   return "#f59e0b";
  return "#ef4444";
}

// ── Formatters ─────────────────────────────────────────────────────────────────
const fmt = {
  rm:  (v) => `RM ${(v / 1_000_000).toFixed(2)}M`,
  psf: (v) => `RM ${v.toFixed(0)}/sqft`,
  pct: (v) => `${(v * 100).toFixed(1)}%`,
  sen: (v) => `${v.toFixed(2)} sen`,
  num: (v) => v.toLocaleString(),
  y:   (v) => `${v.toFixed(1)} yrs`,
};

// ── Sub-components ─────────────────────────────────────────────────────────────
function ProgressBar({ value, max = 1, color = "#0ea5e9", height = 6 }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ background: "#1e293b", borderRadius: 4, height, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.4s" }} />
    </div>
  );
}

function SectionHeader({ children }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: 1.2,
                  textTransform: "uppercase", marginBottom: 8, marginTop: 16 }}>
      {children}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8,
                  padding: "14px 16px", ...style }}>
      {children}
    </div>
  );
}

function InfoRow({ label, value, highlight = false }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "4px 0", borderBottom: "1px solid #1e293b" }}>
      <span style={{ fontSize: 11, color: "#64748b" }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 600, color: highlight ? "#0ea5e9" : "#e2e8f0" }}>{value}</span>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function TwinValREIT() {
  const [tick, setTick]         = useState(0);
  const [running, setRunning]   = useState(true);
  const [activeTab, setActiveTab] = useState("portfolio");
  const [selectedProp, setSelectedProp] = useState(PROPERTIES[0].id);
  const [overrides, setOverrides] = useState({});
  const [auditLog, setAuditLog]   = useState([]);
  const [history, setHistory]     = useState({});
  const intervalRef = useRef(null);

  // Tick engine
  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(() => setTick(t => t + 1), 2000);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [running]);

  // Build live state per property
  const propStates = PROPERTIES.map(p => {
    const sensors = generateSensorData(tick + p.id.charCodeAt(0));
    const health  = computeHealthFactor(sensors);
    const ci      = computeCI(sensors);
    const rtpmv   = computeRTPMV(p.govtVal, health, ci, overrides[p.id] || 0);
    return { ...p, sensors, health, ci, rtpmv };
  });

  // History accumulation
  useEffect(() => {
    setHistory(prev => {
      const next = { ...prev };
      propStates.forEach(ps => {
        const arr = prev[ps.id] || [];
        next[ps.id] = [...arr.slice(-59), { tick, rtpmv: ps.rtpmv, health: ps.health, ci: ps.ci }];
      });
      return next;
    });
  }, [tick]);

  // REIT-level metrics
  const cs           = CAPITAL_STRUCTURE;
  const totalGRI     = PROPERTIES.reduce((s, p) => s + computeGRI(p), 0);
  const totalNPI     = PROPERTIES.reduce((s, p) => s + computeNPI(p), 0);
  const interestExp  = cs.totalDebt * cs.avgCostOfDebt;
  const mgmtFee      = totalNPI * cs.managementFeeRate;
  const trusteeFee   = totalNPI * cs.trusteeFeeRate;
  const distributable= Math.max(0, totalNPI - interestExp - mgmtFee - trusteeFee);
  const dpu          = (distributable / cs.unitsOutstanding) * 100;   // sen
  const portfolioRTPMV = propStates.reduce((s, p) => s + p.rtpmv, 0);
  const totalArea    = PROPERTIES.reduce((s, p) => s + p.area, 0);
  const portfolioWALE = PROPERTIES.reduce((s, p) =>
    s + LEASE_PROFILES[p.id].wale * p.area, 0) / totalArea;
  const netAssets    = portfolioRTPMV - cs.totalDebt;
  const navPerUnit   = netAssets / cs.unitsOutstanding;
  const gearing      = cs.totalDebt / portfolioRTPMV;
  const pNavRatio    = cs.marketUnitPrice / navPerUnit;
  const npiYield     = totalNPI / portfolioRTPMV;

  const activeProp = propStates.find(p => p.id === selectedProp) || propStates[0];

  // Override handler
  const handleOverride = useCallback((id, val) => {
    const num = parseFloat(val);
    if (isNaN(num) || num < 0) return;
    setOverrides(prev => ({ ...prev, [id]: num }));
    setAuditLog(prev => [...prev, {
      ts: new Date().toISOString(),
      action: `Manual override: ${id} → RM ${(num / 1_000_000).toFixed(2)}M`,
    }]);
  }, []);

  // ── Styles ──────────────────────────────────────────────────────────────────
  const S = {
    app:  { fontFamily: "'SF Mono', 'Fira Code', monospace", background: "#020617",
            color: "#e2e8f0", minHeight: "100vh", padding: "16px 20px", fontSize: 12 },
    hdr:  { display: "flex", justifyContent: "space-between", alignItems: "center",
            borderBottom: "1px solid #1e293b", paddingBottom: 12, marginBottom: 16 },
    logo: { fontSize: 18, fontWeight: 700, color: "#0ea5e9", letterSpacing: 1 },
    badge:{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 4,
            padding: "3px 8px", fontSize: 10, color: "#64748b" },
    kpiRow:{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 16 },
    kpiBox:{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8,
             padding: "10px 14px" },
    kpiLbl:{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.8 },
    kpiVal:{ fontSize: 16, fontWeight: 700, color: "#0ea5e9", marginTop: 2 },
    kpiSub:{ fontSize: 10, color: "#475569", marginTop: 2 },
    tabs: { display: "flex", gap: 2, borderBottom: "1px solid #1e293b", marginBottom: 16, flexWrap: "wrap" },
    tab:  (active) => ({
      padding: "7px 14px", cursor: "pointer", fontSize: 11, fontWeight: 600,
      borderBottom: active ? "2px solid #0ea5e9" : "2px solid transparent",
      color: active ? "#0ea5e9" : "#64748b",
      background: "transparent", border: "none", outline: "none",
    }),
    grid2:{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
    grid3:{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 },
  };

  const TABS = [
    { id: "portfolio",  label: "Portfolio" },
    { id: "income",     label: "Income & DPU" },
    { id: "leases",     label: "Tenants & Leases" },
    { id: "capital",    label: "Capital Structure" },
    { id: "capex",      label: "Capex" },
    { id: "detail",     label: "Property Detail" },
    { id: "exchange",   label: "Exchange" },
    { id: "override",   label: "Override" },
    { id: "audit",      label: "Audit Log" },
  ];

  // ── Sector donut data ────────────────────────────────────────────────────────
  const sectorData = Object.entries(
    PROPERTIES.reduce((acc, p) => {
      acc[p.type] = (acc[p.type] || 0) + p.govtVal;
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={S.app}>

      {/* Header */}
      <div style={S.hdr}>
        <div>
          <div style={S.logo}>TwinVal REIT Intelligence</div>
          <div style={{ fontSize: 10, color: "#475569", marginTop: 2 }}>
            Real-Time Digital Twin · Kuala Lumpur &amp; Selangor, Malaysia
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ ...S.badge, color: running ? "#10b981" : "#f59e0b" }}>
            {running ? "● LIVE" : "⏸ PAUSED"}
          </span>
          <button onClick={() => setRunning(r => !r)}
            style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 4,
                     color: "#e2e8f0", padding: "4px 10px", cursor: "pointer", fontSize: 11 }}>
            {running ? "Pause" : "Resume"}
          </button>
          <span style={S.badge}>Tick #{tick}</span>
        </div>
      </div>

      {/* KPI Bar */}
      <div style={S.kpiRow}>
        <div style={S.kpiBox}>
          <div style={S.kpiLbl}>Portfolio RTPMV</div>
          <div style={S.kpiVal}>{fmt.rm(portfolioRTPMV)}</div>
          <div style={S.kpiSub}>NPI Yield {fmt.pct(npiYield)}</div>
        </div>
        <div style={S.kpiBox}>
          <div style={S.kpiLbl}>NAV / Unit</div>
          <div style={{ ...S.kpiVal, color: "#10b981" }}>RM {navPerUnit.toFixed(4)}</div>
          <div style={S.kpiSub}>P/NAV {pNavRatio.toFixed(2)}x · Mkt RM {cs.marketUnitPrice.toFixed(2)}</div>
        </div>
        <div style={S.kpiBox}>
          <div style={S.kpiLbl}>Annualised DPU</div>
          <div style={{ ...S.kpiVal, color: "#a78bfa" }}>{fmt.sen(dpu)}</div>
          <div style={S.kpiSub}>Distributable {fmt.rm(distributable)}</div>
        </div>
        <div style={S.kpiBox}>
          <div style={S.kpiLbl}>Gearing Ratio</div>
          <div style={{ ...S.kpiVal, color: gearColor(gearing) }}>{fmt.pct(gearing)}</div>
          <div style={S.kpiSub}>SC Limit 50% · Debt {fmt.rm(cs.totalDebt)}</div>
        </div>
        <div style={S.kpiBox}>
          <div style={S.kpiLbl}>Portfolio WALE</div>
          <div style={{ ...S.kpiVal, color: waleColor(portfolioWALE) }}>{fmt.y(portfolioWALE)}</div>
          <div style={S.kpiSub}>5 assets · {fmt.num(totalArea)} sqft</div>
        </div>
      </div>

      {/* Tabs */}
      <div style={S.tabs}>
        {TABS.map(t => (
          <button key={t.id} style={S.tab(activeTab === t.id)} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── TAB: PORTFOLIO ─────────────────────────────────────────────────────── */}
      {activeTab === "portfolio" && (
        <div>
          <div style={S.grid2}>
            {/* Asset table */}
            <Card>
              <SectionHeader>Asset Summary</SectionHeader>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ color: "#64748b", fontSize: 10, textAlign: "left" }}>
                    <th style={{ padding: "4px 6px" }}>Property</th>
                    <th style={{ padding: "4px 6px" }}>Type</th>
                    <th style={{ padding: "4px 6px", textAlign: "right" }}>RTPMV</th>
                    <th style={{ padding: "4px 6px", textAlign: "right" }}>Health</th>
                    <th style={{ padding: "4px 6px", textAlign: "right" }}>Occ%</th>
                    <th style={{ padding: "4px 6px", textAlign: "right" }}>WALE</th>
                  </tr>
                </thead>
                <tbody>
                  {propStates.map(p => (
                    <tr key={p.id} onClick={() => { setSelectedProp(p.id); setActiveTab("detail"); }}
                      style={{ cursor: "pointer", borderTop: "1px solid #1e293b" }}>
                      <td style={{ padding: "5px 6px", color: "#0ea5e9", fontSize: 11 }}>{p.name}</td>
                      <td style={{ padding: "5px 6px", color: "#64748b", fontSize: 10 }}>{p.type}</td>
                      <td style={{ padding: "5px 6px", textAlign: "right" }}>{fmt.rm(p.rtpmv)}</td>
                      <td style={{ padding: "5px 6px", textAlign: "right", color: p.health > 0.8 ? "#10b981" : "#f59e0b" }}>
                        {fmt.pct(p.health)}
                      </td>
                      <td style={{ padding: "5px 6px", textAlign: "right" }}>
                        {fmt.pct(INCOME_DATA[p.id].occupancyRate)}
                      </td>
                      <td style={{ padding: "5px 6px", textAlign: "right", color: waleColor(LEASE_PROFILES[p.id].wale) }}>
                        {fmt.y(LEASE_PROFILES[p.id].wale)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            {/* Sector donut */}
            <Card>
              <SectionHeader>Sector Diversification (by Govt Valuation)</SectionHeader>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={sectorData} cx="50%" cy="50%" outerRadius={80} innerRadius={45}
                    dataKey="value" nameKey="name" paddingAngle={2}>
                    {sectorData.map(entry => (
                      <Cell key={entry.name} fill={SECTOR_COLORS[entry.name] || "#334155"} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => fmt.rm(v)} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 10 }} />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Portfolio RTPMV history */}
          <Card style={{ marginTop: 12 }}>
            <SectionHeader>Live RTPMV History (60 ticks)</SectionHeader>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={Array.from({ length: Math.max(...PROPERTIES.map(p => (history[p.id] || []).length)) }, (_, i) => {
                const pt = { tick: i };
                PROPERTIES.forEach(p => {
                  const h = history[p.id] || [];
                  if (h[i]) pt[p.id] = h[i].rtpmv / 1_000_000;
                });
                return pt;
              })}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="tick" tick={{ fontSize: 9, fill: "#475569" }} />
                <YAxis tick={{ fontSize: 9, fill: "#475569" }} tickFormatter={v => `${v.toFixed(0)}M`} />
                <Tooltip formatter={(v) => [`RM ${v.toFixed(2)}M`]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                {PROPERTIES.map(p => (
                  <Line key={p.id} dataKey={p.id} name={p.name} dot={false} strokeWidth={1.5}
                    stroke={Object.values(SECTOR_COLORS)[PROPERTIES.indexOf(p) % 5]} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── TAB: INCOME & DPU ─────────────────────────────────────────────────── */}
      {activeTab === "income" && (
        <div>
          <div style={S.grid3}>
            <Card>
              <SectionHeader>Income Waterfall</SectionHeader>
              {[
                { label: "Gross Rental Income (GRI)", value: totalGRI, color: "#0ea5e9" },
                { label: "Operating Expenses",        value: -(totalGRI - totalNPI), color: "#ef4444" },
                { label: "Net Property Income (NPI)", value: totalNPI, color: "#10b981" },
                { label: "Interest Expense",          value: -interestExp, color: "#f97316" },
                { label: "Management Fee",            value: -mgmtFee, color: "#f59e0b" },
                { label: "Trustee Fee",               value: -trusteeFee, color: "#f59e0b" },
                { label: "Distributable Income",      value: distributable, color: "#a78bfa" },
              ].map(row => (
                <div key={row.label} style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 10, color: "#94a3b8" }}>{row.label}</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color: row.color }}>
                      {row.value < 0 ? `-${fmt.rm(Math.abs(row.value))}` : fmt.rm(row.value)}
                    </span>
                  </div>
                  <ProgressBar value={Math.abs(row.value)} max={totalGRI} color={row.color} />
                </div>
              ))}
            </Card>

            <Card>
              <SectionHeader>NPI by Property</SectionHeader>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={PROPERTIES.map(p => ({ name: p.name.split(" ")[0], npi: computeNPI(p) / 1_000_000 }))} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" tick={{ fontSize: 9, fill: "#475569" }} tickFormatter={v => `${v.toFixed(0)}M`} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: "#94a3b8" }} width={55} />
                  <Tooltip formatter={(v) => [`RM ${v.toFixed(2)}M`]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                  <Bar dataKey="npi" name="NPI" radius={[0, 4, 4, 0]}>
                    {PROPERTIES.map((p, i) => <Cell key={p.id} fill={Object.values(SECTOR_COLORS)[i % 5]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <SectionHeader>DPU Metrics</SectionHeader>
              <InfoRow label="Gross Rental Income" value={fmt.rm(totalGRI)} />
              <InfoRow label="Net Property Income" value={fmt.rm(totalNPI)} highlight />
              <InfoRow label="NPI Margin" value={fmt.pct(totalNPI / totalGRI)} />
              <InfoRow label="Interest Expense" value={fmt.rm(interestExp)} />
              <InfoRow label="Management Fee" value={fmt.rm(mgmtFee)} />
              <InfoRow label="Trustee Fee" value={fmt.rm(trusteeFee)} />
              <InfoRow label="Distributable Income" value={fmt.rm(distributable)} highlight />
              <InfoRow label="Units Outstanding" value={fmt.num(cs.unitsOutstanding)} />
              <InfoRow label="DPU (Annualised)" value={fmt.sen(dpu)} highlight />
              <InfoRow label="Dividend Yield" value={fmt.pct(dpu / 100 / cs.marketUnitPrice)} />

              <SectionHeader>Occupancy by Property</SectionHeader>
              {PROPERTIES.map(p => (
                <div key={p.id} style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 10, color: "#94a3b8" }}>{p.name.split(" ").slice(0, 2).join(" ")}</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color: INCOME_DATA[p.id].occupancyRate >= 0.9 ? "#10b981" : "#f59e0b" }}>
                      {fmt.pct(INCOME_DATA[p.id].occupancyRate)}
                    </span>
                  </div>
                  <ProgressBar value={INCOME_DATA[p.id].occupancyRate} color={INCOME_DATA[p.id].occupancyRate >= 0.9 ? "#10b981" : "#f59e0b"} />
                </div>
              ))}
            </Card>
          </div>
        </div>
      )}

      {/* ── TAB: TENANTS & LEASES ─────────────────────────────────────────────── */}
      {activeTab === "leases" && (
        <div>
          <div style={S.grid2}>
            {/* WALE by property */}
            <Card>
              <SectionHeader>WALE by Property (years)</SectionHeader>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={PROPERTIES.map(p => ({ name: p.name.split(" ")[0], wale: LEASE_PROFILES[p.id].wale }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#475569" }} />
                  <YAxis tick={{ fontSize: 9, fill: "#475569" }} domain={[0, 10]} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                  <ReferenceLine y={3} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "Min 3yr", fill: "#ef4444", fontSize: 9 }} />
                  <Bar dataKey="wale" name="WALE" radius={[4, 4, 0, 0]}>
                    {PROPERTIES.map(p => <Cell key={p.id} fill={waleColor(LEASE_PROFILES[p.id].wale)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Lease expiry profile */}
            <Card>
              <SectionHeader>Lease Expiry Profile — {activeProp.name}</SectionHeader>
              <div style={{ marginBottom: 8, fontSize: 10, color: "#64748b" }}>
                Select property in header KPI to change view
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={LEASE_PROFILES[selectedProp].expiry}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="year" tick={{ fontSize: 9, fill: "#475569" }} />
                  <YAxis tick={{ fontSize: 9, fill: "#475569" }} tickFormatter={v => `${v}%`} />
                  <Tooltip formatter={(v) => [`${v}% of NLA`]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                  <Bar dataKey="pct" name="NLA %" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Tenant concentration per property */}
          <SectionHeader>Tenant Concentration Risk</SectionHeader>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
            {PROPERTIES.map(p => {
              const tenants = LEASE_PROFILES[p.id].tenants;
              const topTenant = tenants[0];
              const concentration = topTenant.pct / 100;
              return (
                <Card key={p.id}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 8 }}>
                    {p.name.split(" ").slice(0, 2).join(" ")}
                  </div>
                  {tenants.map(t => (
                    <div key={t.name} style={{ marginBottom: 6 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                        <span style={{ fontSize: 9, color: "#64748b", maxWidth: "60%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.name}</span>
                        <span style={{ fontSize: 9, fontWeight: 600, color: t.pct > 40 ? "#f59e0b" : "#94a3b8" }}>{t.pct}%</span>
                      </div>
                      <ProgressBar value={t.pct} max={100} color={t.pct > 40 ? "#f59e0b" : "#0ea5e9"} height={4} />
                    </div>
                  ))}
                  <div style={{ marginTop: 8, fontSize: 9, color: concentration > 0.4 ? "#f59e0b" : "#10b981" }}>
                    {concentration > 0.4 ? "⚠ Concentration Risk" : "✓ Diversified"}
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {/* ── TAB: CAPITAL STRUCTURE ────────────────────────────────────────────── */}
      {activeTab === "capital" && (
        <div>
          <div style={S.grid3}>
            {/* Gearing gauge */}
            <Card>
              <SectionHeader>Gearing Ratio</SectionHeader>
              <div style={{ textAlign: "center", padding: "20px 0" }}>
                <div style={{ fontSize: 48, fontWeight: 700, color: gearColor(gearing) }}>
                  {fmt.pct(gearing)}
                </div>
                <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
                  SC Malaysia limit: 50%
                </div>
                <div style={{ marginTop: 16 }}>
                  <ProgressBar value={gearing} max={0.5} color={gearColor(gearing)} height={10} />
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 4 }}>
                    <span>0%</span><span style={{ color: "#10b981" }}>35%</span><span style={{ color: "#f59e0b" }}>45%</span><span style={{ color: "#ef4444" }}>50%</span>
                  </div>
                </div>
              </div>
              <InfoRow label="Total Debt" value={fmt.rm(cs.totalDebt)} />
              <InfoRow label="Portfolio Value" value={fmt.rm(portfolioRTPMV)} />
              <InfoRow label="Avg Cost of Debt" value={fmt.pct(cs.avgCostOfDebt)} />
              <InfoRow label="Interest Expense (pa)" value={fmt.rm(interestExp)} />
              <InfoRow label="Headroom to Limit" value={fmt.rm(portfolioRTPMV * 0.5 - cs.totalDebt)} highlight />
            </Card>

            {/* NAV Bridge */}
            <Card>
              <SectionHeader>NAV Bridge</SectionHeader>
              {[
                { label: "Gross Portfolio Value",  value: portfolioRTPMV,          color: "#0ea5e9" },
                { label: "Less: Total Debt",        value: -cs.totalDebt,            color: "#ef4444" },
                { label: "Net Asset Value (NAV)",   value: netAssets,                color: "#10b981" },
                { label: "Units Outstanding (M)",   value: cs.unitsOutstanding / 1e6, raw: true, color: "#94a3b8" },
                { label: "NAV per Unit (RM)",       value: navPerUnit,               raw: true, color: "#a78bfa" },
                { label: "Market Price (RM)",       value: cs.marketUnitPrice,       raw: true, color: "#64748b" },
                { label: "P/NAV",                   value: pNavRatio,                raw: true, color: pNavRatio < 1 ? "#f59e0b" : "#10b981" },
              ].map(row => (
                <InfoRow key={row.label} label={row.label}
                  value={row.raw
                    ? (row.value < 10 ? row.value.toFixed(4) : row.value.toFixed(1))
                    : (row.value < 0 ? `-${fmt.rm(Math.abs(row.value))}` : fmt.rm(row.value))}
                  highlight={row.color === "#10b981" || row.color === "#a78bfa"} />
              ))}
            </Card>

            {/* Debt maturity */}
            <Card>
              <SectionHeader>Debt Maturity Profile</SectionHeader>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={cs.debtMaturity}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="year" tick={{ fontSize: 9, fill: "#475569" }} />
                  <YAxis tick={{ fontSize: 9, fill: "#475569" }} tickFormatter={v => `${(v / 1e6).toFixed(0)}M`} />
                  <Tooltip formatter={(v) => [fmt.rm(v)]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                  <Bar dataKey="amount" name="Debt Due" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <InfoRow label="Mgmt Fee Rate" value={fmt.pct(cs.managementFeeRate)} />
              <InfoRow label="Trustee Fee Rate" value={fmt.pct(cs.trusteeFeeRate)} />
            </Card>
          </div>
        </div>
      )}

      {/* ── TAB: CAPEX ────────────────────────────────────────────────────────── */}
      {activeTab === "capex" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 12 }}>
            {PROPERTIES.map(p => {
              const cx = CAPEX_BUDGET[p.id];
              const totalBudget = cx.sustainingBudget + cx.enhancementBudget;
              const pctSpent = cx.ytdSpend / totalBudget;
              return (
                <Card key={p.id}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 8 }}>
                    {p.name.split(" ").slice(0, 2).join(" ")}
                  </div>
                  <div style={{ fontSize: 11, color: "#0ea5e9", marginBottom: 4 }}>
                    YTD: {fmt.rm(cx.ytdSpend)}
                  </div>
                  <ProgressBar value={cx.ytdSpend} max={totalBudget} color={pctSpent > 0.85 ? "#f97316" : "#0ea5e9"} height={8} />
                  <div style={{ fontSize: 9, color: "#475569", marginTop: 4, marginBottom: 8 }}>
                    {fmt.pct(pctSpent)} of {fmt.rm(totalBudget)} budget
                  </div>
                  <InfoRow label="Sustaining" value={fmt.rm(cx.sustainingBudget)} />
                  <InfoRow label="Enhancement" value={fmt.rm(cx.enhancementBudget)} />
                  {cx.sensorFlags.length > 0 && (
                    <div style={{ marginTop: 8, padding: "6px 8px", background: "#1c1109", border: "1px solid #7c2d12", borderRadius: 4, fontSize: 9, color: "#f97316" }}>
                      ⚠ {cx.sensorFlags.join("; ")}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>

          <Card>
            <SectionHeader>Capex Pipeline</SectionHeader>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ color: "#64748b", fontSize: 10, textAlign: "left" }}>
                  <th style={{ padding: "4px 8px" }}>Property</th>
                  <th style={{ padding: "4px 8px" }}>Initiative</th>
                  <th style={{ padding: "4px 8px" }}>Budget</th>
                  <th style={{ padding: "4px 8px" }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {PROPERTIES.flatMap(p =>
                  CAPEX_BUDGET[p.id].pipeline.map(item => (
                    <tr key={`${p.id}-${item}`} style={{ borderTop: "1px solid #1e293b" }}>
                      <td style={{ padding: "5px 8px", color: "#0ea5e9", fontSize: 11 }}>{p.name.split(" ")[0]}</td>
                      <td style={{ padding: "5px 8px", fontSize: 11 }}>{item.replace(/ Q[0-9]/, "")}</td>
                      <td style={{ padding: "5px 8px", fontSize: 11, color: "#94a3b8" }}>—</td>
                      <td style={{ padding: "5px 8px", fontSize: 10 }}>
                        <span style={{ background: "#0f2027", border: "1px solid #0ea5e9", borderRadius: 3, padding: "1px 6px", color: "#0ea5e9" }}>
                          {item.match(/Q\d/)?.[0] || "FY25"}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ── TAB: PROPERTY DETAIL ─────────────────────────────────────────────── */}
      {activeTab === "detail" && (
        <div>
          {/* Property selector */}
          <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
            {PROPERTIES.map(p => (
              <button key={p.id} onClick={() => setSelectedProp(p.id)}
                style={{ padding: "5px 12px", cursor: "pointer", fontSize: 11, borderRadius: 4,
                  background: selectedProp === p.id ? "#0ea5e9" : "#1e293b",
                  color: selectedProp === p.id ? "#fff" : "#94a3b8",
                  border: "1px solid " + (selectedProp === p.id ? "#0ea5e9" : "#334155") }}>
                {p.name.split(" ")[0]}
              </button>
            ))}
          </div>

          <div style={S.grid3}>
            {/* Asset info */}
            <Card>
              <SectionHeader>Asset Information</SectionHeader>
              <InfoRow label="Property ID" value={activeProp.id} />
              <InfoRow label="Type" value={activeProp.type} />
              <InfoRow label="Location" value={activeProp.location} />
              <InfoRow label="GFA" value={fmt.num(activeProp.area) + " sqft"} />
              <InfoRow label="Govt Valuation" value={fmt.rm(activeProp.govtVal)} />
              <InfoRow label="RTPMV (live)" value={fmt.rm(activeProp.rtpmv)} highlight />
              <InfoRow label="Health Factor" value={fmt.pct(activeProp.health)} />
              <InfoRow label="Confidence Index" value={fmt.pct(activeProp.ci)} />
              <InfoRow label="Occupancy Rate" value={fmt.pct(INCOME_DATA[activeProp.id].occupancyRate)} />
              <InfoRow label="Monthly Rent/sqft" value={`RM ${INCOME_DATA[activeProp.id].monthlyRatePerSqft.toFixed(2)}`} />
              <InfoRow label="GRI (Annual)" value={fmt.rm(computeGRI(activeProp))} />
              <InfoRow label="NPI (Annual)" value={fmt.rm(computeNPI(activeProp))} highlight />
              <InfoRow label="WALE" value={fmt.y(LEASE_PROFILES[activeProp.id].wale)} />
            </Card>

            {/* Market comparables */}
            <Card>
              <SectionHeader>Market Comparable Benchmarking</SectionHeader>
              {(() => {
                const comp = MARKET_COMPS[activeProp.id];
                const impliedPsf = activeProp.rtpmv / activeProp.area;
                const premDisc = (impliedPsf - comp.compPsf) / comp.compPsf;
                const impliedYield = computeNPI(activeProp) / activeProp.rtpmv;
                const yieldSpread = impliedYield - comp.yieldBenchmark;
                return (
                  <>
                    <InfoRow label="RTPMV / sqft (implied)" value={fmt.psf(impliedPsf)} highlight />
                    <InfoRow label="Comp Market PSF" value={fmt.psf(comp.compPsf)} />
                    <InfoRow label="Premium / (Discount)" value={`${premDisc >= 0 ? "+" : ""}${fmt.pct(premDisc)}`} />
                    <div style={{ marginTop: 8, marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 10, color: "#64748b" }}>vs Market PSF</span>
                        <span style={{ fontSize: 10, color: premDisc >= 0 ? "#10b981" : "#ef4444" }}>
                          {premDisc >= 0 ? "Above Market" : "Below Market"}
                        </span>
                      </div>
                      <ProgressBar value={Math.min(impliedPsf, comp.compPsf * 1.3)} max={comp.compPsf * 1.3}
                        color={premDisc >= 0 ? "#10b981" : "#ef4444"} height={8} />
                    </div>
                    <InfoRow label="NPI Yield (implied)" value={fmt.pct(impliedYield)} highlight />
                    <InfoRow label="Sector Yield Benchmark" value={fmt.pct(comp.yieldBenchmark)} />
                    <InfoRow label="Yield Spread vs Sector" value={`${yieldSpread >= 0 ? "+" : ""}${(yieldSpread * 100).toFixed(0)} bps`} />
                    <div style={{ marginTop: 12, padding: "8px 10px", background: "#0d1117", border: "1px solid #1e293b", borderRadius: 4 }}>
                      <div style={{ fontSize: 9, color: "#64748b", marginBottom: 4 }}>Recent Comparable Transaction</div>
                      <div style={{ fontSize: 10, color: "#94a3b8" }}>{comp.recentTxn}</div>
                    </div>
                  </>
                );
              })()}
            </Card>

            {/* Sensor radar */}
            <Card>
              <SectionHeader>Sensor Health Radar</SectionHeader>
              <ResponsiveContainer width="100%" height={200}>
                <RadarChart data={[
                  { subject: "Structural",    value: activeProp.sensors.structural * 100 },
                  { subject: "Environmental", value: activeProp.sensors.environmental * 100 },
                  { subject: "Occupancy",     value: activeProp.sensors.occupancy * 100 },
                  { subject: "Electrical",    value: activeProp.sensors.electrical * 100 },
                  { subject: "HVAC",          value: activeProp.sensors.hvac * 100 },
                ]}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 9, fill: "#64748b" }} />
                  <Radar name="Sensors" dataKey="value" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.2} />
                  <Tooltip formatter={(v) => [`${v.toFixed(1)}%`]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                </RadarChart>
              </ResponsiveContainer>

              <SectionHeader>RTPMV History</SectionHeader>
              <ResponsiveContainer width="100%" height={100}>
                <AreaChart data={(history[activeProp.id] || []).map((h, i) => ({ i, v: h.rtpmv / 1e6 }))}>
                  <defs>
                    <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="i" hide />
                  <YAxis hide domain={["auto", "auto"]} />
                  <Area type="monotone" dataKey="v" stroke="#0ea5e9" strokeWidth={1.5} fill="url(#grad)" dot={false} />
                  <Tooltip formatter={(v) => [`RM ${v.toFixed(2)}M`]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 10 }} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}

      {/* ── TAB: EXCHANGE ─────────────────────────────────────────────────────── */}
      {activeTab === "exchange" && (
        <div style={S.grid2}>
          {propStates.map(p => {
            const spread  = p.rtpmv * 0.005;
            const bid     = p.rtpmv - spread;
            const ask     = p.rtpmv + spread;
            const mid     = p.rtpmv;
            const depth   = Array.from({ length: 5 }, (_, i) => ({
              price:  (bid - i * spread * 0.4) / 1e6,
              bidVol: Math.round(15 - i * 2.5),
              askVol: Math.round(12 - i * 2),
            }));
            return (
              <Card key={p.id}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#0ea5e9" }}>{p.name}</div>
                    <div style={{ fontSize: 9, color: "#475569" }}>{p.type}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 14, fontWeight: 700 }}>{fmt.rm(mid)}</div>
                    <div style={{ fontSize: 9, color: "#64748b" }}>Mid RTPMV</div>
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 10 }}>
                  {[
                    { label: "BID", value: fmt.rm(bid), color: "#10b981" },
                    { label: "SPREAD", value: fmt.rm(spread * 2), color: "#64748b" },
                    { label: "ASK", value: fmt.rm(ask), color: "#ef4444" },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ textAlign: "center", background: "#020617", borderRadius: 4, padding: "6px 4px" }}>
                      <div style={{ fontSize: 9, color: "#475569" }}>{label}</div>
                      <div style={{ fontSize: 11, fontWeight: 700, color }}>{value}</div>
                    </div>
                  ))}
                </div>
                <SectionHeader>Order Book Depth</SectionHeader>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                  <thead>
                    <tr style={{ color: "#475569", fontSize: 9 }}>
                      <th style={{ textAlign: "left", padding: "2px 4px" }}>Bid Vol</th>
                      <th style={{ textAlign: "center", padding: "2px 4px" }}>Price (RM M)</th>
                      <th style={{ textAlign: "right", padding: "2px 4px" }}>Ask Vol</th>
                    </tr>
                  </thead>
                  <tbody>
                    {depth.map((d, i) => (
                      <tr key={i} style={{ borderTop: "1px solid #0f172a" }}>
                        <td style={{ color: "#10b981", padding: "2px 4px" }}>{d.bidVol}</td>
                        <td style={{ textAlign: "center", color: "#94a3b8", padding: "2px 4px" }}>{d.price.toFixed(2)}</td>
                        <td style={{ color: "#ef4444", textAlign: "right", padding: "2px 4px" }}>{d.askVol}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            );
          })}
        </div>
      )}

      {/* ── TAB: OVERRIDE ─────────────────────────────────────────────────────── */}
      {activeTab === "override" && (
        <div style={S.grid2}>
          {PROPERTIES.map(p => (
            <Card key={p.id}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#0ea5e9", marginBottom: 6 }}>{p.name}</div>
              <InfoRow label="Govt Valuation" value={fmt.rm(p.govtVal)} />
              <InfoRow label="Live RTPMV" value={fmt.rm(propStates.find(x => x.id === p.id)?.rtpmv || 0)} highlight />
              {overrides[p.id] ? (
                <div style={{ marginTop: 6, padding: "4px 8px", background: "#0d1b2a", borderRadius: 4, border: "1px solid #0ea5e9", fontSize: 10 }}>
                  Override active: {fmt.rm(overrides[p.id])}
                </div>
              ) : null}
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>Manual Valuation Override (RM)</div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input id={`ov-${p.id}`} type="number" placeholder={p.govtVal.toString()}
                    style={{ flex: 1, background: "#020617", border: "1px solid #334155", borderRadius: 4,
                             color: "#e2e8f0", padding: "5px 8px", fontSize: 11, outline: "none" }} />
                  <button onClick={() => {
                    const el = document.getElementById(`ov-${p.id}`);
                    if (el) handleOverride(p.id, el.value);
                  }}
                    style={{ background: "#0ea5e9", border: "none", borderRadius: 4, color: "#fff",
                             padding: "5px 12px", cursor: "pointer", fontSize: 11, fontWeight: 600 }}>
                    Apply
                  </button>
                  <button onClick={() => setOverrides(prev => { const n = { ...prev }; delete n[p.id]; return n; })}
                    style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 4,
                             color: "#94a3b8", padding: "5px 10px", cursor: "pointer", fontSize: 11 }}>
                    Clear
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ── TAB: AUDIT LOG ────────────────────────────────────────────────────── */}
      {activeTab === "audit" && (
        <Card>
          <SectionHeader>Audit Trail (G1 Hash Chain)</SectionHeader>
          {auditLog.length === 0 ? (
            <div style={{ color: "#475569", fontSize: 11, padding: "20px 0", textAlign: "center" }}>
              No audit events recorded yet. Apply overrides to generate entries.
            </div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ color: "#64748b", fontSize: 10, textAlign: "left" }}>
                  <th style={{ padding: "4px 8px" }}>Timestamp</th>
                  <th style={{ padding: "4px 8px" }}>Event</th>
                </tr>
              </thead>
              <tbody>
                {[...auditLog].reverse().map((entry, i) => (
                  <tr key={i} style={{ borderTop: "1px solid #1e293b" }}>
                    <td style={{ padding: "5px 8px", color: "#475569", fontSize: 10, whiteSpace: "nowrap" }}>
                      {new Date(entry.ts).toLocaleTimeString()}
                    </td>
                    <td style={{ padding: "5px 8px", fontSize: 11, color: "#0ea5e9" }}>{entry.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Live tick log */}
          <SectionHeader>Live System Events</SectionHeader>
          <div style={{ maxHeight: 200, overflowY: "auto" }}>
            {Array.from({ length: Math.min(tick, 20) }, (_, i) => tick - i).map(t => (
              <div key={t} style={{ display: "flex", gap: 12, padding: "3px 0", borderBottom: "1px solid #0f172a", fontSize: 10 }}>
                <span style={{ color: "#475569", minWidth: 60 }}>Tick #{t}</span>
                <span style={{ color: "#10b981" }}>RTPMV_UPDATE</span>
                <span style={{ color: "#64748b" }}>all 5 assets · hash verified · CI avg {fmt.pct(propStates.reduce((s, p) => s + p.ci, 0) / propStates.length)}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Footer */}
      <div style={{ marginTop: 24, borderTop: "1px solid #1e293b", paddingTop: 10, display: "flex",
                    justifyContent: "space-between", color: "#334155", fontSize: 9 }}>
        <span>TwinVal REIT Intelligence · ASHRAE GEPIII Profiles · Kuala Lumpur, Malaysia</span>
        <span>Patent Pending · Real-Time Digital Twin Valuation · v2.0</span>
      </div>
    </div>
  );
}
