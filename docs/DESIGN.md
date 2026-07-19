# NeuroAegis — Visual & UX Design System

> **Canonical reference for every visual and interaction decision in the NeuroAegis dashboard.**
> All contributors — design, frontend, QA — must treat this document as the single source of truth.

---

## 1 · Design Language

| Attribute | Value |
|-----------|-------|
| **Mood** | Ultra-premium · Futuristic · Cinematic · Scientific · Minimal-but-information-rich |
| **Theme** | Permanent dark mode — no light-mode variant exists |
| **Surface** | Glassmorphic panels with holographic accents |
| **Archetype** | Secret AI research lab / mission control |

### 1.1 Inspiration Board

| Source | What We Borrow |
|--------|---------------|
| Apple Vision Pro | Spatial glass layering, depth-of-field blur, typographic restraint |
| Neuralink | Clinical precision, neural imagery, monochrome + single-accent palette |
| OpenAI | Generous white-space, understated confidence, API-grade information density |
| IBM Watson | Data-dense panels that remain scannable, scientific credibility |
| Iron Man / JARVIS | Holographic HUD rings, floating translucent cards, ambient glow |
| Tesla UI | Full-bleed dark canvas, real-time telemetry aesthetic, zero-chrome controls |
| Cyberpunk 2077 HUD | Neon edge lighting, scan-line textures, futuristic iconography |
| NASA Mission Control | Multi-panel telemetry layout, status-at-a-glance discipline |

> [!IMPORTANT]
> The resulting feel must be **cinematic yet trustworthy**. Every decoration must justify itself by improving scanability or reinforcing the scientific context of seizure detection.

---

## 2 · Color Tokens

All colors are exposed as CSS custom properties on `:root` and as Tailwind `extend.colors` entries.

| Token | Hex | Swatch | Usage |
|-------|-----|--------|-------|
| `--bg-1` | `#040814` | ![#040814](https://via.placeholder.com/16/040814/040814.png) | Deepest background — page canvas, modal overlays |
| `--bg-2` | `#07111D` | ![#07111D](https://via.placeholder.com/16/07111D/07111D.png) | Card interiors, sidebar surface |
| `--bg-3` | `#0B1625` | ![#0B1625](https://via.placeholder.com/16/0B1625/0B1625.png) | Elevated card fill, hover-state backgrounds |
| `--accent-primary` | `#00E5FF` | ![#00E5FF](https://via.placeholder.com/16/00E5FF/00E5FF.png) | **Electric Cyan** — primary interactive elements, active nav, glow source |
| `--accent-secondary` | `#4B7DFF` | ![#4B7DFF](https://via.placeholder.com/16/4B7DFF/4B7DFF.png) | **Royal Blue** — secondary buttons, links, chart accents |
| `--accent-highlight` | `#8B5CF6` | ![#8B5CF6](https://via.placeholder.com/16/8B5CF6/8B5CF6.png) | **Neon Purple** — badges, highlighted values, holographic ring tints |
| `--state-success` | `#00FFA3` | ![#00FFA3](https://via.placeholder.com/16/00FFA3/00FFA3.png) | Positive classification, healthy signal, success toasts |
| `--state-warning` | `#FFB020` | ![#FFB020](https://via.placeholder.com/16/FFB020/FFB020.png) | Uncertain predictions, threshold alerts, caution badges |
| `--state-danger` | `#FF4D6D` | ![#FF4D6D](https://via.placeholder.com/16/FF4D6D/FF4D6D.png) | Seizure detected, error states, destructive actions |
| `--text-primary` | `#F8FAFC` | ![#F8FAFC](https://via.placeholder.com/16/F8FAFC/F8FAFC.png) | Headings, body text, primary labels |
| `--text-secondary` | `#94A3B8` | ![#94A3B8](https://via.placeholder.com/16/94A3B8/94A3B8.png) | Captions, timestamps, supporting copy, axis labels |

> [!NOTE]
> Never use raw hex values in component code. Always reference tokens so palette changes propagate globally.

---

## 3 · Background Treatment

The background is **never flat**. Every page canvas must layer the following:

```
┌─────────────────────────────────────────────┐
│  Layer 4 — Ambient particles (CSS / Canvas) │  opacity: 0.25, slow drift
│  Layer 3 — Subtle noise texture (PNG)       │  opacity: 0.02–0.03, blend: overlay
│  Layer 2 — Neural-network SVG pattern       │  opacity: 0.04, tiled, faint cyan tint
│  Layer 1 — Radial gradient                  │  deep navy center → --bg-1 edges
│  Layer 0 — Solid --bg-1                     │  fallback
└─────────────────────────────────────────────┘
```

### 3.1 Radial Gradient

```css
background: radial-gradient(
  ellipse at 50% 0%,
  rgba(11, 22, 37, 1) 0%,       /* --bg-3 */
  rgba(7, 17, 29, 1) 40%,       /* --bg-2 */
  rgba(4, 8, 20, 1) 100%        /* --bg-1 */
);
```

### 3.2 Noise Overlay

- Format: 200 × 200 px seamless PNG, monochrome Gaussian noise.
- CSS: `background-image: url('/textures/noise.png'); opacity: 0.02; mix-blend-mode: overlay; pointer-events: none;`
- Noise intensity must stay between **2 %–3 %** opacity — visible only on close inspection.

### 3.3 Neural-Network Pattern

A tiled SVG of faint interconnected nodes and edges, tinted `rgba(0, 229, 255, 0.04)`. The pattern reinforces the neuroscience context without competing with content.

### 3.4 Ambient Particles

Tiny luminous dots (1–2 px) drifting slowly upward. Rendered via lightweight Canvas or CSS `@keyframes`. Max **40 particles** on-screen to avoid performance drag.

---

## 4 · Glass Card Specification

Every panel, card, and modal in the system uses the glassmorphic surface defined below.

### 4.1 Base Styles

```css
.glass-card {
  /* ── Surface ── */
  background: rgba(11, 22, 37, 0.6);             /* --bg-3 at 60 % */
  backdrop-filter: blur(30px);
  -webkit-backdrop-filter: blur(30px);            /* Safari */

  /* ── Shape ── */
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.08);

  /* ── Glow ── */
  box-shadow:
    0 0 20px rgba(0, 229, 255, 0.05),             /* soft outer cyan glow */
    inset 0 1px 0 rgba(255, 255, 255, 0.06);      /* inner top-edge reflection */

  /* ── Transitions ── */
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
```

### 4.2 Hover / Active Elevation

```css
.glass-card:hover {
  transform: translateY(-2px);
  box-shadow:
    0 0 30px rgba(0, 229, 255, 0.10),             /* enhanced outer glow */
    0 8px 32px rgba(0, 0, 0, 0.25),                /* depth shadow */
    inset 0 1px 0 rgba(255, 255, 255, 0.08);       /* brighter reflection */
}
```

### 4.3 Rules

- **No harsh drop shadows.** Cards appear lit from within, not cast against a wall.
- **No opaque fills.** The background content must subtly bleed through every card.
- Nested cards reduce backdrop-blur to `blur(20px)` and lower fill opacity to `0.4` to avoid compounding blur.

---

## 5 · Layout Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ TopNav  (72 px glass bar, sticky)                                          │
│  ├─ Left   : Logo icon + "NeuroAegis" wordmark                            │
│  ├─ Center : Dashboard · Analysis · Reports · Settings                     │
│  └─ Right  : Search (⌘K) · Notifications bell · Profile avatar            │
├────┬─────────────────────────────────────────────────────────┬──────────────┤
│Side│                                                         │   Right      │
│bar │              Hero Section  (~60 % width)                │   Analysis   │
│    │                                                         │   Panel      │
│icon│   Holographic 3D Brain Visualization                    │              │
│rail│   ┌───────────────────────────────────────────┐         │  ┌────────┐  │
│    │   │  Transparent rotating brain model         │         │  │EEG Wave│  │
│    │   │  • Neuron clusters with glowing nodes     │         │  ├────────┤  │
│    │   │  • Animated synapse pulses along edges    │         │  │Freq    │  │
│    │   │  • Translucent cortical shells            │         │  │Bands   │  │
│    │   │  • Orbital holographic rings              │         │  ├────────┤  │
│    │   │  • Bloom + depth-of-field post-process    │         │  │Classif.│  │
│    │   │  • Floating particle field                │         │  │+ Models│  │
│    │   │  • Slow auto-rotation (drag to orbit)     │         │  ├────────┤  │
│    │   └───────────────────────────────────────────┘         │  │Confid. │  │
│    │                                                         │  │Gauge   │  │
│    │                                                         │  ├────────┤  │
│    │                                                         │  │SHAP    │  │
│    │                                                         │  │Panel   │  │
│    │                                                         │  ├────────┤  │
│    │                                                         │  │Model   │  │
│    │                                                         │  │Selector│  │
│    │                                                         │  └────────┘  │
├────┴──────┬──────────┬──────────┬──────────┬──────────┬──────┴──────────────┤
│  Metric   │  Metric  │  Metric  │  Metric  │  Metric  │                    │
│  Card 1   │  Card 2  │  Card 3  │  Card 4  │  Card 5  │                    │
│ icon+name │ icon+name│ icon+name│ icon+name│ icon+name│                    │
│ value     │ value    │ value    │ value    │ value    │                    │
│ sparkline │ sparkline│ sparkline│ sparkline│ sparkline│                    │
│ badge     │ badge    │ badge    │ badge    │ badge    │                    │
├───────────┴──────────┴──────────┴──────────┴──────────┴────────────────────┤
│  Bottom Timeline — full-width neural signal history (scrollable)           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 TopNav

| Property | Value |
|----------|-------|
| Height | `72px` |
| Surface | Glass card treatment (§ 4) |
| Position | `sticky`, `top: 0`, `z-index: 50` |
| Left slot | 28 px logo icon + "NeuroAegis" in `--text-primary`, weight 600 |
| Center slot | Nav links: `Dashboard` · `Analysis` · `Reports` · `Settings` — active link underlined with `--accent-primary` |
| Right slot | Search trigger (`⌘K` shortcut badge), notification bell (dot indicator), profile avatar (32 px circle) |

### 5.2 Sidebar

| Property | Value |
|----------|-------|
| Width | `64px` collapsed (icon rail only) |
| Icons | Thin outline (Lucide / Phosphor), `20px`, `--text-secondary`, active: `--accent-primary` |
| Tooltip | On hover — glass mini-card with label |
| Expand | Optional `200px` expanded drawer (future iteration) |

### 5.3 Hero — 3D Brain Visualization

Occupies ~60 % of the main content width. Rendered via **Three.js / React Three Fiber**.

| Element | Description |
|---------|-------------|
| Brain mesh | Semi-transparent cortical surface, subtle wireframe overlay |
| Neurons | Glowing point-light nodes scattered across the cortex |
| Synapses | Animated energy pulses traveling along edges between neurons |
| Shells | 2–3 concentric translucent shells representing cortical layers |
| Rings | Holographic orbital rings (JARVIS-style) rotating on offset axes |
| Bloom | `UnrealBloomPass` — threshold `0.8`, strength `0.6`, radius `0.4` |
| Particles | 200–400 floating luminous dots, slow Brownian drift |
| Interaction | Auto-rotation at 0.3 rad/s; user drag overrides to orbit; scroll-zoom |

### 5.4 Right Analysis Panel

Vertically stacked glass cards, each independently scrollable.

| Sub-panel | Content |
|-----------|---------|
| **EEG Waveform** | Live multi-channel EEG trace — 4–8 channels, color-coded by brain region, `--accent-primary` dominant |
| **Frequency Bands** | Delta / Theta / Alpha / Beta / Gamma — horizontal frequency curve or stacked area, labeled |
| **Classification** | Prediction label (`Normal` / `Pre-ictal` / `Seizure`) with per-model breakdown table (model name, prediction, confidence) |
| **Confidence Gauge** | Circular arc gauge, 0–100 %, color-mapped: `--state-success` → `--state-warning` → `--state-danger` |
| **SHAP Panel** | Horizontal SHAP waterfall / force-plot showing top feature contributions — positive bars cyan, negative bars pink |
| **Model Selector** | Dropdown or segmented control listing available models (e.g., CNN-LSTM, Transformer, Random Forest) |

### 5.5 Reports Page

Accessible via the **Reports** nav link. Dedicated page with the following glass-card widgets:

| Widget | Visualization |
|--------|--------------|
| Accuracy | Single large number + sparkline trend |
| Precision | Single large number + sparkline trend |
| Recall | Single large number + sparkline trend |
| F1 Score | Single large number + sparkline trend |
| ROC-AUC Curve | Themed line chart — `--accent-primary` curve, `--text-secondary` diagonal reference |
| Confusion Matrix | Heatmap — 2 × 2 (or n × n) grid, intensity-mapped from `--bg-3` → `--accent-primary` |

### 5.6 Bottom Metric Cards (× 5)

Five equal-width glass cards arranged horizontally.

Each card contains:

```
┌─────────────────────────┐
│  ◉  Metric Name         │   ◉ = themed icon (Lucide)
│  1,247                  │   large value, --text-primary, weight 600
│  ▁▂▃▅▆▇█▇▅▃            │   sparkline (--accent-primary stroke)
│  ● +12.4 %              │   badge (success / warning / danger)
└─────────────────────────┘
```

### 5.7 Bottom Timeline

- Full-width glass card pinned to the bottom of the dashboard.
- Displays a continuous neural signal history as a scrollable EEG-style waveform strip.
- Time axis labeled in `--text-secondary`; signal amplitude in `--accent-primary`.
- Highlighted seizure-event regions marked with translucent `--state-danger` overlays.

---

## 6 · Chart & Visualization Rules

> [!CAUTION]
> **No bar charts. No pie charts. No donut charts.** These belong to corporate BI dashboards, not a neuroscience command center.

### 6.1 Permitted Visualization Types

| Type | Use Case | Key Styling Notes |
|------|----------|-------------------|
| EEG Waveform | Raw / processed brain signal | Multi-channel, color-coded, live-animated |
| ECG Trace | Heart-rate correlation | Single trace, `--accent-secondary` stroke |
| Frequency Curve | Spectral power distribution | Smooth area fill with gradient, labeled bands |
| Heatmap | Confusion matrix, electrode map | Sequential color ramp from `--bg-3` → `--accent-primary` |
| Circular Gauge | Confidence, accuracy, single KPI | Arc gauge, never a full donut |
| Sparkline | Inline trend in metric cards | Thin stroke, no axes, `--accent-primary` |
| ROC Curve | Model evaluation | Line chart with diagonal reference, themed axes |

### 6.2 Chart Theming Checklist

- [ ] Background: `transparent` (inherits glass card).
- [ ] Grid lines: `rgba(255, 255, 255, 0.04)` — barely visible.
- [ ] Axis labels: `--text-secondary`, `12px`, weight 400.
- [ ] Data strokes: Token colors only — never default library colors.
- [ ] Tooltips: Glass card mini-panel, `backdrop-filter: blur(16px)`.
- [ ] Animations: Smooth draw-in on mount (`waveformDraw` preset, § 9).
- [ ] No default chart library skins. Strip all vendor styles and re-skin from scratch.

---

## 7 · Typography

| Role | Family | Weight | Size | Tracking |
|------|--------|--------|------|----------|
| Display heading | SF Pro Display / Geist | 600 | 28–32 px | `-0.02em` |
| Section heading | Inter / Geist | 600 | 20–24 px | `-0.01em` |
| Body | Inter / IBM Plex Sans | 400 | 14–16 px | `0` |
| Caption / Label | Inter / IBM Plex Sans | 400 | 11–13 px | `0.02em` |
| Metric value | SF Pro Display / Geist | 600 | 32–48 px | `-0.03em` |
| Code / mono | IBM Plex Mono | 400 | 13 px | `0` |

### 7.1 Rules

- **Permitted weights: 400 · 500 · 600 only.** No thin (300), no bold (700), no black (900).
- **Generous spacing.** `line-height` ≥ 1.5 for body, ≥ 1.2 for headings.
- **Color.** Headings use `--text-primary`; body/captions use `--text-secondary`; interactive text uses `--accent-primary`.
- **No decorative fonts.** No serif, no hand-written, no novelty typefaces.

### 7.2 Font Stack (CSS)

```css
:root {
  --font-display: 'SF Pro Display', 'Geist', 'Inter', system-ui, sans-serif;
  --font-body:    'Inter', 'IBM Plex Sans', 'Geist', system-ui, sans-serif;
  --font-mono:    'IBM Plex Mono', 'Geist Mono', ui-monospace, monospace;
}
```

---

## 8 · Iconography

| Property | Value |
|----------|-------|
| Library | **Lucide** (primary) or **Phosphor** (alternative) |
| Style | Thin outline only — no filled, no duotone, no solid |
| Size | `20px` default; `16px` inline; `24px` nav / hero |
| Stroke width | `1.5px` |
| Color | `--text-secondary` default; `--accent-primary` active / interactive |

> [!WARNING]
> Never use filled or solid icon variants. They break the thin, futuristic aesthetic.

---

## 9 · Motion & Animation

### 9.1 Principles

- Smooth, purposeful micro-interactions — every animation must convey meaning (state change, spatial relationship, data update).
- **Nothing bouncy, nothing flashy.** No spring physics, no overshoot, no confetti.
- Ease curves: `ease`, `ease-out`, or custom cubic-bézier — never `linear` for UI transitions.
- Duration: 200–400 ms for micro-interactions; 600–1000 ms for page-level transitions.

### 9.2 Framer Motion Presets

```tsx
// ── src/lib/motion-presets.ts ──

export const cardFloat = {
  animate: { y: [0, -6, 0] },
  transition: { duration: 4, repeat: Infinity, ease: 'easeInOut' },
};

export const hoverElevate = {
  whileHover: { y: -2, boxShadow: '0 0 30px rgba(0,229,255,0.10)' },
  transition: { duration: 0.3, ease: 'easeOut' },
};

export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { duration: 0.5, ease: 'easeOut' },
};

export const slideUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: 'easeOut' },
};

export const pulseGlow = {
  animate: {
    boxShadow: [
      '0 0 10px rgba(0,229,255,0.05)',
      '0 0 25px rgba(0,229,255,0.12)',
      '0 0 10px rgba(0,229,255,0.05)',
    ],
  },
  transition: { duration: 3, repeat: Infinity, ease: 'easeInOut' },
};

export const waveformDraw = {
  initial: { pathLength: 0, opacity: 0 },
  animate: { pathLength: 1, opacity: 1 },
  transition: { duration: 1.2, ease: 'easeOut' },
};
```

### 9.3 Ambient Motion Inventory

| Element | Motion |
|---------|--------|
| Glass cards | Gentle floating via `cardFloat` preset |
| Card borders | Animated gradient sweep (conic-gradient rotation) |
| Background particles | Slow upward drift, randomized opacity flicker |
| Brain model | Continuous auto-rotation, bloom pulse on classification events |
| EEG waveform | Live scroll, real-time data append, smooth path interpolation |
| Metric sparklines | Draw-in on mount via `waveformDraw` |
| Hover states | `hoverElevate` — lift + enhanced glow |
| Page transitions | `fadeIn` + `slideUp` stagger across cards |

---

## 10 · Four UI States

Every data-driven component **must** implement all four states. No exceptions.

### 10.1 Loading

```
┌─────────────────────────────────────┐
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   Skeleton shimmer
│  ░░░░░░░░░░░░░░                    │   Animated gradient sweep
│  ░░░░░░░░░░░░░░░░░░░░░░            │   left → right, 1.5s loop
│  ░░░░░░░░░░░░░░░░░░                │   Matches final layout shape
└─────────────────────────────────────┘
```

- Use skeleton placeholders shaped to match the final content.
- Shimmer gradient: `linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent)`.
- Animate with `translateX(-100% → 100%)`, duration `1.5s`, infinite.

### 10.2 Empty

```
┌─────────────────────────────────────┐
│                                     │
│         ◇  No EEG data yet         │   On-brand illustration / icon
│     Begin a session to see live     │   Concise, helpful copy
│     neural signal analysis.         │   Optional CTA button
│                                     │
└─────────────────────────────────────┘
```

- Copy must be on-brand and contextual — not generic "No data found."
- Icon: thin outline, `--text-secondary`, `48px`.
- Optional call-to-action button in `--accent-primary`.

### 10.3 Error

```
┌─────────────────────────────────────┐
│                                     │
│      ⚠  Signal connection lost     │   --state-danger accent
│   We couldn't reach the EEG feed.  │   Descriptive error copy
│                                     │
│        [ Retry Connection ]         │   Retry button, danger outline
│                                     │
└─────────────────────────────────────┘
```

- Border tinted `--state-danger` at 20 % opacity.
- Retry button: outline style, `--state-danger` border and text.
- Never expose raw error codes or stack traces to the user.

### 10.4 Success (Data Loaded)

The full, rich visualization as designed — all charts, metrics, and interactive elements rendered.

---

## 11 · Anti-Patterns

> [!CAUTION]
> The following are **strictly forbidden** across the entire NeuroAegis interface. Violations must be caught in code review.

| ❌ Anti-Pattern | Why It's Banned |
|----------------|-----------------|
| Bootstrap / Material UI defaults | Generic, corporate, antithetical to the glassmorphic language |
| Flat design (no depth) | NeuroAegis requires layered glass depth — flatness reads as unfinished |
| Bright saturated fills | Overwhelm the dark canvas; only accents should carry chroma |
| Thick borders (> 1 px) | Destroy the delicate, frosted-glass illusion |
| Hospital / clinical aesthetic | White backgrounds, pastel blues, serif fonts — we are a research lab, not an EMR |
| Corporate BI dashboard style | Bar charts, pie charts, KPI tiles on white — antithetical to the command-center mood |
| Oversized charts | Charts must be information-dense, not presentation-deck large |
| Cluttered layouts | Every element must earn its pixels; remove before adding |
| Heavy typography (≥ 700 weight) | Contradicts the refined, light typographic hierarchy |
| Default chart library skins | Recharts / Chart.js / Highcharts defaults must be fully stripped and re-themed |
| Bar charts, pie charts, donut charts | Explicitly banned — use waveforms, gauges, heatmaps, sparklines, ROC curves instead |
| Bounce / spring / confetti animations | Unprofessional; motion must be smooth, subtle, and purposeful |

---

## 12 · Quick-Reference Cheat Sheet

```
Background     →  radial gradient + noise (2–3 %) + neural SVG + particles
Card surface   →  rgba(11,22,37,0.6)  blur(30px)  border-radius: 20px
Glow           →  0 0 20px rgba(0,229,255,0.05)  — never harsh shadows
Accent         →  #00E5FF (cyan)  #4B7DFF (blue)  #8B5CF6 (purple)
States         →  #00FFA3 (ok)  #FFB020 (warn)  #FF4D6D (danger)
Fonts          →  SF Pro Display / Inter / Geist / IBM Plex Sans  (400/500/600)
Icons          →  Lucide thin outline  1.5 px stroke  20 px
Charts         →  Waveforms · Gauges · Heatmaps · Sparklines · ROC curves ONLY
Motion         →  Smooth ease-out  200–400 ms  no bounce  no spring
Four states    →  Loading (shimmer) · Empty (on-brand) · Error (retry) · Success
```

---

*Document version 1.0 — authored 2026-07-18. Governed by the NeuroAegis design council. All changes require review.*


## 13 · New Redesign Patterns (V2)

The V2 UI redesign introduces the following patterns to enhance the clinical, futuristic aesthetic:

| Component | Description | Use Case |
|-----------|-------------|----------|
| **HudCornerFrame** | Thin corner-bracket frame wrapping a child element, featuring an integrated label and icon. | Wrapping glass cards and section titles to reinforce the "HUD" aesthetic. |
| **ClinicalMetaTable** | Compact two-column Attribute/Value table. The attribute is styled as secondary text, and the value as primary text. | Displaying session, patient, or model metadata. |
| **CircularScoreGauge** | Circular ring with a large centered number and a vertical gradient scale bar. Uses `--state-success`/`warning`/`danger` mapped colors. | Highlighting primary KPI scores like Confidence or Accuracy. |
| **ShapFeatureRow** | Repeating list row with a horizontal track and a signed SHAP value marker. | Displaying AI explainability feature importance. |
| **ModelSelectorSegmented** | Single-select segmented pill control. Active segment is highlighted with a background fill and active text color. | Switching between machine learning models (e.g., Random Forest vs XGBoost). |
| **InspectorPopoutCard** | Floating card breaking out past parent panel edge, acting as a detailed tooltip or inspector panel. | Showing deeper details on hover/click (e.g., detailed prediction stats). |
| **FrequencyBandRow** | Glowing sparkline paired with a Hz readout and a colored Greek-letter icon chip. | Visualizing specific EEG frequency bands (Alpha, Beta, Gamma, etc.). |
| **SessionFilmstrip** | Horizontal scrollable thumbnail strip of past sessions. Active session is highlighted. | Navigating patient history and past EEG recordings. |
| **DualLineComparisonChart** | Two datasets rendered as dotted/marker lines overlaid on fainter background lines. | Comparing model metrics side-by-side (e.g., Accuracy vs F1 Score). |
| **Toolbar Pattern** | Floating or inline strip with icon buttons (pan, zoom, measure) and toggles. | Enhancing main visualizations like the EEG Monitor or 3D Brain. |
