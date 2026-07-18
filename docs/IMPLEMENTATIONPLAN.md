# NeuroAegis — Ordered Implementation Plan

> **Purpose:** Step-by-step build checklist for the NeuroAegis XAI Seizure-Detection Dashboard.
> Each task maps to its parent phase defined in [`PHASES.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/PHASES.md).
> Mark items `[x]` as they are completed; never delete a line.

---

## Phase 0 — Foundations

> [!NOTE]
> All scaffolding, configuration, and contract types are established here.
> Nothing renders visually yet — the goal is a clean, green `pnpm build`.

- [ ] Create all 9 governance docs (`PHASES.md`, `ARCHITECTURE.md`, `DESIGNSYSTEM.md`, `COMPONENTSSPEC.md`, `STATEMACHINE.md`, `MODELCONTRACT.md`, `IMPLEMENTATIONPLAN.md`, `PRD.md`, `MEMORY.md`)
- [ ] Scaffold root `package.json` (monorepo workspaces: `apps/*`, `packages/*`)
- [ ] Scaffold `apps/web/` with Vite + React 18 + TypeScript
- [ ] Install dependencies:
  - `tailwindcss`
  - `shadcn/ui`
  - `framer-motion`
  - `recharts`
  - `zustand`
  - `@tanstack/react-query`
  - `zod`
  - `react-router-dom`
  - `lucide-react`
- [ ] Configure `tsconfig.json`, `tailwind.config.ts`, `vite.config.ts`
- [ ] Create `.env.example` with placeholder variables
- [ ] Create `packages/model-contracts/` with all shared type files
- [ ] Scaffold all feature directories with barrel `index.ts` files
- [ ] Scaffold `shared/`, `core/`, `design-system/` directories
- [ ] **Gate ✓** — Verify `pnpm build` succeeds with zero errors

---

## Phase 1 — Design System

> [!NOTE]
> Tokens, primitives, and base components that every later phase depends on.
> At the end of this phase the app renders an empty themed glass shell.

- [ ] Create design tokens (colors, typography, spacing, motion curves)
- [ ] Create `theme.css` (CSS custom-property surface)
- [ ] Create glassmorphic primitives:
  - `GlassPanel`
  - `NeonBorder`
  - `HoloRing`
  - `ParticleField`
- [ ] Create `globals.css` (resets, base layers, dark-mode defaults)
- [x] Create compound components:
  - `GlassCard`
  - `SkeletonShimmer`
  - `NeonBadge`
  - `StatusBadge`
  - `WaveformSparkline`
  - `ConfidenceGauge`
  - `ShapBarChart`
  - `EmptyState`
  - `ErrorState`
- [x] **Gate ✓** — Verify themed shell renders in browser with correct token values

---

## Phase 2 — Shell & Navigation

> [!NOTE]
> Application chrome and routing. Every page is reachable but shows only a placeholder.

- [x] Build layout components:
  - `TopNav`
  - `Sidebar`
  - `DashboardShell`
- [x] Define routes for all pages (`/`, `/eeg`, `/frequency`, `/prediction`, `/explainability`, `/reports`, `/patients`, `/settings`)
- [x] Wire `App.tsx` with top-level providers
- [x] Create providers:
  - `QueryProvider` (React Query)
  - `ThemeProvider`
  - `StoreProvider` (Zustand)
- [x] Add placeholder pages for every route
- [x] **Gate ✓** — Verify navigation between all routes works without errors

---

## Phase 3 — Core Layer

> [!NOTE]
> API abstractions, mock data generators, Zod schemas, and state stores.
> After this phase every feature can develop against deterministic mock data.

- [x] Build API layer (`httpClient.ts`, typed endpoint map)
- [x] Define `ModelService.interface.ts` (`IModelService`)
- [x] Create mock data generators (one per feature domain)
- [x] Create `model.config.ts` (model metadata, thresholds, label maps)
- [x] Implement `ModelService.ts` (mock-backed implementation of `IModelService`)
- [x] Create global state stores (UI state, notification state)
- [x] Define Zod validation schemas for all API contracts
- [x] Create per-feature stores (EEG, frequency, prediction, explainability, reports, patients)
- [x] **Gate ✓** — Verify mock data flows through stores and renders in devtools

---

## Phase 4 — Feature Modules

> [!IMPORTANT]
> Every feature component must implement all four UI states defined in
> [`STATEMACHINE.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/STATEMACHINE.md):
> **idle → loading → success → error**.

### 4a — Individual Feature Panels

- [x] `eeg-monitor` — real-time waveform canvas, 4 states (idle · loading · success · error)
- [x] `frequency-analysis` — band-power bar/area charts, 4 states
- [x] `seizure-prediction` — classification result, confidence gauge, model selector dropdown, 4 states
- [x] `explainability` — SHAP waterfall / bar panel, 4 states
- [x] `reports` — aggregate metrics, ROC curve, confusion matrix, 4 states
- [x] `patients` — patient list / detail view, 4 states
- [x] `settings` — theme toggles, model configuration panel
- [x] `brain-analysis` — 3D viewport control strip (rotate, zoom, highlight)

### 4b — Shared UI Components

- [x] `NeonBadge`
- [x] `WaveformSparkline`
- [x] `ConfidenceGauge`
- [x] `ShapBarChart`
- [x] `StatusBadge`

### 4c — Dashboard Composition

- [x] Bottom metric cards (×5: Signal Strength, Active Channels, Confidence, Session Duration, Alert Count)
- [x] Dashboard page — compose all panels into grid layout
- [x] **Gate ✓** — Verify all four states render correctly for every feature panel

---

## Phase 5 — Hero Visualization

> [!NOTE]
> The centrepiece 3D brain visualisation. Performance budget: ≥ 50 fps on mid-range GPU.

- [x] Set up 3D brain scene (Three.js / React Three Fiber)
- [x] Implement sub-elements:
  - Neuron point cloud
  - Synapse connection lines
  - Transparent cortex shells
  - Auto-rotation & mouse interaction
  - Bloom post-processing
  - Holographic ring overlays
- [x] Add ambient particle system
- [x] Add bottom timeline waveform (synced EEG trace)
- [x] Wire visualisation data to `ModelService` (highlight regions, risk heat-map)
- [x] **Gate ✓** — Verify ≥ 50 fps, no dropped frames on 60 Hz monitor

---

## Phase 6 — Polish

> [!NOTE]
> Cross-cutting quality pass. No new features — only refinement.

- [ ] Add micro-interactions (hover glows, press feedback, panel transitions)
- [ ] Responsive layout pass (mobile → tablet → desktop breakpoints)
- [ ] Accessibility audit (WCAG 2.1 AA):
  - ARIA labels on all interactive elements
  - Focus-visible outlines
  - Keyboard navigation through every panel
  - Colour contrast ≥ 4.5 : 1 for text
- [ ] Reduced-motion support (`prefers-reduced-motion` media query disables non-essential animations)
- [ ] Performance optimisation:
  - `React.memo` / `useMemo` on heavy components
  - Code-split feature routes
  - Lazy-load 3D scene
- [ ] Visual audit against [`DESIGNSYSTEM.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/DESIGNSYSTEM.md) token spec

---

## Phase 7 — Model-Readiness Review

> [!CAUTION]
> This phase is a verification-only gate. No production code changes are made here.
> Every item must pass before the codebase is declared model-integration-ready.

- [ ] `grep -r "TODO\|FIXME\|HACK" src/` — resolve or document all remaining markers
- [ ] Verify every public method in `ModelService.ts` satisfies `IModelService` interface contract
- [ ] Verify no component imports data except through `ModelService` (no direct `fetch` / `axios`)
- [ ] Verify mock ↔ real swap is a single-file change (`ModelService.ts` implementation)
- [ ] Verify no cross-feature imports (feature A never imports from feature B's internals)
- [ ] Documentation consistency check — all 9 docs reflect final implementation
- [ ] Final `MEMORY.md` update with lessons learned, deferred decisions, known limitations
- [ ] **Definition of Done ✓** — All gates from Phases 0–6 pass; all checklist items above are `[x]`

---

> [!TIP]
> **Working agreement:** Complete every task within a phase before moving to the next.
> Each phase ends with a **Gate ✓** verification step that must pass before proceeding.
