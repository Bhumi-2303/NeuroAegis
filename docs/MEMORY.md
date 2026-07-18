# NeuroAegis — Decision & Memory Log

> This is a living, append-only document. Never overwrite previous entries. Always append new entries at the bottom with a dated header.

---

## 2026-07-18 — Project Initialization

### Phase Status
- Phase 0 (Foundations): IN PROGRESS — Generating governance documents
- Phases 1-7: NOT STARTED

### Decisions Made
1. **Monorepo structure**: Using workspace-based monorepo with `apps/web/` and `packages/model-contracts/` as specified
2. **Tech stack confirmed**: React 18, TypeScript, Vite, TailwindCSS, shadcn/ui (restyled), Framer Motion, Recharts, Zustand, TanStack Query, Zod, React Router
3. **Mock-first approach**: All model integration uses IModelService interface backed by mock generators. Real model integration is a config flip, not a rewrite.
4. **9 governance docs**: PRD, ARCHITECTURE, RULES, PHASES, DESIGN, MEMORY (this file), TRD, AGENT, IMPLEMENTATIONPLAN

### Assumptions
1. The hero 3D brain visualization will use Three.js / React Three Fiber for WebGL rendering, as this is the most practical approach for holographic/volumetric effects in a browser. The spec does not mandate a specific 3D library.
2. shadcn/ui components will be installed via CLI and restyled — they are a starting point, not used bare.
3. The monorepo will use npm workspaces (not pnpm/yarn) unless a specific tool is required.
4. The `streamEEG` method on IModelService returning `AsyncIterable<GraphDataPoint>` will be implemented as an async generator yielding mock data points at intervals, simulating a real-time stream.

### Open Questions
1. **3D brain model asset**: The spec calls for a 'floating holographic 3D brain' but does not specify whether to use a pre-built 3D model file (.glb/.gltf) or procedurally generate the brain geometry. Decision: Will attempt procedural generation with Three.js primitives first; if insufficient quality, will log and revisit.
2. **shadcn/ui component subset**: The spec says 'restyled to glass/neon system' but doesn't list which shadcn/ui components to install. Will install on-demand as features require them.
3. **Search and notifications**: The spec includes 'global search' and 'notifications' in TopNav but provides no data contracts or feature specs for these. Will implement as visual stubs (icon + dropdown shell) with no backing logic.

### Deviations
None so far.

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->

## 2026-07-18 — End of Phase 0

### Phase Status
- Phase 0 (Foundations): COMPLETED
- Phase 1 (Design System): NEXT

### Decisions Made
1. **Packages**: `model-contracts` is successfully scaffolded with all TS interfaces defined in Section 7.
2. **Web App Scaffolding**: Vite+React+TS environment has been successfully created.
3. **Dependencies Installed**: `tailwindcss`, `@tailwindcss/postcss`, `framer-motion`, `recharts`, `zustand`, `@tanstack/react-query`, `zod`, `react-router-dom`, `lucide-react` all installed at the root workspace `apps/web`.
4. **Folder Structure**: Core architectural directories under `apps/web/src` have been created (UI components, hooks, API layer, model integration layer).

### Assumptions
1. We had to use `@tailwindcss/postcss` for Tailwind V4, instead of the legacy `tailwindcss` PostCSS plugin, to ensure `npm run build` succeeds correctly.
2. Index files (`index.ts`) for features have been scaffolded to enforce the explicit module boundary pattern described in the Architecture doc.

### Open Questions
None currently.

## 2026-07-18 — End of Phase 1

### Phase Status
- Phase 1 (Design System): COMPLETED
- Phase 2 (Shell & Nav): NEXT

### Decisions Made
1. **Design Tokens**: Configured in `theme.css` under `@theme` for Tailwind CSS compatibility.
2. **Primitives**: `GlassPanel`, `NeonBorder`, `HoloRing`, `ParticleField` built using `framer-motion` for hardware accelerated animations.
3. **Components**: `GlassCard` and `SkeletonShimmer` implemented using the design tokens.
4. **Build Fixes**: Used `import type` per TS `verbatimModuleSyntax` config requirement. 

### Assumptions
1. The background textures (`/textures/noise.png` and `/textures/neural-pattern.svg`) will be provided via public assets or added in a later polish phase. Warning is ignored during build for now.

### Open Questions
None currently.

## 2026-07-18 — End of Phase 2

### Phase Status
- Phase 2 (Shell & Nav): COMPLETED
- Phase 3 (Core Layer): NEXT

### Decisions Made
1. **Layout**: DashboardShell constructed with absolute positioning to perfectly fit the TopNav and Sidebar.
2. **Navigation**: `react-router-dom` successfully implemented. 
3. **Motion**: Added global Framer Motion presets to `src/shared/lib/motion-presets.ts`.

### Assumptions
1. Mobile-friendly navigation is deferred to Phase 6 (Polish).
2. The TopNav requires a hardcoded height and fixed positioning per design constraints.

### Open Questions
None currently.
