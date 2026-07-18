# NeuroAegis — Coding Standards & Rules

> **Canonical reference for every contributor.** No PR is merged unless it complies with every rule in this document. When in doubt, the rule wins — ask before deviating.

---

## Table of Contents

1. [TypeScript Standards](#typescript-standards)
2. [Naming Conventions](#naming-conventions)
3. [Component Rules](#component-rules)
4. [State Management](#state-management)
5. [Import / Dependency Rules](#import--dependency-rules)
6. [ML Boundary (Absolute)](#ml-boundary-absolute)
7. [TODO Convention](#todo-convention)
8. [Design System Rules](#design-system-rules)
9. [Commit Conventions](#commit-conventions)
10. [Accessibility (WCAG AA)](#accessibility-wcag-aa)
11. [Anti-Patterns (Explicitly Forbidden)](#anti-patterns-explicitly-forbidden)

---

## TypeScript Standards

> [!IMPORTANT]
> TypeScript is the **only** language used across the entire NeuroAegis codebase — frontend, tooling, scripts, and shared packages. There are zero exceptions.

| Rule | Detail |
|---|---|
| **No `any` — ever** | The `any` type is banned without exception. Use `unknown` when a type truly cannot be determined at author-time, then narrow immediately. ESLint's `@typescript-eslint/no-explicit-any` is set to `error`. |
| **Strict mode** | `tsconfig.json` must enable `"strict": true` at the root. All strict-family flags (`strictNullChecks`, `strictFunctionTypes`, `strictBindCallApply`, `strictPropertyInitialization`, `noImplicitAny`, `noImplicitThis`, `alwaysStrict`) are enforced. |
| **Single source of truth for data shapes** | Every interface, type alias, enum, and constant that describes a domain data shape lives in `packages/model-contracts`. No file outside that package may redefine, shadow, or extend these shapes locally. Import them — never copy them. |
| **Explicit return types on exports** | Every exported function, method, and hook must declare an explicit return type. Internal/private helpers may rely on inference. |
| **`interface` over `type` for objects** | Prefer `interface` when defining the shape of an object. Reserve `type` for unions, intersections, mapped types, and utility derivations. |

```typescript
// ✅ Correct
import type { SeizurePrediction } from "@neuroaegis/model-contracts";

export function formatPrediction(raw: SeizurePrediction): FormattedPrediction {
  // ...
}

// ❌ Wrong — local redefinition, implicit return, `any`
type SeizurePrediction = { score: any };
export const formatPrediction = (raw: SeizurePrediction) => ({ /* ... */ });
```

---

## Naming Conventions

Consistent naming eliminates cognitive overhead. Every name in the codebase must follow the table below.

| Construct | Convention | Example |
|---|---|---|
| **Files** | `kebab-case` | `seizure-timeline.tsx`, `use-eeg-stream.ts` |
| **Components** | `PascalCase` | `SeizureTimeline`, `PatientHeader` |
| **Hooks** | `camelCase` with `use` prefix | `useEegStream`, `usePatientAlerts` |
| **Types / Interfaces** | `PascalCase` | `EegChannelData`, `PredictionResult` |
| **Constants** | `UPPER_SNAKE_CASE` | `MAX_CHANNEL_COUNT`, `DEFAULT_WINDOW_MS` |
| **Zustand stores** | `camelCase` with `use` prefix and `Store` suffix | `useAlertStore`, `useDashboardStore` |
| **CSS custom properties** | `kebab-case` with category prefix | `--color-primary-500`, `--spacing-panel-md`, `--radius-card` |

> [!NOTE]
> The category prefix for CSS custom properties must reflect the token's domain: `--color-*`, `--spacing-*`, `--radius-*`, `--shadow-*`, `--font-*`, `--z-*`, etc. Unprefixed custom properties are not allowed.

---

## Component Rules

NeuroAegis enforces a strict **Presentational / Container** split. Mixing responsibilities is a rejectable offence.

### Presentational Components

Presentational components are pure rendering units. They must satisfy **all** of the following constraints:

- Accept data and callbacks exclusively through **props**.
- Contain **no data fetching** logic (no `fetch`, no Axios, no TanStack Query hooks).
- Contain **no business rules** or domain logic.
- Have **no direct store access** (no Zustand `useStore` calls).
- Have **no TanStack Query hooks** (`useQuery`, `useMutation`, `useSuspenseQuery`, etc.).

```typescript
// ✅ Presentational — props only
interface EegWaveformProps {
  channels: EegChannelData[];
  onChannelSelect: (id: string) => void;
}

export function EegWaveform({ channels, onChannelSelect }: EegWaveformProps): ReactElement {
  return ( /* pure render */ );
}
```

### Container Components

Container components are the **wiring layer** between feature hooks and presentational components. Their sole job is to:

1. Call feature hooks (which internally use Zustand stores and TanStack Query).
2. Map hook outputs to presentational component props.
3. **Handle all four async states** for every asynchronous data panel (see below).

### The Four Async States

> [!IMPORTANT]
> Every panel that renders server-fetched or model-derived data **must** implement all four states. No exceptions.

| State | Requirement |
|---|---|
| **Loading** | Skeleton shimmer placeholder matching the panel's final layout. No generic spinners. |
| **Empty** | On-brand copy explaining the absence of data. Contextual to the feature (e.g., *"No seizure events detected in the selected window."*). |
| **Error** | Retry action button. Uses the `danger` accent colour from the design system. Displays a user-friendly message — never a raw stack trace. |
| **Success** | The full visualization or data view, themed and accessible. |

```typescript
// ✅ Container — wires hooks to presentational shell
export function SeizureTimelineContainer(): ReactElement {
  const { data, isLoading, isError, refetch } = useSeizureEvents();

  if (isLoading) return <SeizureTimelineSkeleton />;
  if (isError)   return <PanelError onRetry={refetch} />;
  if (!data?.length) return <PanelEmpty message="No seizure events detected in the selected window." />;

  return <SeizureTimeline events={data} />;
}
```

---

## State Management

NeuroAegis separates **server state** from **client UI state** with two complementary tools.

| Concern | Tool | Location |
|---|---|---|
| **Server / async state** | TanStack Query | Feature hooks (`features/*/hooks/`) |
| **Client UI state** | Zustand | Feature stores and global stores (see below) |

### Zustand Rules

| Rule | Detail |
|---|---|
| **One slice per feature** | Each feature directory owns exactly one Zustand store slice. |
| **No cross-feature coupling** | A feature's store must never import from, subscribe to, or mutate another feature's store. |
| **Feature stores location** | `features/<feature-name>/store/` |
| **Global stores location** | `core/services/state/` — reserved for truly cross-cutting concerns (theme, layout, auth session). |

```
src/
├── features/
│   ├── seizure-detection/
│   │   └── store/
│   │       └── use-seizure-detection-store.ts   ← feature-scoped
│   └── patient-overview/
│       └── store/
│           └── use-patient-overview-store.ts     ← feature-scoped
└── core/
    └── services/
        └── state/
            ├── use-theme-store.ts                ← global
            └── use-layout-store.ts               ← global
```

---

## Import / Dependency Rules

> [!CAUTION]
> Violating the dependency graph below creates circular imports, breaks code-splitting, and couples features that must remain independent. Lint rules enforce these boundaries — do not disable them.

### Dependency Flow (Allowed Direction →)

```
UI Components → Hooks → API Layer → Model Integration Layer
```

| Layer | May Import From |
|---|---|
| UI Components (Presentational) | `shared/`, `design-system/`, own props only |
| Hooks | `shared/`, `design-system/`, API Layer, Model Integration Layer |
| API Layer | `shared/`, `packages/model-contracts` |
| Model Integration Layer | `shared/`, `packages/model-contracts` |

### Boundary Rules

| Rule | Detail |
|---|---|
| **`shared/` and `design-system/`** | Importable by **any** layer or feature. These are leaf dependencies with no upward imports. |
| **`features/*` isolation** | A feature directory must **never** import from another `features/*` directory. If two features need shared logic, that logic belongs in `shared/`. |
| **Feature hooks → ModelService** | Feature hooks call the `ModelService` abstraction layer. They never call mocks, stubs, or ML code directly. |
| **All contracts from `packages/model-contracts`** | Every shared interface, type, and constant is imported from `packages/model-contracts`. No local redefinitions. |

```typescript
// ✅ Correct — feature hook imports from ModelService
import { ModelService } from "@/core/services/model/model-service";
import type { PredictionRequest } from "@neuroaegis/model-contracts";

// ❌ Wrong — cross-feature import
import { usePatientAlerts } from "@/features/alert-manager/hooks/use-patient-alerts";
```

---

## ML Boundary (Absolute)

> [!CAUTION]
> This is a **hard architectural boundary**. The NeuroAegis frontend is a visualization and command-center layer — it does **not** contain, execute, or simulate machine-learning logic.

| Rule | Detail |
|---|---|
| **Never write ML logic anywhere** | No inference code, no weight loading, no preprocessing pipelines, no feature extraction, no model evaluation — anywhere in the frontend codebase. |
| **No model code outside `core/services/model/`** | The `ModelService` in `core/services/model/` is the **sole** integration surface between the UI and the trained seizure-detection model. All model calls are routed through this service. |
| **Every integration point is marked** | Every location in `ModelService` where a real model call will eventually land must carry the marker comment `// TODO: Integrate Trained Model`. |
| **Feature hooks call `ModelService` only** | Feature hooks consume `ModelService` methods. They never instantiate models, call Python backends directly, or import model utilities outside the service boundary. |

```typescript
// core/services/model/model-service.ts

export class ModelService {
  async getPrediction(request: PredictionRequest): Promise<PredictionResult> {
    // TODO: Integrate Trained Model
    throw new Error("Model integration pending.");
  }

  async getExplanation(eventId: string): Promise<ExplanationResult> {
    // TODO: Integrate Trained Model
    throw new Error("Model integration pending.");
  }
}
```

---

## TODO Convention

The `// TODO: Integrate Trained Model` marker is a **controlled placeholder** — not a casual note.

| Rule | Detail |
|---|---|
| **Exact string** | The marker must be the exact string `// TODO: Integrate Trained Model`. No variations, no additional text on the same comment line. |
| **Allowed locations** | `ModelService.ts` and configuration files only. |
| **Forbidden locations** | UI components, hooks, presentational files, store files, or any file outside the model integration boundary. |
| **One per integration point** | Every method or configuration value that will eventually connect to the trained model must carry exactly one marker. |

> [!WARNING]
> Finding `// TODO: Integrate Trained Model` inside a React component or a Zustand store means the ML boundary has been violated. Refactor the call into `ModelService` immediately.

---

## Design System Rules

NeuroAegis implements a **luxury, futuristic, glassmorphic** visual identity. Every visual decision must flow through the design system.

| Rule | Detail |
|---|---|
| **Design tokens are mandatory** | All colours, spacing, radii, shadows, typography, and z-indices are consumed via CSS custom properties (`--color-*`, `--spacing-*`, etc.). Hard-coded values are rejected. |
| **shadcn/ui must be restyled** | shadcn/ui primitives are used for behaviour (accessibility, keyboard handling). Their default visual styles must be **completely overridden** with NeuroAegis design tokens to achieve the glassmorphic aesthetic. Bare/unstyled shadcn/ui components are not permitted. |
| **No Bootstrap or Material UI** | These frameworks are banned from the dependency tree. They conflict with the glassmorphic design language and introduce visual inconsistency. |
| **Charts are always themed** | Every chart, graph, and data visualization must use colours, fonts, and spacing from the design token palette. Default library themes are never shipped to production. |
| **Icon libraries** | Only **Lucide** or **Phosphor** icons are permitted, and exclusively in their **thin outline** weight. Filled, bold, or duotone variants are not allowed. |

> [!TIP]
> When adding a new shadcn/ui primitive, immediately create a wrapper in `design-system/components/` that applies NeuroAegis tokens. Import only the wrapper throughout the app — never the raw primitive.

---

## Commit Conventions

All commits follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Allowed Prefixes

| Prefix | Use |
|---|---|
| `feat` | A new user-facing feature or capability |
| `fix` | A bug fix |
| `docs` | Documentation-only changes |
| `refactor` | Code restructuring with no behaviour change |
| `style` | Formatting, whitespace, design token updates (no logic change) |
| `test` | Adding or updating tests |
| `chore` | Build config, dependency bumps, CI pipeline changes |

### PR Rules

| Rule | Detail |
|---|---|
| **One feature per PR** | A pull request addresses exactly one feature, fix, or chore. Omnibus PRs are rejected. |
| **PR references phase** | The PR title or description must reference the project phase it belongs to (e.g., `Phase 2 — Dashboard Shell`). |
| **No cross-feature imports** | PRs that introduce imports between `features/*` directories are automatically blocked. |

```
feat(seizure-detection): add EEG waveform panel skeleton — Phase 3

fix(design-system): correct glassmorphic blur radius on alert cards

docs(rules): update ML boundary section with ModelService examples
```

---

## Accessibility (WCAG AA)

> [!IMPORTANT]
> NeuroAegis is a medical-context dashboard. Accessibility is a **clinical requirement**, not a nice-to-have. Every component must meet WCAG 2.1 AA or higher.

| Requirement | Standard | Detail |
|---|---|---|
| **Keyboard navigation** | WCAG 2.1.1 | Every interactive element (buttons, links, tabs, dialogs, menus, chart controls) must be fully operable via keyboard alone. |
| **Colour contrast — normal text** | WCAG 1.4.3 | Minimum contrast ratio of **4.5 : 1** against its background. |
| **Colour contrast — large text / UI components** | WCAG 1.4.3 / 1.4.11 | Minimum contrast ratio of **3 : 1**. |
| **Alt text / aria-labels** | WCAG 1.1.1 | All images, icons, and non-text content must have meaningful `alt` text or `aria-label`. Decorative elements use `alt=""` or `aria-hidden="true"`. |
| **Visible focus states** | WCAG 2.4.7 | Focus indicators must be clearly visible and styled consistently with the design system (glassmorphic glow ring or equivalent). Never remove `outline` without providing a replacement. |
| **Semantic HTML** | WCAG 1.3.1 | Use correct HTML elements (`<nav>`, `<main>`, `<section>`, `<button>`, `<table>`, etc.). Do not repurpose `<div>` or `<span>` for interactive roles without ARIA. |
| **Screen reader support for charts** | WCAG 1.1.1 | Every chart and visualization must provide a screen-reader-accessible summary — either via a visually hidden data table, `aria-label` on the chart container, or a descriptive `<figcaption>`. |

---

## Anti-Patterns (Explicitly Forbidden)

> [!CAUTION]
> The following patterns are **explicitly banned**. Any occurrence in a PR is grounds for immediate rejection.

### Code Anti-Patterns

| ❌ Forbidden | Why |
|---|---|
| `any` type | Destroys type safety across the entire call chain. Use `unknown` and narrow. |
| Local type redefinitions | Causes drift from the canonical `packages/model-contracts` shapes. Always import. |
| Cross-feature imports (`features/A` → `features/B`) | Creates hidden coupling and breaks independent deployability. Extract to `shared/`. |
| Generic spinners (dots, wheels, circular loaders) | Violate the skeleton-shimmer loading standard and feel cheap. Use layout-matched skeletons. |
| Raw stack traces in UI | Exposing internals is a security and UX failure. Show user-friendly messages with retry. |
| Invented domain metrics | Never fabricate medical/ML metrics (accuracy, sensitivity, etc.) not defined in `packages/model-contracts`. |

### Design Anti-Patterns

| ❌ Forbidden | Why |
|---|---|
| Bare/unstyled shadcn/ui | Breaks the glassmorphic design language. Every primitive must be wrapped and themed. |
| Bootstrap or Material UI | Banned frameworks — incompatible with NeuroAegis visual identity. |
| Flat design | NeuroAegis is glassmorphic and futuristic. Flat, non-dimensional UI is off-brand. |
| Bar charts or pie charts | These are banned visualization types. Use radial gauges, area/stream charts, heatmaps, waveforms, or other advanced visualizations appropriate for the clinical-futuristic aesthetic. |
| Bright saturated colours | Colours must be muted, frosted, and consistent with the glassmorphic palette defined in design tokens. Neon or fully saturated hues are rejected. |
| Thick borders | Borders must be subtle (`1px` max, translucent). Heavy borders break the glass effect. |
| Hospital aesthetic | NeuroAegis is a **luxury AI command center**, not a hospital EMR. Clinical sterility, plain white backgrounds, and institutional colour schemes are off-brand. |

---

> **Last updated:** 2026-07-18
>
> **Maintained by:** NeuroAegis Core Team
