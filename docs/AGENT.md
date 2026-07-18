# NeuroAegis — Agent Operating Rules

> **Scope:** Binding behavioural rules for any AI agent or human contributor building the NeuroAegis XAI Seizure-Detection Dashboard.
> **Canonical References:** [`PRD.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/PRD.md) · [`ARCHITECTURE.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/ARCHITECTURE.md) · [`IMPLEMENTATIONPLAN.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/IMPLEMENTATIONPLAN.md) · [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md)
> **Version:** 1.0.0

---

## Table of Contents

1. [The Golden Rule](#1-the-golden-rule)
2. [Ambiguity Handling Protocol](#2-ambiguity-handling-protocol)
3. [Definition of Done](#3-definition-of-done)
4. [File-Editing Protocol](#4-file-editing-protocol)
5. [When to Update MEMORY.md](#5-when-to-update-memorymd)
6. [ML Boundary — Hard Line](#6-ml-boundary--hard-line)
7. [Quality Bars](#7-quality-bars)
8. [Phase Gate Protocol](#8-phase-gate-protocol)

---

## 1. The Golden Rule

> **If it is not in the spec, do not guess. Log it. Proceed with the nearest constraint. Silence is a stop sign.**

| Principle | Rule |
|---|---|
| **Spec is law** | Every implementation decision must trace to a requirement in [`PRD.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/PRD.md) (Sections 4, 6, 7), [`ARCHITECTURE.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/ARCHITECTURE.md), or a companion governance doc. |
| **No invention** | Never fabricate domain language, metric names, panel names, data shapes, or feature behaviour that is not explicitly specified. |
| **Log the gap** | If the spec is silent on something you need, log the question in [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) under **Open Questions** with a dated entry. |
| **Nearest constraint** | After logging, proceed using the closest matching constraint from the spec — document the assumption you made. |
| **Silence = stop** | If there is no nearest constraint and the decision could affect data shapes, contracts, or feature behaviour — **stop and ask**. Do not continue. |

> [!CAUTION]
> Guessing at domain concepts (e.g. inventing a new EEG frequency band, adding a metric not listed in PRD §4.4, or creating an unlisted panel) is a **blocking violation**. Always defer to the spec.

---

## 2. Ambiguity Handling Protocol

Not all ambiguity is equal. Classify every unclear situation into one of two categories before acting:

### 2a. Assumption-Safe (Green Path)

Use when the risk of a wrong assumption is **low** and recoverable.

| Step | Action |
|---|---|
| 1 | Infer the most reasonable interpretation from existing spec context. |
| 2 | Log the assumption in [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) → **Assumptions** with a dated entry. |
| 3 | Proceed with implementation. |

**Examples of assumption-safe decisions:**
- Choosing a specific Tailwind utility class for a glassmorphic blur radius.
- Deciding the internal order of imports within a file.
- Picking a spring-physics curve for a Framer Motion transition.
- Selecting a specific shade within the established neon palette range.

### 2b. Blocking (Red Path)

Use when a wrong assumption risks **building the wrong thing** — incorrect data shapes, wrong contracts, missing features, or architectural violations.

| Step | Action |
|---|---|
| 1 | **Stop.** Do not write code. |
| 2 | Log the question in [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) → **Open Questions**. |
| 3 | Wait for clarification or explicit spec amendment. |

**Examples of blocking decisions:**
- A panel or feature not listed in [`PRD.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/PRD.md) §6 (Feature List).
- A metric not listed in [`PRD.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/PRD.md) §4.4 (Evaluation Metrics) or §7 (Success Metrics).
- A data shape or contract field not defined in model-contracts.
- A dependency direction that contradicts the [import matrix](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/ARCHITECTURE.md) in `ARCHITECTURE.md` §5.
- Any cross-feature import (`features/A` → `features/B`).

> [!IMPORTANT]
> **Never invent domain language, metric names, panel identifiers, or data shapes outside of what is defined in [`PRD.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/PRD.md) Sections 4 and 7.** If you need a new term, it must be proposed and approved before use.

---

## 3. Definition of Done

A task is **not done** until every item in this checklist passes. No exceptions.

### 3a. Type Safety

| # | Criterion |
|---|---|
| D-01 | **Zero TypeScript errors** — `tsc --noEmit` exits with code 0. |
| D-02 | **No `any`** — the `any` type is never used, not even as a cast. Use `unknown` with narrowing or explicit types. |
| D-03 | **Strict mode** — `tsconfig.json` has `"strict": true`. No opt-outs (`skipLibCheck` for node_modules is acceptable). |

### 3b. Async State Coverage

| # | Criterion |
|---|---|
| D-04 | **Four UI states** — every async-driven component renders all four states: **idle**, **loading**, **success**, **error**. |
| D-05 | **Loading state** — uses `SkeletonShimmer` or equivalent glassmorphic loading indicator. Never a blank panel. |
| D-06 | **Error state** — displays a user-facing error message within the panel. Never an unhandled crash. |

### 3c. Contract Compliance

| # | Criterion |
|---|---|
| D-07 | **Shapes match contracts** — every data object consumed by a component matches the type defined in `packages/model-contracts/`. Zod schemas validate at runtime boundaries. |
| D-08 | **No cross-feature imports** — `features/*` directories are fully isolated. Shared concerns live in `shared/` or `design-system/`. Verified by ESLint `no-restricted-imports`. |

### 3d. Code Hygiene

| # | Criterion |
|---|---|
| D-09 | **TODO markers** — any point where a future model integration will replace mock logic is marked with `// TODO(model): <description>`. |
| D-10 | **Organised imports** — imports follow the order: external packages → `@shared/` → `@design-system/` → `@services/` → relative. |
| D-11 | **No dead code** — unused variables, unreachable branches, and commented-out blocks are removed. |

### 3e. Visual & Documentation

| # | Criterion |
|---|---|
| D-12 | **Visual match** — the rendered component matches the glassmorphic/neon design language. No bare, unstyled browser-chrome elements. No default shadcn/ui styling. |
| D-13 | **MEMORY.md updated** — [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) is appended with the phase completion entry, any assumptions made, and any deviations from spec. |

> [!WARNING]
> Skipping D-04 (four UI states) or D-07 (contract compliance) results in a component that **will break** when real model data is connected. These are non-negotiable.

---

## 4. File-Editing Protocol

### 4a. Read Before Write

| Rule | Detail |
|---|---|
| **Always read first** | Open and read the target file before making any edit. Never write to a file based on memory or assumption of its contents. |
| **Understand context** | Identify the surrounding code structure, existing patterns, and naming conventions before inserting new code. |

### 4b. Targeted Edits

| Rule | Detail |
|---|---|
| **Surgical changes** | Edit only the lines that need to change. Never rewrite an entire file to change a few lines. |
| **Preserve comments** | All existing comments and docstrings unrelated to your change must survive the edit. Deleting a colleague's documentation is a defect. |
| **Preserve formatting** | Match the existing indentation, bracket style, and whitespace conventions of the file. |

### 4c. Model-Integration Markers

| Rule | Detail |
|---|---|
| **TODO at seams** | Every point where mock data will be replaced by real model output must carry a `// TODO(model):` marker explaining what will change and referencing the `IModelService` method. |
| **Example** | `// TODO(model): Replace with IModelService.getClassification() — see ModelService.interface.ts` |

### 4d. Import Discipline

| Rule | Detail |
|---|---|
| **Canonical order** | External packages → `@shared/*` → `@design-system/*` → `@services/*` → feature-relative imports. |
| **Barrel imports** | Import from barrel `index.ts` files, not deep internal paths, unless a specific sub-module is required for tree-shaking. |
| **No circular imports** | A file must never transitively import itself. Feature barrel exports must be carefully curated to prevent cycles. |

---

## 5. When to Update MEMORY.md

[`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) is the project's living memory. It is **append-only** — never overwrite or delete previous entries.

### 5a. Mandatory Update Triggers

| Trigger | What to Log |
|---|---|
| **Phase completion** | Phase status change, what was built, any gate verification results. |
| **Assumption made** | The assumption text, which spec section it relates to, and the reasoning. |
| **Deviation from spec** | What was specified vs. what was implemented, and why. |
| **Open question logged** | The question, which spec section is unclear, and the blocking/non-blocking classification. |
| **Decision made** | The decision, alternatives considered, and rationale. |
| **Lessons learned** | Non-obvious discoveries, gotchas, or patterns worth remembering. |

### 5b. Entry Format

Every new entry must follow this structure:

```markdown
## YYYY-MM-DD — <Brief Title>

### Phase Status
- Phase N: <STATUS> — <one-line summary>

### Decisions Made
1. **<Decision title>**: <details>

### Assumptions
1. <Assumption text> — <spec reference>

### Open Questions
1. <Question text> — <blocking / non-blocking>

### Deviations
1. <Deviation description> — <rationale>
```

> [!IMPORTANT]
> **Never overwrite, reorder, or delete previous MEMORY.md entries.** Always append below the `<!-- APPEND NEW ENTRIES BELOW THIS LINE -->` marker. The log is a chronological, immutable record.

---

## 6. ML Boundary — Hard Line

> **The agent builds UI architecture. The agent does NOT build ML engineering.**

### 6a. What the Agent NEVER Does

| Forbidden Action | Explanation |
|---|---|
| **Write ML logic** | No signal processing, no feature extraction, no model training, no inference code. |
| **Import ML libraries** | No `tensorflow`, `pytorch`, `scikit-learn`, `scipy.signal`, or equivalent. |
| **Process real EEG data** | No parsing of `.edf`, `.csv`, or any raw EEG file format. |
| **Compute real metrics** | No actual accuracy/precision/recall/F1/AUC computation from model outputs. |
| **Implement SHAP** | No real SHAP value calculation. SHAP visualisations consume mock data only. |

### 6b. What the Agent DOES

| Permitted Action | Explanation |
|---|---|
| **Mock data generators** | Write functions that produce data **shaped exactly like the model-contracts** — correct types, realistic ranges, plausible distributions. |
| **`IModelService` consumers** | Build UI components that consume the `IModelService` interface without knowing whether the backing implementation is mock or real. |
| **TODO markers** | Place `// TODO(model):` markers at every seam where mock data will be swapped for real model output. |
| **Type definitions** | Define and maintain TypeScript interfaces and Zod schemas for all model data contracts. |

> [!CAUTION]
> The mock-to-real swap must be a **configuration change**, not a rewrite. If connecting a real model would require modifying any UI component, the abstraction boundary has been violated. See [`ARCHITECTURE.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/ARCHITECTURE.md) §2 — Layer 5 (Model Integration).

---

## 7. Quality Bars

Every deliverable must meet **all four** quality bars simultaneously.

### 7a. Visual — Premium Glassmorphic

| Criterion | Standard |
|---|---|
| **Design language** | Luxury, futuristic, glassmorphic AI command-center aesthetic. |
| **Glass panels** | Frosted-glass backgrounds with `backdrop-filter: blur()`, subtle borders, layered depth. |
| **Neon accents** | Neon glow effects on interactive elements, status indicators, and data highlights. |
| **No bare elements** | Every shadcn/ui component must be restyled to the design system. Default browser chrome is never acceptable. |
| **Consistent tokens** | All colours, typography, spacing, and effects are drawn from `design-system/tokens/`. No magic numbers. |

### 7b. Architectural — Five-Layer Compliance

| Criterion | Standard |
|---|---|
| **Layer separation** | Code lives in the correct layer directory as defined in [`ARCHITECTURE.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/ARCHITECTURE.md) §2. |
| **Dependency direction** | Imports flow downward only: UI → Hooks → API → State → Model. Never upward. |
| **Feature isolation** | `features/*` directories never import from each other. |
| **Cross-cutting packages** | `shared/` and `design-system/` are the only packages importable by all layers. |
| **Model abstraction** | All model access flows through `IModelService`. No component directly calls `fetch` or `axios` for model data. |

### 7c. Type Safety — Strict, No `any`

| Criterion | Standard |
|---|---|
| **Strict mode** | `tsconfig.json` enforces `"strict": true`. |
| **Zero `any`** | The `any` type is banned. Use `unknown` + type narrowing, generics, or explicit types. |
| **Zod at boundaries** | Every API response and external data input is validated by a Zod schema before entering business logic. |
| **Contract alignment** | Component props and store shapes are derived from (or identical to) `model-contracts` types. |

### 7d. State Coverage — Four States

Every async-driven UI component must render all four states:

```
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌─────────┐
│  idle    │ ──▶ │ loading  │ ──▶ │  success  │     │  error  │
│          │     │          │     │           │     │         │
│ (initial │     │ skeleton │     │ data      │     │ message │
│  state)  │     │ shimmer) │     │ rendered) │     │ + retry)│
└─────────┘     └──────────┘     └───────────┘     └─────────┘
                                       │                 ▲
                                       └── (refetch) ────┘
```

| State | Requirement |
|---|---|
| **Idle** | Default state before any data is requested. May show a placeholder or empty-state graphic. |
| **Loading** | `SkeletonShimmer` or glassmorphic loading indicator. Never a blank or invisible panel. |
| **Success** | Full data render matching the design-system aesthetic. |
| **Error** | In-panel error message with optional retry action. Never an unhandled crash or white screen. |

---

## 8. Phase Gate Protocol

> **No Phase N+1 work begins until Phase N is complete and its gate passes.**

### 8a. The Rule

| Principle | Detail |
|---|---|
| **Sequential execution** | Phases are defined in [`IMPLEMENTATIONPLAN.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/IMPLEMENTATIONPLAN.md) and must be completed in order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. |
| **Gate verification** | Every phase ends with a **Gate ✓** step. All checklist items within the phase must be marked `[x]` before the gate is attempted. |
| **Log in MEMORY.md** | When a phase gate passes, append a dated entry to [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) recording the phase status, what was built, and any notes. |
| **No skipping** | Jumping to a later phase because it seems "easier" or "more interesting" is a **blocking violation**. |
| **If unsure, re-read** | When uncertain whether a phase is truly complete, re-read [`IMPLEMENTATIONPLAN.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/IMPLEMENTATIONPLAN.md) for that phase's checklist and verify each item. |

### 8b. Gate Failure

If a gate verification step **fails**:

| Step | Action |
|---|---|
| 1 | Identify the failing checklist item(s). |
| 2 | Log the failure in [`MEMORY.md`](file:///Users/tirthkosambia/Documents/NeuroAegis/docs/MEMORY.md) with details. |
| 3 | Fix the failing item(s) within the current phase. |
| 4 | Re-attempt the gate. |
| 5 | Only proceed to the next phase after a clean gate pass. |

### 8c. Phase Reference

| Phase | Name | Gate Criterion |
|---|---|---|
| 0 | Foundations | `pnpm build` succeeds with zero errors. |
| 1 | Design System | Themed glass shell renders with correct token values. |
| 2 | Shell & Navigation | Navigation between all routes works without errors. |
| 3 | Core Layer | Mock data flows through stores and renders in devtools. |
| 4 | Feature Modules | All four UI states render correctly for every feature panel. |
| 5 | Hero Visualization | ≥ 50 fps, no dropped frames on 60 Hz monitor. |
| 6 | Polish | Visual audit, accessibility, performance pass complete. |
| 7 | Model-Readiness Review | All gates 0–6 pass; all checklist items `[x]`. |

---

> [!TIP]
> **When in doubt, apply the Golden Rule:** If it's not in the spec, don't guess — log it, proceed with the nearest constraint, and treat silence as a stop sign.

---

_Document maintained by the NeuroAegis engineering team._
_Last updated: 2026-07-18_
