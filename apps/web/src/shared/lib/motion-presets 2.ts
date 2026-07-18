// ── src/shared/lib/motion-presets.ts ──
// Framer Motion presets as defined in DESIGN.md § 9.2
// These are the canonical animation presets for the entire NeuroAegis system.

export const cardFloat = {
  animate: { y: [0, -6, 0] },
  transition: { duration: 4, repeat: Infinity, ease: 'easeInOut' as const },
};

export const hoverElevate = {
  whileHover: { y: -2, boxShadow: '0 0 30px rgba(0,229,255,0.10)' },
  transition: { duration: 0.3, ease: 'easeOut' as const },
};

export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { duration: 0.5, ease: 'easeOut' as const },
};

export const slideUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: 'easeOut' as const },
};

export const pulseGlow = {
  animate: {
    boxShadow: [
      '0 0 10px rgba(0,229,255,0.05)',
      '0 0 25px rgba(0,229,255,0.12)',
      '0 0 10px rgba(0,229,255,0.05)',
    ],
  },
  transition: { duration: 3, repeat: Infinity, ease: 'easeInOut' as const },
};

export const waveformDraw = {
  initial: { pathLength: 0, opacity: 0 },
  animate: { pathLength: 1, opacity: 1 },
  transition: { duration: 1.2, ease: 'easeOut' as const },
};

export const staggerChildren = {
  animate: { transition: { staggerChildren: 0.08 } },
};

export const pageTransition = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
  transition: { duration: 0.4, ease: 'easeOut' as const },
};
