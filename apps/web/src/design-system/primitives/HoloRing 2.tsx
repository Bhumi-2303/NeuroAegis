import { forwardRef } from 'react';
import { motion } from 'framer-motion';
import type { HTMLMotionProps } from 'framer-motion';

export interface HoloRingProps extends HTMLMotionProps<"div"> {
  size?: number;
  color?: string;
  delay?: number;
  duration?: number;
}

export const HoloRing = forwardRef<HTMLDivElement, HoloRingProps>(
  ({ size = 200, color = 'rgba(0, 229, 255, 0.2)', delay = 0, duration = 10, className = '', ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        className={`absolute rounded-full border border-solid pointer-events-none ${className}`}
        style={{
          width: size,
          height: size,
          borderColor: color,
          borderWidth: 1,
          boxShadow: `0 0 10px ${color}, inset 0 0 10px ${color}`,
        }}
        animate={{
          rotateX: [0, 180, 360],
          rotateY: [0, 360, 180],
          rotateZ: [0, 180, 360],
        }}
        transition={{
          duration,
          repeat: Infinity,
          ease: "linear",
          delay,
        }}
        {...props}
      />
    );
  }
);

HoloRing.displayName = 'HoloRing';
