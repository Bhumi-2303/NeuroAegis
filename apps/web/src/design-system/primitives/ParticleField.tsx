import { useEffect, useRef } from 'react';

export interface ParticleFieldProps {
  count?: number;
  className?: string;
}

export const ParticleField = ({ count = 40, className = '' }: ParticleFieldProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let particles: { x: number; y: number; size: number; speedY: number; opacity: number; opacitySpeed: number }[] = [];
    let animationFrameId: number;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      ctx.scale(dpr, dpr);
    };
    
    window.addEventListener('resize', resize);
    resize();

    const logicalWidth = window.innerWidth;
    const logicalHeight = window.innerHeight;

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * logicalWidth,
        y: Math.random() * logicalHeight,
        size: Math.random() * 1.5 + 0.5,
        speedY: Math.random() * -0.5 - 0.1,
        opacity: Math.random() * 0.4 + 0.1,
        opacitySpeed: (Math.random() * 0.02) - 0.01,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, logicalWidth, logicalHeight);
      
      particles.forEach((p) => {
        if (!prefersReducedMotion) {
          p.y += p.speedY;
          p.opacity += p.opacitySpeed;
          
          if (p.opacity <= 0.1 || p.opacity >= 0.5) {
            p.opacitySpeed *= -1;
          }

          if (p.y < 0) {
            p.y = logicalHeight;
            p.x = Math.random() * logicalWidth;
          }
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(20, 184, 166, ${p.opacity})`;
        ctx.fill();
      });

      if (!prefersReducedMotion) {
        animationFrameId = requestAnimationFrame(render);
      }
    };

    render();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [count]);

  return (
    <canvas
      ref={canvasRef}
      className={`fixed inset-0 pointer-events-none z-0 ${className}`}
    />
  );
};
