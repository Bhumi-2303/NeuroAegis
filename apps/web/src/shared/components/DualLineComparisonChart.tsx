import type { HTMLAttributes } from 'react';
import { forwardRef, useId } from 'react';

export interface DualLineDataPoint {
  x: number;
  y1: number;
  y2?: number;
}

export interface DualLineComparisonChartProps extends HTMLAttributes<HTMLDivElement> {
  data: DualLineDataPoint[];
  label1?: string;
  label2?: string;
  color1?: string;
  color2?: string;
  showDiagonal?: boolean;
  className?: string;
}

/**
 * DualLineComparisonChart — two datasets as dotted/marker lines
 * overlaid on fainter background lines. Used for ROC curves or
 * two-model confidence comparisons.
 */
export const DualLineComparisonChart = forwardRef<HTMLDivElement, DualLineComparisonChartProps>(
  (
    {
      data,
      label1 = 'Dataset 1',
      label2 = 'Dataset 2',
      color1 = 'var(--accent-primary)',
      color2 = 'var(--accent-highlight)',
      showDiagonal = false,
      className = '',
      ...props
    },
    ref
  ) => {
    const gradId1 = useId();
    const gradId2 = useId();

    const viewWidth = 400;
    const viewHeight = 240;
    const pad = { top: 20, right: 20, bottom: 30, left: 35 };

    const chartW = viewWidth - pad.left - pad.right;
    const chartH = viewHeight - pad.top - pad.bottom;

    if (!data || data.length === 0) {
      return (
        <div ref={ref} className={className} {...props}>
          <div className="flex items-center justify-center h-48 text-xs text-[var(--text-secondary)] font-mono">
            No data available
          </div>
        </div>
      );
    }

    const maxX = Math.max(...data.map((d) => d.x), 1);
    const maxY1 = Math.max(...data.map((d) => d.y1), 1);
    const maxY2 = data.some((d) => d.y2 !== undefined)
      ? Math.max(...data.filter((d) => d.y2 !== undefined).map((d) => d.y2!), 1)
      : 0;
    const maxY = Math.max(maxY1, maxY2, 1);

    const toSvgX = (x: number) => pad.left + (x / maxX) * chartW;
    const toSvgY = (y: number) => pad.top + chartH - (y / maxY) * chartH;

    const buildPath = (values: { x: number; y: number }[]) =>
      values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(v.x)} ${toSvgY(v.y)}`).join(' ');

    const path1 = buildPath(data.map((d) => ({ x: d.x, y: d.y1 })));
    const hasY2 = data.some((d) => d.y2 !== undefined);
    const path2 = hasY2
      ? buildPath(data.filter((d) => d.y2 !== undefined).map((d) => ({ x: d.x, y: d.y2! })))
      : '';

    // Fill area paths
    const fillPath1 = `${path1} L ${toSvgX(data[data.length - 1].x)} ${toSvgY(0)} L ${toSvgX(data[0].x)} ${toSvgY(0)} Z`;

    // Grid lines
    const gridLines = [0, 0.25, 0.5, 0.75, 1];

    return (
      <div ref={ref} className={`w-full ${className}`} {...props}>
        <svg
          width="100%"
          viewBox={`0 0 ${viewWidth} ${viewHeight}`}
          preserveAspectRatio="xMidYMid meet"
          className="overflow-visible"
        >
          <defs>
            <linearGradient id={gradId1} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color1} stopOpacity={0.15} />
              <stop offset="100%" stopColor={color1} stopOpacity={0} />
            </linearGradient>
            <linearGradient id={gradId2} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color2} stopOpacity={0.1} />
              <stop offset="100%" stopColor={color2} stopOpacity={0} />
            </linearGradient>
          </defs>

          {/* Grid */}
          {gridLines.map((v) => (
            <g key={v}>
              <line
                x1={pad.left}
                y1={toSvgY(v * maxY)}
                x2={viewWidth - pad.right}
                y2={toSvgY(v * maxY)}
                stroke="rgba(255,255,255,0.05)"
                strokeDasharray="3 3"
              />
              <text
                x={pad.left - 6}
                y={toSvgY(v * maxY) + 3}
                fill="var(--text-secondary)"
                fontSize={9}
                fontFamily="var(--font-mono)"
                textAnchor="end"
              >
                {(v * maxY).toFixed(1)}
              </text>
            </g>
          ))}

          {/* Diagonal reference */}
          {showDiagonal && (
            <line
              x1={toSvgX(0)}
              y1={toSvgY(0)}
              x2={toSvgX(maxX)}
              y2={toSvgY(maxY)}
              stroke="rgba(255,255,255,0.1)"
              strokeDasharray="4 4"
            />
          )}

          {/* Fill area for line 1 */}
          <path d={fillPath1} fill={`url(#${gradId1})`} />

          {/* Line 1 */}
          <path
            d={path1}
            fill="none"
            stroke={color1}
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ filter: `drop-shadow(0 0 4px ${color1})` }}
          />

          {/* Dots for line 1 */}
          {data.map((d, i) => (
            <circle
              key={`d1-${i}`}
              cx={toSvgX(d.x)}
              cy={toSvgY(d.y1)}
              r={2.5}
              fill={color1}
              style={{ filter: `drop-shadow(0 0 3px ${color1})` }}
            />
          ))}

          {/* Line 2 (dotted) */}
          {hasY2 && (
            <>
              <path
                d={path2}
                fill="none"
                stroke={color2}
                strokeWidth={1.5}
                strokeDasharray="4 3"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ filter: `drop-shadow(0 0 4px ${color2})` }}
              />
              {data
                .filter((d) => d.y2 !== undefined)
                .map((d, i) => (
                  <rect
                    key={`d2-${i}`}
                    x={toSvgX(d.x) - 2.5}
                    y={toSvgY(d.y2!) - 2.5}
                    width={5}
                    height={5}
                    fill={color2}
                    rx={1}
                    style={{ filter: `drop-shadow(0 0 3px ${color2})` }}
                  />
                ))}
            </>
          )}

          {/* X-axis labels */}
          {[0, 0.5, 1].map((v) => (
            <text
              key={`x-${v}`}
              x={toSvgX(v * maxX)}
              y={viewHeight - 6}
              fill="var(--text-secondary)"
              fontSize={9}
              fontFamily="var(--font-mono)"
              textAnchor="middle"
            >
              {(v * maxX).toFixed(1)}
            </text>
          ))}
        </svg>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-2 px-2">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 rounded-full" style={{ background: color1 }} />
            <span className="text-[10px] font-mono text-[var(--text-secondary)]">{label1}</span>
          </div>
          {hasY2 && (
            <div className="flex items-center gap-1.5">
              <div
                className="w-3 h-0.5 rounded-full"
                style={{ background: color2, opacity: 0.7 }}
              />
              <span className="text-[10px] font-mono text-[var(--text-secondary)]">{label2}</span>
            </div>
          )}
        </div>
      </div>
    );
  }
);

DualLineComparisonChart.displayName = 'DualLineComparisonChart';
