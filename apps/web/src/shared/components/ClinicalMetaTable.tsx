import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';

export interface ClinicalMetaRow {
  attribute: string;
  value: string | number;
}

export interface ClinicalMetaTableProps extends HTMLAttributes<HTMLDivElement> {
  rows: ClinicalMetaRow[];
  className?: string;
}

/**
 * ClinicalMetaTable — compact two-column Attribute/Value table
 * for session/patient metadata with tight row height.
 */
export const ClinicalMetaTable = forwardRef<HTMLDivElement, ClinicalMetaTableProps>(
  ({ rows, className = '', ...props }, ref) => {
    if (!rows || rows.length === 0) return null;

    return (
      <div
        ref={ref}
        className={`w-full overflow-hidden ${className}`}
        {...props}
      >
        <table className="w-full border-collapse" role="table">
          <thead>
            <tr>
              <th
                className="text-left text-[10px] font-mono uppercase tracking-[0.15em] text-[var(--text-secondary)] pb-2 border-b border-[rgba(255,255,255,0.06)]"
              >
                Attribute
              </th>
              <th
                className="text-right text-[10px] font-mono uppercase tracking-[0.15em] text-[var(--text-secondary)] pb-2 border-b border-[rgba(255,255,255,0.06)]"
              >
                Value
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr
                key={`${row.attribute}-${idx}`}
                className="border-b border-[rgba(255,255,255,0.03)] last:border-b-0 hover:bg-[rgba(0,229,255,0.03)] transition-colors duration-200"
              >
                <td className="py-1.5 text-xs font-body text-[var(--text-secondary)]">
                  {row.attribute}
                </td>
                <td className="py-1.5 text-xs font-mono text-[var(--text-primary)] text-right">
                  {row.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
);

ClinicalMetaTable.displayName = 'ClinicalMetaTable';
