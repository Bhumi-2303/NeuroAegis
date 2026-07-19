import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { motion } from 'framer-motion';

export interface ModelSelectorSegmentedProps extends HTMLAttributes<HTMLDivElement> {
  models: readonly string[];
  selected: string;
  onSelect: (model: string) => void;
  className?: string;
}

/**
 * ModelSelectorSegmented — single-select segmented pill control
 * for model selection (Random Forest / XGBoost / LightGBM).
 */
export const ModelSelectorSegmented = forwardRef<HTMLDivElement, ModelSelectorSegmentedProps>(
  ({ models, selected, onSelect, className = '', ...props }, ref) => {
    const displayName = (model: string): string => {
      const nameMap: Record<string, string> = {
        random_forest: 'Random Forest',
        xgboost: 'XGBoost',
        lightgbm: 'LightGBM',
      };
      return nameMap[model] || model;
    };

    return (
      <div
        ref={ref}
        className={`inline-flex items-center p-1 rounded-full bg-[rgba(11,22,37,0.8)] border border-[rgba(255,255,255,0.08)] backdrop-blur-sm ${className}`}
        role="radiogroup"
        aria-label="Model Selector"
        {...props}
      >
        {models.map((model) => {
          const isSelected = model === selected;
          return (
            <button
              key={model}
              role="radio"
              aria-checked={isSelected}
              onClick={() => onSelect(model)}
              className={`relative px-4 py-1.5 text-xs font-mono rounded-full transition-colors duration-200 outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-primary)] ${
                isSelected
                  ? 'text-[var(--bg-1)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              {isSelected && (
                <motion.div
                  layoutId="model-selector-pill"
                  className="absolute inset-0 rounded-full"
                  style={{
                    background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                    boxShadow: '0 0 16px rgba(0, 229, 255, 0.3)',
                  }}
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
              <span className="relative z-10 whitespace-nowrap">
                {displayName(model)}
              </span>
            </button>
          );
        })}
      </div>
    );
  }
);

ModelSelectorSegmented.displayName = 'ModelSelectorSegmented';
