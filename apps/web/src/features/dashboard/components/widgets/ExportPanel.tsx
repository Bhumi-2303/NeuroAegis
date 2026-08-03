import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { Download, FileJson, FileText, Image, FileSpreadsheet, ActivitySquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export function ExportPanel() {
  const exports = [
    { id: 'pdf', label: 'Export PDF Report', icon: <FileText size={16} /> },
    { id: 'csv', label: 'Export Data (CSV)', icon: <FileSpreadsheet size={16} /> },
    { id: 'json', label: 'Export Raw JSON', icon: <FileJson size={16} /> },
    { id: 'shap', label: 'SHAP Analysis', icon: <ActivitySquare size={16} /> },
    { id: 'img', label: 'Signal Image', icon: <Image size={16} /> },
  ];

  const [downloading, setDownloading] = useState<string | null>(null);

  const handleDownload = (id: string) => {
    setDownloading(id);
    setTimeout(() => setDownloading(null), 1500); // mock animation
  };

  return (
    <WidgetCard title="Export Options" icon={<Download size={16} />}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 h-full">
        {exports.map(exp => (
          <button
            key={exp.id}
            onClick={() => handleDownload(exp.id)}
            disabled={downloading !== null}
            className="relative overflow-hidden flex items-center justify-center gap-2 p-3 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors disabled:opacity-80"
          >
            <div className="relative z-10 flex items-center gap-2">
              <span className="text-gray-400">{exp.icon}</span>
              {downloading === exp.id ? 'Exporting...' : exp.label}
            </div>
            
            <AnimatePresence>
              {downloading === exp.id && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1.2, ease: 'easeOut' }}
                  className="absolute left-0 top-0 bottom-0 bg-blue-50/50 z-0"
                />
              )}
            </AnimatePresence>
          </button>
        ))}
      </div>
    </WidgetCard>
  );
}
