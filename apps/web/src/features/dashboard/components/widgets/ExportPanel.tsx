import { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { Download, FileJson, FileText, FileSpreadsheet, ActivitySquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  data?: any;
  datasetName?: string | null;
}

export function ExportPanel({ data, datasetName }: Props) {
  const exports = [
    { id: 'pdf', label: 'Export PDF Report', icon: <FileText size={16} /> },
    { id: 'csv', label: 'Export Features (CSV)', icon: <FileSpreadsheet size={16} /> },
    { id: 'json', label: 'Export Raw JSON', icon: <FileJson size={16} /> },
    { id: 'shap', label: 'SHAP Analysis (CSV)', icon: <ActivitySquare size={16} /> },
  ];

  const [downloading, setDownloading] = useState<string | null>(null);

  const downloadFile = (content: string, fileName: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownload = (id: string) => {
    if (!data && id !== 'pdf') return;
    setDownloading(id);

    setTimeout(() => {
      if (id === 'pdf') {
        window.print();
      } else if (id === 'json') {
        downloadFile(JSON.stringify(data, null, 2), `neuroaegis_${datasetName || 'report'}.json`, 'application/json');
      } else if (id === 'csv') {
        if (data?.explanation?.features) {
          const headers = 'Feature Name,Computed Value\n';
          const rows = data.explanation.features.map((f: any) => `"${f.featureName}",${f.originalValue}`).join('\n');
          downloadFile(headers + rows, `neuroaegis_features.csv`, 'text/csv');
        }
      } else if (id === 'shap') {
        if (data?.explanation?.features) {
          const headers = 'Feature Name,SHAP Value\n';
          const rows = data.explanation.features.map((f: any) => `"${f.featureName}",${f.value}`).join('\n');
          downloadFile(headers + rows, `neuroaegis_shap.csv`, 'text/csv');
        }
      }
      setDownloading(null);
    }, 800);
  };

  return (
    <WidgetCard title="Export Options" icon={<Download size={16} />}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 h-full">
        {exports.map(exp => (
          <button
            key={exp.id}
            onClick={() => handleDownload(exp.id)}
            disabled={downloading !== null || (!data && exp.id !== 'pdf')}
            className="relative overflow-hidden flex items-center justify-center gap-2 p-3 text-sm font-medium text-[var(--text-primary)] bg-transparent border border-[var(--bg-4)] rounded-xl hover:bg-[var(--bg-3)] transition-colors disabled:opacity-80"
          >
            <div className="relative z-10 flex items-center gap-2">
              <span className="text-[var(--text-muted)]">{exp.icon}</span>
              {downloading === exp.id ? 'Exporting...' : exp.label}
            </div>
            
            <AnimatePresence>
              {downloading === exp.id && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                  className="absolute left-0 top-0 bottom-0 bg-[var(--accent-primary)]/20 z-0"
                />
              )}
            </AnimatePresence>
          </button>
        ))}
      </div>
    </WidgetCard>
  );
}
