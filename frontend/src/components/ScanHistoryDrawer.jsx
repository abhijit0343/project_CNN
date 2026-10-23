// components/ScanHistoryDrawer.jsx
import React, { useState, useEffect } from 'react';
import { History, Trash2, Clock, ChevronRight, X } from 'lucide-react';
import { CLASS_METADATA } from '../data/model';

export default function ScanHistoryDrawer({ isOpen, onClose, onSelectScan }) {
  const [history, setHistory] = useState([]);

  const loadHistory = () => {
    try {
      const stored = localStorage.getItem('neuroscan-history');
      if (stored) {
        setHistory(JSON.parse(stored));
      }
    } catch {
      setHistory([]);
    }
  };

  useEffect(() => {
    loadHistory();
    const handleUpdate = () => loadHistory();
    window.addEventListener('neuroscan-history-updated', handleUpdate);
    return () => window.removeEventListener('neuroscan-history-updated', handleUpdate);
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('neuroscan-history');
    setHistory([]);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/50 backdrop-blur-sm flex justify-end animate-in fade-in duration-200">
      <div className="w-full max-w-md bg-[#FBFCFA] h-full shadow-2xl border-l border-[#DCE3E2] flex flex-col justify-between p-6 overflow-y-auto">
        {/* Header */}
        <div>
          <div className="flex items-center justify-between pb-4 border-b border-[#E3EAE9]">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-[#D62839]" />
              <h3 className="font-mono text-xs uppercase tracking-widest text-[#003153] font-bold">
                LOCAL SCAN CACHE
              </h3>
            </div>
            <button
              onClick={onClose}
              className="p-1 text-[#647C89] hover:text-[#003153] hover:bg-[#F2F6F5] transition-colors"
              aria-label="Close History"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <p className="text-xs text-[#718590] mt-3 leading-relaxed">
            Recently analyzed scans stored securely in your browser's local sandbox. Click any record to restore its full model path and Grad-CAM attention.
          </p>

          {/* List */}
          <div className="mt-6 space-y-3">
            {history.length === 0 ? (
              <div className="py-12 text-center text-xs text-[#8799A3] border border-dashed border-[#CBD8D9] p-6">
                No previous scans recorded. Upload an MRI or test a sample scan to view history here.
              </div>
            ) : (
              history.map((scan, idx) => {
                const meta = CLASS_METADATA[scan.prediction] || CLASS_METADATA.glioma;
                const isHealthy = scan.prediction === 'notumor';

                return (
                  <div
                    key={`${scan.filename}-${scan.timestamp}-${idx}`}
                    onClick={() => {
                      onSelectScan(scan);
                      onClose();
                    }}
                    className="p-3 bg-white border border-[#DCE3E2] hover:border-[#003153] cursor-pointer transition-all flex items-center justify-between group shadow-xs"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {scan.previewUrl ? (
                        <div className="w-10 h-10 rounded-sm bg-black overflow-hidden shrink-0 border border-[#D0DBDE]">
                          <img src={scan.previewUrl} alt={scan.filename} className="w-full h-full object-cover" />
                        </div>
                      ) : (
                        <div className="w-10 h-10 rounded-sm bg-[#003153] text-white flex items-center justify-center font-mono text-[9px] shrink-0">
                          MRI
                        </div>
                      )}

                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs font-bold truncate ${
                              isHealthy ? 'text-[#003153]' : 'text-[#D62839]'
                            }`}
                          >
                            {meta.name}
                          </span>
                          <span className="text-[9px] font-mono px-1 bg-[#F2F6F5] text-[#556F7E]">
                            {(scan.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <span className="text-[10px] text-[#788C97] font-mono truncate block mt-0.5">
                          {scan.filename}
                        </span>
                        <span className="text-[9px] text-[#9CB0BA] font-mono flex items-center gap-1 mt-0.5">
                          <Clock className="w-2.5 h-2.5" />
                          {new Date(scan.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                    </div>

                    <ChevronRight className="w-4 h-4 text-[#A1B3BE] group-hover:text-[#D62839] group-hover:translate-x-1 transition-all" />
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer */}
        {history.length > 0 && (
          <div className="pt-4 border-t border-[#E3EAE9] flex justify-between items-center text-xs">
            <span className="text-[10px] font-mono text-[#8C9FA9]">{history.length} scans saved</span>
            <button
              onClick={clearHistory}
              className="inline-flex items-center gap-1.5 text-xs text-[#D62839] hover:text-[#941B26] font-mono font-semibold"
            >
              <Trash2 className="w-3.5 h-3.5" /> Clear History
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
