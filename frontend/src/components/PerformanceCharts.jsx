// components/PerformanceCharts.jsx
import React, { useState } from 'react';
import { EVALUATION_METRICS, TRAINING_CURVES } from '../data/model';

export default function PerformanceCharts() {
  const [activeTab, setActiveTab] = useState('curves');
  const [curveType, setCurveType] = useState('accuracy');

  const p1 = TRAINING_CURVES.phase1;
  const p2 = TRAINING_CURVES.phase2;
  const allEpochs = [...p1.epochs, ...p2.epochs];

  const trainData = curveType === 'accuracy' ? [...p1.trainAcc, ...p2.trainAcc] : [...p1.trainLoss, ...p2.trainLoss];
  const valData = curveType === 'accuracy' ? [...p1.valAcc, ...p2.valAcc] : [...p1.valLoss, ...p2.valLoss];

  const svgWidth = 640;
  const svgHeight = 240;
  const padding = { top: 20, right: 20, bottom: 35, left: 45 };
  const graphWidth = svgWidth - padding.left - padding.right;
  const graphHeight = svgHeight - padding.top - padding.bottom;

  const minVal = curveType === 'accuracy' ? 75 : 0.2;
  const maxVal = curveType === 'accuracy' ? 100 : 0.8;

  const getX = (idx) => padding.left + (idx / (allEpochs.length - 1)) * graphWidth;
  const getY = (val) => padding.top + graphHeight - ((val - minVal) / (maxVal - minVal)) * graphHeight;

  const trainPath = trainData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d)}`).join(' ');
  const valPath = valData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d)}`).join(' ');

  const phase2StartX = getX(p1.epochs.length);

  return (
    <div className="bg-[#00253F] text-white p-6 sm:p-8 border border-[#1B4561] shadow-xl">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#1B4561]">
        <div>
          <span className="font-mono text-[10px] tracking-widest uppercase text-[#8EAAB8] font-semibold">
            04 / TEST SET EVALUATION BENCHMARKS
          </span>
          <h3 className="text-2xl sm:text-3xl font-display mt-1 text-white">
            Measured on <em>1,600 held-out scans.</em>
          </h3>
        </div>

        {/* Tab Controls */}
        <div className="inline-flex border border-[#2B5470] bg-[#001D33] p-0.5 text-xs font-mono">
          <button
            onClick={() => setActiveTab('curves')}
            className={`px-3 py-1.5 transition-colors ${
              activeTab === 'curves' ? 'bg-[#D62839] text-white font-bold' : 'text-[#8CA6B5] hover:text-white'
            }`}
          >
            Learning Curves
          </button>
          <button
            onClick={() => setActiveTab('confusion')}
            className={`px-3 py-1.5 transition-colors ${
              activeTab === 'confusion' ? 'bg-[#D62839] text-white font-bold' : 'text-[#8CA6B5] hover:text-white'
            }`}
          >
            Confusion Matrix
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-3 py-1.5 transition-colors ${
              activeTab === 'metrics' ? 'bg-[#D62839] text-white font-bold' : 'text-[#8CA6B5] hover:text-white'
            }`}
          >
            Per-Class Metrics
          </button>
        </div>
      </div>

      {/* Main KPI Stat Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 py-6 border-b border-[#1B4561]">
        <div className="bg-[#001F35] border border-[#163E5A] p-4">
          <span className="text-[9px] font-mono text-[#8EAAB8] uppercase tracking-wider block">
            Overall Test Accuracy
          </span>
          <div className="font-display text-4xl sm:text-5xl text-white mt-1">
            {EVALUATION_METRICS.testAccuracy}
            <span className="text-xl font-sans text-[#D62839]">%</span>
          </div>
          <span className="text-[9px] text-[#718F9F] font-mono mt-1 block">
            1,489 / 1,600 correctly classified
          </span>
        </div>

        <div className="bg-[#001F35] border border-[#163E5A] p-4">
          <span className="text-[9px] font-mono text-[#8EAAB8] uppercase tracking-wider block">
            Macro F1-Score
          </span>
          <div className="font-display text-4xl sm:text-5xl text-white mt-1">
            {EVALUATION_METRICS.macroF1}
            <span className="text-xl font-sans text-[#8EAAB8]">%</span>
          </div>
          <span className="text-[9px] text-[#718F9F] font-mono mt-1 block">
            Harmonic balance across all 4 classes
          </span>
        </div>

        <div className="bg-[#001F35] border border-[#163E5A] p-4">
          <span className="text-[9px] font-mono text-[#8EAAB8] uppercase tracking-wider block">
            Glioma Precision
          </span>
          <div className="font-display text-4xl sm:text-5xl text-white mt-1">
            97.86
            <span className="text-xl font-sans text-[#8EAAB8]">%</span>
          </div>
          <span className="text-[9px] text-[#718F9F] font-mono mt-1 block">
            Minimal false positive detections
          </span>
        </div>

        <div className="bg-[#001F35] border border-[#163E5A] p-4">
          <span className="text-[9px] font-mono text-[#8EAAB8] uppercase tracking-wider block">
            Normal / Healthy Recall
          </span>
          <div className="font-display text-4xl sm:text-5xl text-[#D62839] mt-1">
            100.0
            <span className="text-xl font-sans text-white">%</span>
          </div>
          <span className="text-[9px] text-[#718F9F] font-mono mt-1 block">
            0 healthy scans misclassified as tumor
          </span>
        </div>
      </div>

      {/* Tab 1: Interactive Learning Curves */}
      {activeTab === 'curves' && (
        <div className="mt-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="text-[#8EAAB8]">Metric:</span>
              <button
                onClick={() => setCurveType('accuracy')}
                className={`px-2.5 py-1 rounded-sm transition-colors ${
                  curveType === 'accuracy' ? 'bg-[#D62839] text-white font-bold' : 'text-[#8EAAB8] hover:text-white'
                }`}
              >
                Validation Accuracy (%)
              </button>
              <button
                onClick={() => setCurveType('loss')}
                className={`px-2.5 py-1 rounded-sm transition-colors ${
                  curveType === 'loss' ? 'bg-[#D62839] text-white font-bold' : 'text-[#8EAAB8] hover:text-white'
                }`}
              >
                Cross-Entropy Loss
              </button>
            </div>

            <div className="flex items-center gap-4 text-[10px] font-mono text-[#8EAAB8]">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-[#4F92BC]" /> Training Set
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-[#D62839]" /> Validation Set
              </span>
            </div>
          </div>

          <div className="w-full overflow-x-auto bg-[#001F35] border border-[#163E5A] p-3">
            <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto min-w-[540px]">
              {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                const y = padding.top + ratio * graphHeight;
                const labelVal = maxVal - ratio * (maxVal - minVal);
                return (
                  <g key={ratio}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={svgWidth - padding.right}
                      y2={y}
                      stroke="rgba(189, 210, 223, 0.1)"
                      strokeDasharray="3 3"
                    />
                    <text x={padding.left - 8} y={y + 4} textAnchor="end" fill="#718F9F" fontSize="9" fontFamily="monospace">
                      {curveType === 'accuracy' ? `${Math.round(labelVal)}%` : labelVal.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              <line
                x1={phase2StartX}
                y1={padding.top}
                x2={phase2StartX}
                y2={svgHeight - padding.bottom}
                stroke="#D62839"
                strokeDasharray="4 4"
                strokeWidth="1.2"
              />
              <text x={phase2StartX - 6} y={padding.top + 14} textAnchor="end" fill="#8EAAB8" fontSize="8" fontFamily="monospace">
                PHASE 1 (FROZEN)
              </text>
              <text x={phase2StartX + 6} y={padding.top + 14} textAnchor="start" fill="#D62839" fontSize="8" fontFamily="monospace">
                PHASE 2 (FINE-TUNE)
              </text>

              <path d={trainPath} fill="none" stroke="#4F92BC" strokeWidth="2" strokeLinecap="round" />
              <path d={valPath} fill="none" stroke="#D62839" strokeWidth="2.4" strokeLinecap="round" />

              {valData.map((d, i) => (
                <circle
                  key={i}
                  cx={getX(i)}
                  cy={getY(d)}
                  r={i === valData.length - 1 ? 4 : 2}
                  fill={i === valData.length - 1 ? '#D62839' : '#FFFFFF'}
                  stroke="#001F35"
                  strokeWidth="1"
                />
              ))}

              {allEpochs.filter((_, i) => i % 4 === 0 || i === allEpochs.length - 1).map((ep) => {
                const actualIndex = allEpochs.indexOf(ep);
                return (
                  <text
                    key={ep}
                    x={getX(actualIndex)}
                    y={svgHeight - 12}
                    textAnchor="middle"
                    fill="#718F9F"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    Ep {ep}
                  </text>
                );
              })}
            </svg>
          </div>
        </div>
      )}

      {/* Tab 2: Confusion Matrix */}
      {activeTab === 'confusion' && (
        <div className="mt-6 space-y-4">
          <div className="flex justify-between items-center text-xs font-mono text-[#8EAAB8]">
            <span>TEST SET CONFUSION MATRIX (1,600 IMAGES, 400 PER CLASS)</span>
            <span>DIAGONAL = TRUE POSITIVES</span>
          </div>

          <div className="overflow-x-auto bg-[#001F35] border border-[#163E5A] p-4">
            <table className="w-full text-center text-xs border-collapse">
              <thead>
                <tr className="border-b border-[#2C536E] text-[#8EAAB8] font-mono text-[10px]">
                  <th className="p-3 text-left">ACTUAL \ PREDICTED</th>
                  <th className="p-3">GLIOMA</th>
                  <th className="p-3">MENINGIOMA</th>
                  <th className="p-3">NO TUMOR</th>
                  <th className="p-3">PITUITARY</th>
                  <th className="p-3 text-right">RECALL</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#163E5A]">
                {['Glioma', 'Meningioma', 'No tumor', 'Pituitary'].map((clsName, rowIdx) => {
                  const row = EVALUATION_METRICS.confusionMatrix[rowIdx];
                  const total = row.reduce((a, b) => a + b, 0);
                  const recall = ((row[rowIdx] / total) * 100).toFixed(1);

                  return (
                    <tr key={clsName} className="hover:bg-white/5 transition-colors">
                      <td className="p-3 text-left font-bold font-mono text-[#CAD7DC]">
                        {clsName}
                      </td>
                      {row.map((val, colIdx) => {
                        const isDiag = rowIdx === colIdx;
                        return (
                          <td
                            key={colIdx}
                            className={`p-3 font-mono font-bold transition-all ${
                              isDiag
                                ? 'bg-[#D62839]/25 text-white ring-1 ring-[#D62839]'
                                : val > 0
                                ? 'text-[#8EAAB8]'
                                : 'text-white/20'
                            }`}
                          >
                            {val}
                          </td>
                        );
                      })}
                      <td className="p-3 text-right font-mono font-bold text-[#4F92BC]">
                        {recall}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Detailed Per-Class Breakdown */}
      {activeTab === 'metrics' && (
        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {EVALUATION_METRICS.classBreakdown.map((item) => (
            <div key={item.name} className="bg-[#001F35] border border-[#163E5A] p-4 space-y-3">
              <div className="flex justify-between items-baseline border-b border-[#2C536E] pb-2">
                <h4 className="font-bold text-white text-sm">{item.name}</h4>
                <span className="text-[9px] font-mono text-[#8EAAB8]">{item.support} scans</span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between text-[#CAD7DC]">
                  <span className="font-mono text-[10px]">Precision</span>
                  <span className="font-mono font-bold text-white">{item.precision}%</span>
                </div>
                <div className="w-full h-1 bg-[#163E5A] rounded-full overflow-hidden">
                  <div className="h-full bg-[#4F92BC]" style={{ width: `${item.precision}%` }} />
                </div>

                <div className="flex justify-between text-[#CAD7DC]">
                  <span className="font-mono text-[10px]">Recall</span>
                  <span className="font-mono font-bold text-white">{item.recall}%</span>
                </div>
                <div className="w-full h-1 bg-[#163E5A] rounded-full overflow-hidden">
                  <div className="h-full bg-[#D62839]" style={{ width: `${item.recall}%` }} />
                </div>

                <div className="flex justify-between text-[#CAD7DC]">
                  <span className="font-mono text-[10px]">F1-Score</span>
                  <span className="font-mono font-bold text-[#FFA2A9]">{item.f1}%</span>
                </div>
                <div className="w-full h-1 bg-[#163E5A] rounded-full overflow-hidden">
                  <div className="h-full bg-[#FFA2A9]" style={{ width: `${item.f1}%` }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
