// components/CNNInteractiveFlow.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight, Layers, Cpu, Zap } from 'lucide-react';
import { MODEL_STAGES } from '../data/model';

export default function CNNInteractiveFlow({
  currentStageIndex,
  onSelectStage,
  isPlaying,
  onTogglePlay,
  speed,
  onChangeSpeed,
  scanResult,
  inputPreviewUrl,
  isProcessing = false,
}) {
  const [hoveredStage, setHoveredStage] = useState(null);
  const scrollContainerRef = useRef(null);

  useEffect(() => {
    if (scrollContainerRef.current && currentStageIndex >= 0) {
      const activeElement = scrollContainerRef.current.children[0]?.children[currentStageIndex];
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
    }
  }, [currentStageIndex]);

  const activeStage = MODEL_STAGES[Math.max(0, currentStageIndex)] || MODEL_STAGES[0];

  return (
    <div className="bg-[#FBFCFA] border border-[#DCE3E2] rounded-none p-5 sm:p-6 shadow-sm flex flex-col justify-between">
      {/* Top Header Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[#E3EAE9]">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] tracking-widest uppercase text-[#778995] font-semibold">
            02 / TENSOR FLOW ARCHITECTURE
          </span>
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 text-[9px] font-mono uppercase tracking-wider rounded-sm bg-[#003153]/5 text-[#003153] border border-[#003153]/10">
            <Cpu className="w-2.5 h-2.5 text-[#003153]" /> EfficientNet-B0 · PyTorch
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[#778995]">
            STAGE {currentStageIndex >= 0 ? `${currentStageIndex + 1} / ${MODEL_STAGES.length}` : 'STANDBY'}
          </span>
          {isProcessing && (
            <span className="flex items-center gap-1 text-[9px] font-mono text-[#D62839] animate-pulse">
              <Zap className="w-3 h-3 fill-[#D62839]" /> INFERENCE RUNNING
            </span>
          )}
        </div>
      </div>

      {/* Interactive Horizontal Scrollable Pipeline Track */}
      <div
        ref={scrollContainerRef}
        className="relative my-6 overflow-x-auto pb-4 pt-2 scrollbar-thin scrollbar-thumb-[#CBD5D7] scrollbar-track-transparent select-none"
      >
        <div className="flex items-center gap-2 sm:gap-3 min-w-[880px] px-2">
          {MODEL_STAGES.map((stage, idx) => {
            const isCurrent = currentStageIndex === idx;
            const isVisited = currentStageIndex > idx;

            return (
              <React.Fragment key={stage.id}>
                {/* Stage Node */}
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectStage(idx)}
                  onMouseEnter={() => setHoveredStage(stage)}
                  onMouseLeave={() => setHoveredStage(null)}
                  className={`group relative flex flex-col items-center p-2.5 rounded-sm transition-all duration-300 cursor-pointer min-w-[76px] ${
                    isCurrent
                      ? 'bg-[#003153] text-white shadow-md shadow-[#003153]/20 ring-1 ring-[#D62839]'
                      : isVisited
                      ? 'bg-[#EBF1F4] text-[#003153] border border-[#CBDCDF]'
                      : 'bg-[#F4F7F5] text-[#3A5666] border border-[#DCE3E2] hover:border-[#003153]/40'
                  }`}
                  aria-label={`${stage.title} - ${stage.kind}`}
                >
                  {/* Step Index Badge */}
                  <span
                    className={`font-mono text-[8px] tracking-wider mb-1.5 ${
                      isCurrent ? 'text-[#FFA2A9]' : isVisited ? 'text-[#D62839]' : 'text-[#87969E]'
                    }`}
                  >
                    {String(idx + 1).padStart(2, '0')}
                  </span>

                  {/* Visual 3D Stack / Miniature Feature Map */}
                  <div className="relative w-10 h-10 mb-2 flex items-center justify-center">
                    {stage.filterType === 'input' ? (
                      <div className="w-8 h-8 rounded-sm overflow-hidden border border-current bg-black/40 relative">
                        {inputPreviewUrl ? (
                          <img src={inputPreviewUrl} alt="MRI Input" className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full bg-[#173247] flex items-center justify-center text-[7px] font-mono text-[#CAD7DC]">
                            MRI
                          </div>
                        )}
                        {isCurrent && (
                          <div className="absolute inset-0 bg-[#D62839]/30 animate-pulse border border-[#D62839]" />
                        )}
                      </div>
                    ) : stage.filterType === 'output' ? (
                      <div className="flex flex-col gap-1 items-center justify-center w-8 h-8">
                        {[0, 1, 2, 3].map((n) => (
                          <div
                            key={n}
                            className={`w-5 h-1 rounded-full transition-colors ${
                              isCurrent
                                ? 'bg-[#D62839] shadow-sm shadow-[#D62839]'
                                : isVisited
                                ? 'bg-[#003153]'
                                : 'bg-[#98AAB3]'
                            }`}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="relative w-7 h-7">
                        <div
                          className={`absolute inset-0 rounded-[2px] transition-transform duration-300 ${
                            isCurrent
                              ? 'bg-[#D62839]/20 border border-[#D62839] translate-x-1.5 translate-y-1.5'
                              : 'bg-[#C2D4DC] border border-[#9CB6C2] translate-x-1.5 translate-y-1.5'
                          }`}
                        />
                        <div
                          className={`absolute inset-0 rounded-[2px] transition-transform duration-300 ${
                            isCurrent
                              ? 'bg-[#D62839]/40 border border-[#D62839] translate-x-0.5 translate-y-0.5'
                              : 'bg-[#D7E4E8] border border-[#AFC5CE] translate-x-0.5 translate-y-0.5'
                          }`}
                        />
                        <div
                          className={`absolute inset-0 rounded-[2px] transition-colors duration-300 flex items-center justify-center ${
                            isCurrent
                              ? 'bg-[#D62839] text-white shadow-sm shadow-[#D62839]/50'
                              : isVisited
                              ? 'bg-[#003153] text-white'
                              : 'bg-[#E3ECED] text-[#2C4E61] border border-[#AFC5CE]'
                          }`}
                        >
                          <Layers className="w-3.5 h-3.5" />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Stage Title */}
                  <span className="text-[9px] font-bold tracking-tight text-center truncate w-full">
                    {stage.title}
                  </span>

                  {/* Tensor Spatial Shape Tag */}
                  <span
                    className={`font-mono text-[8px] mt-0.5 px-1 py-0.2 rounded-sm ${
                      isCurrent
                        ? 'bg-white/20 text-white'
                        : isVisited
                        ? 'bg-[#003153]/10 text-[#003153]'
                        : 'text-[#7D8F98]'
                    }`}
                  >
                    {stage.tag}
                  </span>
                </div>

                {/* Connecting Signal Arrow with Traveling Pulse */}
                {idx < MODEL_STAGES.length - 1 && (
                  <div className="relative flex items-center justify-center w-4 text-[#8BA0AB]">
                    <span className="text-xs select-none">→</span>
                    {isCurrent && (
                      <span className="absolute w-2 h-2 rounded-full bg-[#D62839] shadow-sm shadow-[#D62839] animate-ping" />
                    )}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Explanatory Stage Focus Strip */}
      <div className="bg-[#F2F6F5] border border-[#DEE6E5] p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-start gap-3 flex-1">
          <span className="font-mono text-xs font-bold text-[#D62839] mt-0.5">
            {String(activeStage ? MODEL_STAGES.indexOf(activeStage) + 1 : 1).padStart(2, '0')}
          </span>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="text-xs font-bold text-[#003153] uppercase tracking-wide">
                {activeStage.title}
              </h4>
              <span className="text-[10px] text-[#5A7482] font-mono">
                {activeStage.kind} · {activeStage.outputShape}
              </span>
              <span className="text-[9px] px-1.5 py-0.2 font-mono bg-white border border-[#CBDCDF] text-[#2B4E62]">
                RF: {activeStage.receptiveField}
              </span>
            </div>
            <p className="text-[11px] text-[#556D7A] leading-relaxed mt-1 max-w-2xl">
              {activeStage.explanation}
            </p>
          </div>
        </div>

        {/* Playback & Step Controls */}
        <div className="flex items-center gap-2 self-end md:self-center shrink-0">
          <div className="flex items-center border border-[#CBD8D9] bg-white rounded-none overflow-hidden">
            <button
              onClick={() => onSelectStage(Math.max(0, currentStageIndex - 1))}
              disabled={currentStageIndex <= 0}
              className="p-1.5 text-[#003153] hover:bg-[#F2F6F5] disabled:opacity-30 disabled:hover:bg-white transition-colors"
              title="Previous Layer"
              aria-label="Previous Layer"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={onTogglePlay}
              className="p-1.5 text-[#003153] hover:bg-[#F2F6F5] border-x border-[#CBD8D9] transition-colors"
              title={isPlaying ? 'Pause Animation' : 'Play Flow Animation'}
              aria-label={isPlaying ? 'Pause Animation' : 'Play Flow Animation'}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            </button>
            <button
              onClick={() => onSelectStage(Math.min(MODEL_STAGES.length - 1, currentStageIndex + 1))}
              disabled={currentStageIndex >= MODEL_STAGES.length - 1}
              className="p-1.5 text-[#003153] hover:bg-[#F2F6F5] disabled:opacity-30 disabled:hover:bg-white transition-colors"
              title="Next Layer"
              aria-label="Next Layer"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onSelectStage(0)}
              className="p-1.5 text-[#003153] hover:bg-[#F2F6F5] border-l border-[#CBD8D9] transition-colors"
              title="Restart Flow"
              aria-label="Restart Flow"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Speed Selector */}
          <div className="flex items-center gap-1 text-[10px] font-mono border border-[#CBD8D9] bg-white px-2 py-1">
            <span className="text-[#87969E]">Speed:</span>
            {[0.5, 1, 2].map((s) => (
              <button
                key={s}
                onClick={() => onChangeSpeed(s)}
                className={`px-1 rounded-sm transition-colors ${
                  speed === s ? 'bg-[#003153] text-white font-bold' : 'text-[#415C6D] hover:text-[#003153]'
                }`}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Layer Architecture Legend */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 mt-3 border-t border-[#E8EFF0] text-[9px] text-[#718590]">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-[1px] bg-[#2C536E]" /> Image Input
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-[1px] bg-[#AFC5CE]" /> Stem 3×3 & Inverted Residuals (MBConv)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#D62839]" /> Classification Logits (4 Cls)
          </span>
        </div>
        <div className="font-mono text-[8px] text-[#8EA1AB]">
          Hover or click any block to isolate intermediate spatial transformations
        </div>
      </div>
    </div>
  );
}
