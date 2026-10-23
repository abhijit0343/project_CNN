// pages/LiveExplorerSection.jsx
import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Zap, AlertCircle, RefreshCw } from 'lucide-react';
import { useScanAnalysis } from '../hooks/useScanAnalysis';
import { MODEL_STAGES } from '../data/model';
import SampleScansPicker from '../components/SampleScansPicker';
import CNNInteractiveFlow from '../components/CNNInteractiveFlow';
import ResultPanel from '../components/ResultPanel';

export default function LiveExplorerSection() {
  const {
    file,
    previewUrl,
    loading,
    error,
    result,
    selectFile,
    loadSample,
    analyze,
    clearScan,
  } = useScanAnalysis();

  const [isDragging, setIsDragging] = useState(false);
  const [currentStageIndex, setCurrentStageIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (loading) {
      setCurrentStageIndex(0);
      const interval = setInterval(() => {
        setCurrentStageIndex((prev) => (prev + 1) % MODEL_STAGES.length);
      }, 350);
      return () => clearInterval(interval);
    }
  }, [loading]);

  useEffect(() => {
    if (!result || !isPlaying) return;
    const interval = setInterval(() => {
      setCurrentStageIndex((prev) => (prev >= MODEL_STAGES.length - 1 ? 0 : prev + 1));
    }, 1100 / speed);
    return () => clearInterval(interval);
  }, [result, isPlaying, speed]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <section id="scan" className="py-20 sm:py-28 bg-[#EEF2F1] border-b border-[#DCE3E2]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        {/* Section Heading */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-2">
          <div className="space-y-3 max-w-2xl">
            <div className="flex items-center gap-2 text-xs font-mono tracking-widest uppercase text-[#778995]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#D62839]" />
              <span>02 / LIVE MODEL INFERENCE & VISUALIZATION</span>
            </div>
            <h2 className="text-4xl sm:text-5xl font-sans font-normal tracking-tight text-[#003153]">
              Upload an MRI. <br />
              <span className="font-display italic text-[#426679]">Trace</span> the activation flow.
            </h2>
            <p className="text-sm text-[#526D7C] leading-relaxed">
              Drop an axial, coronal, or sagittal brain MRI below. Watch the signal travel through the 7 compound-scaled MBConv stages of EfficientNet-B0 to produce a 4-class differential diagnosis.
            </p>
          </div>

          {/* Model Status Pill */}
          <div className="bg-white border border-[#DCE3E2] p-3.5 flex items-center gap-3.5 shadow-xs shrink-0">
            <div className="relative flex items-center justify-center">
              <span
                className={`w-3 h-3 rounded-full ${
                  result?.mode === 'live'
                    ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50'
                    : 'bg-[#D62839]'
                }`}
              />
              <span
                className={`absolute w-4 h-4 rounded-full animate-ping opacity-75 ${
                  result?.mode === 'live' ? 'bg-emerald-400' : 'bg-[#D62839]'
                }`}
              />
            </div>
            <div>
              <span className="font-mono text-[9px] uppercase tracking-wider text-[#778995] block">
                ENGINE STATE
              </span>
              <span className="font-mono text-xs font-bold text-[#003153] block">
                {result
                  ? result.mode === 'live'
                    ? 'LIVE PYTORCH MODEL'
                    : 'RESEARCH SIMULATION'
                  : 'AWAITING SCAN UPLOAD'}
              </span>
            </div>
          </div>
        </div>

        {/* 1-Click Sample Scans Bar */}
        <SampleScansPicker
          disabled={loading}
          onSelectSample={(url, name, expectedClass) => {
            loadSample(url, name, expectedClass);
          }}
        />

        {/* Main Interactive Grid: Upload Zone + Tensor Flow */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload Drop Card */}
          <div className="lg:col-span-5 bg-[#FBFCFA] border border-[#DCE3E2] p-6 shadow-sm">
            <div className="flex items-center justify-between pb-3 border-b border-[#E3EAE9]">
              <span className="font-mono text-[10px] tracking-widest uppercase text-[#778995] font-semibold">
                01 / SCAN INPUT ZONE
              </span>
              <span className="text-[9px] font-mono text-[#8C9FA9]">
                DICOM / JPEG / PNG
              </span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/jpg"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  selectFile(e.target.files[0]);
                }
              }}
            />

            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`relative mt-5 w-full aspect-square max-h-[300px] border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center p-6 text-center select-none overflow-hidden ${
                isDragging
                  ? 'border-[#D62839] bg-[#FFF8F8]'
                  : previewUrl
                  ? 'border-[#003153] bg-black'
                  : 'border-[#CBD8D9] bg-[#F4F7F5] hover:border-[#003153] hover:bg-[#EEF3F2]'
              }`}
            >
              {previewUrl ? (
                <>
                  <img
                    src={previewUrl}
                    alt="Uploaded MRI Preview"
                    className="w-full h-full object-contain pointer-events-none"
                  />
                  {loading && (
                    <div className="absolute inset-x-0 h-1 bg-[#D62839] shadow-[0_0_15px_#D62839] animate-scan pointer-events-none" />
                  )}
                  <span className="absolute top-2 left-2 bg-[#003153]/90 text-white font-mono text-[8px] px-2 py-0.5 tracking-wider border border-white/20">
                    LOADED SLICE
                  </span>
                </>
              ) : (
                <div className="space-y-3 pointer-events-none">
                  <div className="w-12 h-12 rounded-full border border-[#BACCD4] bg-white flex items-center justify-center mx-auto text-[#003153]">
                    <UploadCloud className="w-6 h-6 stroke-[1.5]" />
                  </div>
                  <div>
                    <span className="text-xs font-bold text-[#003153] block">
                      Drop MRI scan here or <u className="text-[#D62839]">browse</u>
                    </span>
                    <span className="text-[10px] text-[#718590] mt-1 block">
                      Supports axial, sagittal & coronal views (max 15 MB)
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-4 flex items-center justify-between text-xs font-mono text-[#526D7C]">
              {file ? (
                <div className="flex items-center gap-2 truncate max-w-[240px]">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
                  <span className="truncate text-[#003153] font-semibold">{file.name}</span>
                  <span className="text-[#8C9FA9] text-[9px]">
                    ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                  </span>
                </div>
              ) : (
                <span className="text-[10px] text-[#8C9FA9]">No file selected</span>
              )}

              {file && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearScan();
                  }}
                  className="text-[10px] text-[#D62839] hover:underline"
                >
                  Clear
                </button>
              )}
            </div>

            <button
              type="button"
              disabled={!file || loading}
              onClick={() => analyze()}
              className="mt-4 w-full py-3.5 bg-[#003153] hover:bg-[#1B4561] disabled:opacity-40 disabled:cursor-not-allowed text-white font-mono text-xs font-bold tracking-wider transition-all flex items-center justify-center gap-2 shadow-xs"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-[#D62839]" />
                  <span>EXTRACTING CONV TENSORS…</span>
                </>
              ) : (
                <>
                  <span>ANALYZE MRI SCAN</span>
                  <Zap className="w-3.5 h-3.5 text-[#D62839]" />
                </>
              )}
            </button>

            {error && (
              <div className="mt-3 p-3 bg-red-50 border border-red-200 text-red-800 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-[#D62839] shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Right Column: Large Interactive CNN Diagram */}
          <div className="lg:col-span-7">
            <CNNInteractiveFlow
              currentStageIndex={currentStageIndex}
              onSelectStage={(idx) => {
                setCurrentStageIndex(idx);
                setIsPlaying(false);
              }}
              isPlaying={isPlaying}
              onTogglePlay={() => setIsPlaying(!isPlaying)}
              speed={speed}
              onChangeSpeed={(s) => setSpeed(s)}
              scanResult={result}
              inputPreviewUrl={previewUrl}
              isProcessing={loading}
            />
          </div>
        </div>

        {/* Diagnostic Output & Saliency Analysis Panel */}
        {result && <ResultPanel result={result} />}
      </div>
    </section>
  );
}
