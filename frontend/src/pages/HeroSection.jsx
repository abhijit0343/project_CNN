// pages/HeroSection.jsx
import React from 'react';
import { ArrowUpRight, ArrowDown, Activity, CheckCircle2 } from 'lucide-react';
import BrainNetwork from '../components/BrainNetwork';

export default function HeroSection() {
  return (
    <section id="top" className="relative bg-[#003153] text-white pt-16 pb-20 sm:pt-24 sm:pb-28 overflow-hidden isolate">
      {/* Hairline Grid Texture */}
      <div
        className="absolute inset-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(240, 248, 252, 0.2) 1px, transparent 1px),
            linear-gradient(90deg, rgba(240, 248, 252, 0.2) 1px, transparent 1px)
          `,
          backgroundSize: '64px 64px',
        }}
      />

      {/* Atmospheric Vignette Mask */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#00253F] via-transparent to-transparent pointer-events-none opacity-80" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Headlines & Actions */}
          <div className="lg:col-span-6 space-y-6">
            {/* Eyebrow Pill */}
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/10 border border-white/15 text-[10px] font-mono tracking-widest uppercase text-[#CAD7DC]">
              <span className="w-2 h-2 rounded-full bg-[#D62839] animate-pulse" />
              <span>CLINICAL RESEARCH PLATFORM · PYTORCH</span>
            </div>

            {/* Main Headline */}
            <h1 className="text-5xl sm:text-6xl md:text-7xl font-sans font-normal tracking-tight text-white leading-[1.02]">
              Look closer. <br />
              <span className="font-display italic text-[#FFA2A9] font-normal">See what’s</span> within.
            </h1>

            {/* Subtitle */}
            <p className="text-sm sm:text-base text-[#CAD7DC] leading-relaxed max-w-lg font-sans">
              An explainable deep convolutional neural network for brain tumor detection on multi-sequence MRI scans. Dissect how high-level feature maps isolate glioma, meningioma, and pituitary lesions layer by layer.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <a
                href="#scan"
                className="inline-flex items-center gap-2.5 px-6 py-3.5 bg-[#D62839] hover:bg-[#B81E2E] text-white font-mono text-xs font-bold tracking-wider transition-all shadow-lg shadow-[#D62839]/20 hover:-translate-y-0.5"
              >
                UPLOAD MRI SCAN <ArrowUpRight className="w-4 h-4" />
              </a>

              <a
                href="#approach"
                className="inline-flex items-center gap-2 px-5 py-3.5 bg-white/5 hover:bg-white/10 text-[#CAD7DC] hover:text-white border border-white/15 font-mono text-xs font-semibold transition-all hover:-translate-y-0.5"
              >
                HOW IT WORKS <ArrowDown className="w-3.5 h-3.5 text-[#D62839]" />
              </a>
            </div>

            {/* Proof Metric Strip */}
            <div className="pt-6 border-t border-white/10 flex flex-wrap items-center gap-6 text-[10px] font-mono text-[#8EAAB8]">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#D62839]" />
                <span>7,023 Patient MRI Scans</span>
              </div>
              <span className="text-white/20">|</span>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#4F92BC]" />
                <span>93.06% Test Set Accuracy</span>
              </div>
              <span className="text-white/20">|</span>
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-[#D62839]" />
                <span>Real-Time Grad-CAM Saliency</span>
              </div>
            </div>
          </div>

          {/* Right Column: 60fps Neural Pathway Graph */}
          <div className="lg:col-span-6 relative">
            <div className="relative border border-white/10 bg-[#00263f]/60 backdrop-blur-sm p-4 overflow-hidden shadow-2xl">
              <BrainNetwork />

              {/* Float Overlay Badge 1 */}
              <div className="absolute top-4 left-4 bg-[#001D33]/90 border border-[#2B5470] p-2.5 text-[10px] font-mono backdrop-blur-md">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#D62839] animate-ping" />
                  <span className="font-bold text-white">SYNAPTIC CONVOLUTION</span>
                </div>
                <span className="text-[#8EAAB8] text-[9px] block mt-0.5">
                  EfficientNet-B0 · 7 Inverted Residuals
                </span>
              </div>

              {/* Float Overlay Badge 2 */}
              <div className="absolute bottom-4 right-4 bg-[#001D33]/90 border border-[#2B5470] p-2.5 text-[10px] font-mono backdrop-blur-md">
                <div className="text-right">
                  <span className="text-[9px] text-[#8EAAB8] block">RESOLUTION</span>
                  <span className="font-bold text-[#FFA2A9]">224 × 224 × 3 TENSOR</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
