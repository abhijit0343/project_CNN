// pages/PerformanceSection.jsx
import React from 'react';
import PerformanceCharts from '../components/PerformanceCharts';
import { ShieldCheck, Database, Award } from 'lucide-react';

export default function PerformanceSection() {
  return (
    <section id="performance" className="py-20 sm:py-28 bg-[#001D33] text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        <div className="max-w-3xl space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono tracking-widest uppercase text-[#8EAAB8]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D62839]" />
            <span>04 / VALIDATION BENCHMARKS</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-sans font-normal tracking-tight text-white">
            Rigorous evaluation <br />
            <span className="font-display italic text-[#FFA2A9]">across 1,600</span> test scans.
          </h2>
          <p className="text-sm sm:text-base text-[#CAD7DC] leading-relaxed">
            Models evaluated on a strictly segregated test split of 1,600 MRI images (400 images per class) unseen during training or hyperparameter optimization.
          </p>
        </div>

        <PerformanceCharts />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
          <div className="p-5 bg-[#00253F] border border-[#163E5A] space-y-2">
            <Database className="w-5 h-5 text-[#4F92BC]" />
            <h4 className="text-sm font-bold text-white font-mono">Stratified Patient Split</h4>
            <p className="text-xs text-[#8EAAB8] leading-relaxed">
              Training and testing partitions maintain balanced demographic distributions across glioma, meningioma, pituitary adenoma, and healthy control scans.
            </p>
          </div>

          <div className="p-5 bg-[#00253F] border border-[#163E5A] space-y-2">
            <ShieldCheck className="w-5 h-5 text-[#D62839]" />
            <h4 className="text-sm font-bold text-white font-mono">100% Benign Specificity</h4>
            <p className="text-xs text-[#8EAAB8] leading-relaxed">
              Achieved 400 out of 400 correct classifications on non-tumor scans, eliminating false positive alarm rates on healthy patient MRIs.
            </p>
          </div>

          <div className="p-5 bg-[#00253F] border border-[#163E5A] space-y-2">
            <Award className="w-5 h-5 text-[#FFA2A9]" />
            <h4 className="text-sm font-bold text-white font-mono">92.96% Macro F1 Score</h4>
            <p className="text-xs text-[#8EAAB8] leading-relaxed">
              Exhibits uniform diagnostic discrimination without performance skew toward higher-frequency training orientations.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
