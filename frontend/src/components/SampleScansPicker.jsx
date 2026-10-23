// components/SampleScansPicker.jsx
import React from 'react';
import { SAMPLE_SCANS } from '../data/samples';
import { Sparkles, ArrowRight } from 'lucide-react';

export default function SampleScansPicker({ onSelectSample, disabled = false }) {
  return (
    <div className="bg-[#FBFCFA] border border-[#DCE3E2] p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 mb-4 border-b border-[#E3EAE9]">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-[#D62839]" />
          <span className="font-mono text-[10px] tracking-widest uppercase text-[#778995] font-semibold">
            INSTANT SAMPLE EXPLORATION
          </span>
        </div>
        <span className="text-[9px] font-mono text-[#8C9FA9]">
          1-CLICK EVALUATION (REAL TEST SCANS)
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {SAMPLE_SCANS.map((sample) => (
          <button
            key={sample.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelectSample(sample.url, sample.name, sample.expectedClass)}
            className="group text-left p-2.5 bg-white border border-[#DCE3E2] hover:border-[#003153] hover:shadow-sm transition-all rounded-none flex flex-col justify-between disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div>
              <div className="w-full aspect-square bg-[#0c1f2e] overflow-hidden mb-2 relative border border-[#BACCD4]">
                <img
                  src={sample.url}
                  alt={sample.label}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <span className="absolute bottom-1 right-1 text-[7px] font-mono bg-[#003153]/85 text-white px-1 py-0.2">
                  {sample.sliceOrientation}
                </span>
              </div>

              <span className="text-[10px] font-bold text-[#003153] block truncate group-hover:text-[#D62839] transition-colors">
                {sample.label}
              </span>
              <p className="text-[8px] text-[#718590] line-clamp-2 mt-0.5 leading-snug">
                {sample.description}
              </p>
            </div>

            <div className="mt-2.5 pt-1.5 border-t border-[#EDF2F2] flex items-center justify-between text-[8px] font-mono text-[#4F6C7D]">
              <span>Test slice</span>
              <span className="text-[#D62839] group-hover:translate-x-0.5 transition-transform flex items-center gap-0.5">
                Load <ArrowRight className="w-2.5 h-2.5" />
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
