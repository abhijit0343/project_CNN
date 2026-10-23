// components/SiteFooter.jsx
import React from 'react';
import { ShieldAlert, Terminal, GitBranch, ArrowUp } from 'lucide-react';

export default function SiteFooter() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="bg-[#001F35] text-[#CAD7DC] border-t border-[#163E5A] pt-12 pb-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10 border-b border-[#163E5A]">
          {/* Brand Col */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2.5">
              <div className="w-6 h-6 border border-white/20 flex items-center justify-center bg-white/5">
                <span className="w-2 h-2 rounded-full bg-[#D62839]" />
              </div>
              <span className="font-bold text-white text-base tracking-tight font-sans">
                NeuroScan <span className="text-[#D62839] font-mono text-xs">AI</span>
              </span>
            </div>
            <p className="text-xs text-[#8EAAB8] leading-relaxed max-w-sm">
              An open clinical-grade research interface exploring convolutional neural network feature representations for brain tumor detection on multi-sequence MRI scans.
            </p>
            <div className="flex items-center gap-3 pt-2 text-[10px] font-mono text-[#718F9F]">
              <span className="flex items-center gap-1">
                <Terminal className="w-3 h-3 text-[#D62839]" /> PyTorch 2.4 / CUDA 12.4
              </span>
              <span>·</span>
              <span className="flex items-center gap-1">
                <GitBranch className="w-3 h-3 text-[#4F92BC]" /> EfficientNet-B0
              </span>
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-2.5 text-xs font-mono">
            <span className="text-[10px] uppercase tracking-wider text-white font-bold block mb-3">
              NAVIGATION
            </span>
            <a href="#approach" className="block text-[#8EAAB8] hover:text-white transition-colors">
              How It Works
            </a>
            <a href="#scan" className="block text-[#8EAAB8] hover:text-white transition-colors">
              Live Model Explorer
            </a>
            <a href="#architecture" className="block text-[#8EAAB8] hover:text-white transition-colors">
              CNN Architecture
            </a>
            <a href="#performance" className="block text-[#8EAAB8] hover:text-white transition-colors">
              Test Set Evaluation
            </a>
            <a href="#faq" className="block text-[#8EAAB8] hover:text-white transition-colors">
              Medical FAQ & Limitations
            </a>
          </div>

          {/* References & Dataset */}
          <div className="space-y-2.5 text-xs font-mono">
            <span className="text-[10px] uppercase tracking-wider text-white font-bold block mb-3">
              SCIENTIFIC CITATIONS
            </span>
            <p className="text-[10px] text-[#8EAAB8] leading-snug">
              • Tan & Le (2019): <em>EfficientNet: Rethinking Model Scaling for CNNs</em> (ICML).
            </p>
            <p className="text-[10px] text-[#8EAAB8] leading-snug">
              • Selvaraju et al. (2017): <em>Grad-CAM: Visual Explanations from Deep Networks</em> (ICCV).
            </p>
            <p className="text-[10px] text-[#8EAAB8] leading-snug">
              • Dataset: 7,023 Brain Tumor MRI Images (Figshare / SARTAJ / Br35H).
            </p>
          </div>
        </div>

        {/* Disclaimer & Bottom Bar */}
        <div className="pt-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-[10px] text-[#718F9F]">
          <div className="flex items-start gap-2 max-w-3xl">
            <ShieldAlert className="w-4 h-4 text-[#D62839] shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong>Clinical Notice:</strong> NeuroScan AI is an investigational software platform strictly intended for computer science, neuroimaging research, and educational demonstration. It is not approved by medical regulatory authorities and must not be utilized for primary patient screening or clinical decisions.
            </p>
          </div>

          <button
            onClick={scrollToTop}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-[#1B4561] hover:border-white text-white font-mono text-[9px] transition-colors shrink-0"
          >
            TOP <ArrowUp className="w-3 h-3" />
          </button>
        </div>
      </div>
    </footer>
  );
}
