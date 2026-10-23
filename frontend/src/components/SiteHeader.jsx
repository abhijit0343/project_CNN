// components/SiteHeader.jsx
import React, { useState, useEffect } from 'react';
import { History, Menu, X, ArrowUpRight } from 'lucide-react';

export default function SiteHeader({ onOpenHistory }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);

  const updateCount = () => {
    try {
      const items = JSON.parse(localStorage.getItem('neuroscan-history') || '[]');
      setHistoryCount(items.length);
    } catch {
      setHistoryCount(0);
    }
  };

  useEffect(() => {
    updateCount();
    window.addEventListener('neuroscan-history-updated', updateCount);
    return () => window.removeEventListener('neuroscan-history-updated', updateCount);
  }, []);

  return (
    <header className="sticky top-0 z-40 bg-[#F8FAF9]/90 backdrop-blur-md border-b border-[#003153]/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        {/* Brand Logo */}
        <a href="#top" className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded-none border-1.5 border-[#003153] flex items-center justify-center bg-white shadow-xs group-hover:border-[#D62839] transition-colors">
            <svg viewBox="0 0 24 24" className="w-5 h-5 stroke-[#003153] group-hover:stroke-[#D62839] transition-colors fill-none stroke-2">
              <circle cx="12" cy="12" r="3" fill="#D62839" stroke="none" />
              <path d="M12 2v3m0 14v3M2 12h3m14 0h3M4.9 4.9l2.1 2.1m10 10l2.1 2.1M4.9 19.1l2.1-2.1m10-10l2.1-2.1" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <span className="font-bold text-lg tracking-tight text-[#003153] flex items-center gap-1 font-sans">
              NeuroScan <span className="text-[#D62839] font-mono text-sm tracking-wider">AI</span>
            </span>
            <span className="text-[9px] font-mono text-[#6A818F] block tracking-widest uppercase -mt-1">
              Brain MRI Classifier
            </span>
          </div>
        </a>

        {/* Desktop Navigation Links */}
        <nav className="hidden lg:flex items-center gap-8 text-xs font-semibold text-[#425969]">
          <a href="#approach" className="hover:text-[#D62839] transition-colors">
            How It Works
          </a>
          <a href="#scan" className="hover:text-[#D62839] transition-colors">
            Live Model Explorer
          </a>
          <a href="#architecture" className="hover:text-[#D62839] transition-colors">
            Architecture
          </a>
          <a href="#performance" className="hover:text-[#D62839] transition-colors">
            Performance
          </a>
          <a href="#faq" className="hover:text-[#D62839] transition-colors">
            FAQ & Specs
          </a>
        </nav>

        {/* Right CTA Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenHistory}
            className="relative p-2 text-[#425969] hover:text-[#003153] hover:bg-[#EEF2F1] rounded-none border border-transparent hover:border-[#CBD8D9] transition-all"
            title="Open Scan History"
            aria-label="Open Scan History"
          >
            <History className="w-4 h-4" />
            {historyCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-[#D62839] text-white text-[9px] font-mono font-bold flex items-center justify-center">
                {historyCount}
              </span>
            )}
          </button>

          <a
            href="#scan"
            className="hidden sm:inline-flex items-center gap-2 px-4 py-2.5 text-xs font-bold font-mono tracking-wider text-white bg-[#D62839] hover:bg-[#B81E2E] transition-all shadow-xs"
          >
            TEST SCAN <ArrowUpRight className="w-3.5 h-3.5" />
          </a>

          {/* Mobile Menu Toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 text-[#003153] hover:bg-[#EEF2F1]"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-[#CBD8D9] bg-[#F8FAF9] p-4 space-y-3 animate-in slide-in-from-top-2 duration-200">
          <a
            href="#approach"
            onClick={() => setMobileMenuOpen(false)}
            className="block py-2 text-sm font-semibold text-[#003153] hover:text-[#D62839]"
          >
            How It Works
          </a>
          <a
            href="#scan"
            onClick={() => setMobileMenuOpen(false)}
            className="block py-2 text-sm font-semibold text-[#003153] hover:text-[#D62839]"
          >
            Live Model Explorer
          </a>
          <a
            href="#architecture"
            onClick={() => setMobileMenuOpen(false)}
            className="block py-2 text-sm font-semibold text-[#003153] hover:text-[#D62839]"
          >
            Architecture
          </a>
          <a
            href="#performance"
            onClick={() => setMobileMenuOpen(false)}
            className="block py-2 text-sm font-semibold text-[#003153] hover:text-[#D62839]"
          >
            Performance
          </a>
          <a
            href="#faq"
            onClick={() => setMobileMenuOpen(false)}
            className="block py-2 text-sm font-semibold text-[#003153] hover:text-[#D62839]"
          >
            FAQ & Specs
          </a>
        </div>
      )}
    </header>
  );
}
