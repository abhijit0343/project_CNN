// App.jsx
import React, { useState } from 'react';
import SiteHeader from './components/SiteHeader';
import SiteFooter from './components/SiteFooter';
import HeroSection from './pages/HeroSection';
import HowItWorksSection from './pages/HowItWorksSection';
import LiveExplorerSection from './pages/LiveExplorerSection';
import ModelArchitectureSection from './pages/ModelArchitectureSection';
import PerformanceSection from './pages/PerformanceSection';
import FaqSection from './pages/FaqSection';
import ScanHistoryDrawer from './components/ScanHistoryDrawer';

export default function App() {
  const [historyOpen, setHistoryOpen] = useState(false);
  const [restoredScan, setRestoredScan] = useState(null);

  const handleSelectHistoryScan = (scan) => {
    setRestoredScan(scan);
    const explorerEl = document.getElementById('scan');
    if (explorerEl) {
      explorerEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAF9] text-[#173247] selection:bg-[#D62839] selection:text-white font-sans antialiased overflow-x-hidden">
      {/* Top Global Navigation Bar */}
      <SiteHeader onOpenHistory={() => setHistoryOpen(true)} />

      {/* Main Flagship Content Sections */}
      <main>
        {/* 01. Hero with 60fps Neural Graph Canvas */}
        <HeroSection />

        {/* 02. How the Pipeline Works */}
        <HowItWorksSection />

        {/* 03. Live MRI Upload & Layer-by-Layer Activation Flow */}
        <LiveExplorerSection />

        {/* 04. EfficientNet-B0 Real Model Architecture & Parameters */}
        <ModelArchitectureSection />

        {/* 05. Rigorous Evaluation Benchmarks on 1,600 Test Scans */}
        <PerformanceSection />

        {/* 06. Technical & Medical FAQ */}
        <FaqSection />
      </main>

      {/* Bottom Global Footer with Disclaimers and Citations */}
      <SiteFooter />

      {/* Slide-out Scan History Drawer */}
      <ScanHistoryDrawer
        isOpen={historyOpen}
        onClose={() => setHistoryOpen(false)}
        onSelectScan={handleSelectHistoryScan}
      />
    </div>
  );
}
