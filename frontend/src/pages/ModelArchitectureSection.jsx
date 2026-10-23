// pages/ModelArchitectureSection.jsx
import React from 'react';
import { Check } from 'lucide-react';

export default function ModelArchitectureSection() {
  const specs = [
    { label: 'Backbone Network', value: 'EfficientNet-B0 (Compound Scaling α=1.2, β=1.1)' },
    { label: 'Input Resolution', value: '224 × 224 × 3 (RGB, ImageNet Normalized)' },
    { label: 'Total Parameters', value: '4,342,884 trainable parameters' },
    { label: 'Feature Extractor Depth', value: '7 Inverted Residual MBConv stages (16 blocks)' },
    { label: 'Squeeze-and-Excitation', value: 'SE Ratio 0.25 (Adaptive channel recalibration)' },
    { label: 'Latent Embedding Size', value: '1,280 feature dimensions (Global Avg Pool)' },
    { label: 'Classification Head', value: 'BatchNorm1d → Linear(256) → ReLU → Dropout(0.4) → Linear(128) → Linear(4)' },
    { label: 'Regularization', value: 'Dual Dropout (0.4 / 0.3) + Weight Decay (1e-4)' },
  ];

  return (
    <section id="architecture" className="py-20 sm:py-28 bg-[#F8FAF9] border-b border-[#DCE3E2]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        <div className="max-w-3xl space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono tracking-widest uppercase text-[#778995]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#003153]" />
            <span>03 / NEURAL NETWORK SPECIFICATION</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-sans font-normal tracking-tight text-[#003153]">
            EfficientNet-B0 <br />
            <span className="font-display italic text-[#426679]">Compound Scaled</span> Architecture.
          </h2>
          <p className="text-sm sm:text-base text-[#526D7C] leading-relaxed">
            Unlike oversized architectures like ResNet-152 or VGG-19 that suffer from over-parameterization on smaller medical datasets, EfficientNet-B0 balances depth, width, and spatial resolution uniformly using compound scaling coefficients.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Phase 1 */}
          <div className="bg-white border border-[#DCE3E2] p-6 space-y-4 rounded-none hover:border-[#003153] transition-colors">
            <div className="flex items-center justify-between pb-3 border-b border-[#EAEFEF]">
              <span className="font-mono text-xs font-bold text-[#003153]">PHASE 1 : ADAPTER CONVERGENCE</span>
              <span className="text-[9px] font-mono px-2 py-0.5 bg-[#EEF2F1] text-[#425969]">
                13 Epochs · lr = 1e-3
              </span>
            </div>
            <h3 className="text-base font-bold text-[#003153]">Frozen ImageNet Backbone</h3>
            <p className="text-xs text-[#526D7C] leading-relaxed">
              Backbone convolutional weights remain frozen to preserve foundational visual primitives (Gabor edges, geometric contours). Gradients backpropagate exclusively into the new 3-layer classification head until training loss stabilizes below 0.51.
            </p>
            <ul className="text-xs space-y-1.5 text-[#3E5866] pt-2 border-t border-[#EAEFEF]">
              <li className="flex items-center gap-2">
                <Check className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                <span>Prevents catastrophic forgetting of low-level visual features</span>
              </li>
              <li className="flex items-center gap-2">
                <Check className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                <span>Rapid convergence in under 15 minutes on RTX 4060 GPU</span>
              </li>
            </ul>
          </div>

          {/* Phase 2 */}
          <div className="bg-white border border-[#DCE3E2] p-6 space-y-4 rounded-none hover:border-[#D62839] transition-colors">
            <div className="flex items-center justify-between pb-3 border-b border-[#EAEFEF]">
              <span className="font-mono text-xs font-bold text-[#D62839]">PHASE 2 : DEEP FINE-TUNING</span>
              <span className="text-[9px] font-mono px-2 py-0.5 bg-[#FFEFEF] text-[#D62839] font-bold">
                10 Epochs · lr = 1e-5
              </span>
            </div>
            <h3 className="text-base font-bold text-[#003153]">Full Network Unfreezing</h3>
            <p className="text-xs text-[#526D7C] leading-relaxed">
              All 16 MBConv blocks are unpinned at a micro-learning rate (1e-5) with Cosine Annealing. The higher-level convolutional filters adapt specifically to MRI tissue contrast, gray-white matter gradients, and dural enhancement boundaries.
            </p>
            <ul className="text-xs space-y-1.5 text-[#3E5866] pt-2 border-t border-[#EAEFEF]">
              <li className="flex items-center gap-2">
                <Check className="w-3.5 h-3.5 text-[#D62839] shrink-0" />
                <span>Pushes test set accuracy from 90.4% up to 93.06%</span>
              </li>
              <li className="flex items-center gap-2">
                <Check className="w-3.5 h-3.5 text-[#D62839] shrink-0" />
                <span>Refines Grad-CAM saliency boundaries around subtle tumors</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Technical Specification Grid */}
        <div className="bg-white border border-[#DCE3E2] p-6 sm:p-8">
          <div className="pb-4 mb-6 border-b border-[#EAEFEF] flex justify-between items-center">
            <h3 className="text-lg font-bold text-[#003153]">Hardware & Algorithmic Parameters</h3>
            <span className="font-mono text-[9px] text-[#788E99]">PYTORCH 2.4 · TIMM 1.0.9</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {specs.map((spec) => (
              <div key={spec.label} className="p-3 bg-[#F8FAF9] border border-[#E3EAE9] flex flex-col justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#778995]">
                  {spec.label}
                </span>
                <span className="text-xs font-bold text-[#003153] mt-1 font-mono">
                  {spec.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
