// pages/HowItWorksSection.jsx
import React from 'react';
import { Layers, Scan, Cpu, Eye } from 'lucide-react';

export default function HowItWorksSection() {
  const steps = [
    {
      number: '01',
      title: 'MRI Tensor Normalization',
      tagline: 'Standardized Preprocessing Pipeline',
      description: 'Incoming brain scans (axial, coronal, sagittal) are decoded, rescaled to 224 × 224 pixels, and channel-normalized against ImageNet mean and variance to ensure invariant intensity distribution.',
      icon: Scan,
      detail: '224 × 224 × 3 Tensor · Linear Interpolation',
    },
    {
      number: '02',
      title: 'Compound MBConv Filtering',
      tagline: '7 Inverted Residual Stages',
      description: 'Depthwise separable convolutions isolate textural features (fluid, necrosis, dural tissue) across compound scaling factors without parameter bloat, passing spatial tensors from 112² down to 7².',
      icon: Layers,
      detail: '16 to 320 Channels · Squeeze-and-Excitation (SE)',
    },
    {
      number: '03',
      title: 'Dense Classification Head',
      tagline: 'Regularized Softmax Scoring',
      description: 'Adaptive average pooling condenses 7 × 7 × 1280 spatial maps into a compact 1280-d latent vector. A 3-layer MLP with BatchNorm and dual dropout (0.4 / 0.3) computes calibrated posterior probabilities for all 4 classes.',
      icon: Cpu,
      detail: '1280 → 256 → 128 → 4 Classes',
    },
    {
      number: '04',
      title: 'Explainable Grad-CAM',
      tagline: 'Visual Gradient Localization',
      description: 'To confirm anatomical validity, gradients of the winning class are backpropagated into the final convolutional stage. A Prussian-to-Red saliency overlay reveals the precise tissue region that triggered the diagnosis.',
      icon: Eye,
      detail: 'Stage 7 Activation Backprop · 2D Saliency Mask',
    },
  ];

  return (
    <section id="approach" className="py-20 sm:py-28 bg-[#F8FAF9] border-b border-[#DCE3E2]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl space-y-3 mb-16">
          <div className="flex items-center gap-2 text-xs font-mono tracking-widest uppercase text-[#778995]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#003153]" />
            <span>01 / METHODOLOGICAL PIPELINE</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-sans font-normal tracking-tight text-[#003153]">
            How the network <br />
            <span className="font-display italic text-[#426679]">interprets</span> a brain scan.
          </h2>
          <p className="text-sm sm:text-base text-[#526D7C] leading-relaxed">
            From raw radiographic pixels to localized Grad-CAM heatmaps, each MRI scan passes through a rigorous four-phase pipeline designed for transparency and clinical interpretability.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((step) => {
            const IconComponent = step.icon;
            return (
              <div
                key={step.number}
                className="bg-white border border-[#DCE3E2] p-6 flex flex-col justify-between hover:border-[#003153] hover:shadow-md transition-all duration-300 rounded-none group"
              >
                <div>
                  <div className="flex items-center justify-between pb-4 border-b border-[#EAEFEF]">
                    <span className="font-mono text-sm font-bold text-[#D62839]">
                      {step.number}
                    </span>
                    <IconComponent className="w-5 h-5 text-[#003153] group-hover:text-[#D62839] transition-colors" />
                  </div>

                  <h3 className="text-base font-bold text-[#003153] mt-5 mb-1 group-hover:text-[#D62839] transition-colors">
                    {step.title}
                  </h3>
                  <span className="text-[10px] font-mono text-[#778995] uppercase tracking-wider block mb-3">
                    {step.tagline}
                  </span>

                  <p className="text-xs text-[#526D7C] leading-relaxed">
                    {step.description}
                  </p>
                </div>

                <div className="pt-4 mt-6 border-t border-[#EAEFEF] text-[9px] font-mono text-[#8296A1]">
                  {step.detail}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
