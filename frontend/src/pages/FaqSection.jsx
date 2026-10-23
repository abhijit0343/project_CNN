// pages/FaqSection.jsx
import React, { useState } from 'react';
import { ChevronDown, HelpCircle } from 'lucide-react';

export default function FaqSection() {
  const [openIndex, setOpenIndex] = useState(0);

  const faqs = [
    {
      category: 'Diagnostic Scope',
      question: 'Which brain tumor categories does the network identify?',
      answer:
        'The model classifies four categories: Glioma (high/low grade astrocytomas, oligodendrogliomas), Meningioma (typically extra-axial dural-based tumors), Pituitary Adenoma (sellar/suprasellar mass in the pituitary fossa), and Healthy / No Tumor (unremarkable brain anatomy). These represent over 80% of primary intracranial neoplasms in adult neuro-oncology.',
    },
    {
      category: 'Interpretability',
      question: 'How does Grad-CAM prevent the model from learning scan artifacts?',
      answer:
        'Standard deep learning models frequently overfit to shortcut features (e.g. scanner watermarks, skull thickness, head coils). By computing Grad-CAM saliency on the final convolutional stage (MBConv Stage 7), we visually inspect the gradient weights contributing to the prediction. If the heatmap highlights text margins or cranial bone rather than the intracranial parenchyma, the scan is invalidated.',
    },
    {
      category: 'Architecture',
      question: 'Why choose EfficientNet-B0 over Vision Transformers (ViT) or ResNet-50?',
      answer:
        'Vision Transformers require tens of millions of training samples to learn spatial induction biases and are prone to overfitting on localized medical datasets. EfficientNet-B0 achieves 93.06% test accuracy with only 4.3 million parameters by utilizing compound depth-width-resolution scaling and depthwise separable inverted residuals (MBConv), enabling real-time browser inference without heavy GPU server overhead.',
    },
    {
      category: 'Privacy & Security',
      question: 'How is patient medical privacy handled during upload?',
      answer:
        'All image transfers are processed either locally via the local Express/Python service or simulated entirely in your browser sandbox. No images are permanently stored on public cloud servers, and no patient metadata (DICOM tags, patient names, dates of birth) is retained.',
    },
    {
      category: 'Clinical Limitations',
      question: 'What are the technical limitations of 2D slice inference versus full 3D MRI?',
      answer:
        'Current classification operates on individual 2D image slices (axial, coronal, sagittal). In clinical practice, radiologists evaluate multi-parametric 3D volumes (T1, T1 post-contrast, T2, FLAIR, DWI/ADC) and perfusion sequences. A single 2D slice may miss an infiltrative tumor margin located in adjacent cranial planes.',
    },
    {
      category: 'Regulatory',
      question: 'Can this software be used for real patient treatment or diagnosis?',
      answer:
        'No. NeuroScan AI is an educational and computer science research demonstration. It is NOT cleared by the US FDA, CE-marked, or approved by any medical regulatory authority. All clinical imaging decisions must be made by qualified medical professionals using certified diagnostic workstations.',
    },
  ];

  return (
    <section id="faq" className="py-20 sm:py-28 bg-[#F8FAF9] border-b border-[#DCE3E2]">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        <div className="max-w-3xl space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono tracking-widest uppercase text-[#778995]">
            <HelpCircle className="w-3.5 h-3.5 text-[#003153]" />
            <span>05 / TECHNICAL & CLINICAL INQUIRIES</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-sans font-normal tracking-tight text-[#003153]">
            Frequently asked <br />
            <span className="font-display italic text-[#426679]">methodological</span> questions.
          </h2>
          <p className="text-sm sm:text-base text-[#526D7C] leading-relaxed">
            Everything you need to know about model validation, interpretability saliency, hardware execution, and investigational boundaries.
          </p>
        </div>

        <div className="border-t border-[#DCE3E2] divide-y divide-[#DCE3E2]">
          {faqs.map((faq, index) => {
            const isOpen = openIndex === index;
            return (
              <div key={faq.question} className="py-5">
                <button
                  type="button"
                  onClick={() => setOpenIndex(isOpen ? null : index)}
                  className="w-full text-left flex items-start justify-between gap-4 group"
                  aria-expanded={isOpen}
                >
                  <div className="space-y-1">
                    <span className="text-[9px] font-mono uppercase tracking-wider text-[#D62839] font-bold block">
                      {faq.category}
                    </span>
                    <h3 className="text-base sm:text-lg font-bold text-[#003153] group-hover:text-[#D62839] transition-colors">
                      {faq.question}
                    </h3>
                  </div>
                  <span className={`p-1.5 rounded-full border border-[#CBD8D9] text-[#003153] shrink-0 transition-transform duration-300 ${isOpen ? 'rotate-180 bg-[#003153] text-white' : 'group-hover:border-[#003153]'}`}>
                    <ChevronDown className="w-4 h-4" />
                  </span>
                </button>

                {isOpen && (
                  <div className="pt-3 pr-8 text-xs sm:text-sm text-[#526D7C] leading-relaxed animate-in fade-in duration-200">
                    {faq.answer}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
