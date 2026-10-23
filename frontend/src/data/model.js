// data/model.js
// Real model architecture, training history, evaluation metrics, and stage definitions
// for the NeuroScan CNN (EfficientNet-B0 + Custom Classification Head).

export const CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary'];

export const CLASS_METADATA = {
  glioma: {
    name: 'Glioma',
    description: 'Arising from glial cells (astrocytes, oligodendrocytes). Often infiltrative along white matter tracts.',
    clinicalNote: 'Requires prompt neuro-oncological evaluation, contrast-enhanced volumetric MRI, and surgical consultation.',
    riskLevel: 'high',
  },
  meningioma: {
    name: 'Meningioma',
    description: 'Arising from the meninges surrounding the brain and spinal cord. Typically extra-axial and dural-based.',
    clinicalNote: 'Often slow-growing and well-circumscribed. Monitored via serial imaging or evaluated for stereotactic resection.',
    riskLevel: 'moderate',
  },
  notumor: {
    name: 'Healthy / No Tumor',
    description: 'No focal mass effect, abnormal contrast enhancement, or architectural distortion identified.',
    clinicalNote: 'Ventricular symmetry and parenchymal signal intensity remain within normal radiographic limits.',
    riskLevel: 'clear',
  },
  pituitary: {
    name: 'Pituitary Adenoma',
    description: 'Sellar or suprasellar mass originating in the pituitary fossa. May compress the optic chiasm.',
    clinicalNote: 'Endocrine panel and formal visual field testing indicated. Transsphenoidal approach common when symptomatic.',
    riskLevel: 'moderate',
  },
};

export const MODEL_STAGES = [
  {
    id: 'input',
    title: 'Input Layer',
    kind: 'Normalized MRI Slice',
    detail: 'Standardized to 224 × 224 pixels across 3 channels (RGB)',
    channels: 3,
    outputShape: '224 × 224 × 3',
    tag: '224²',
    receptiveField: '1 × 1',
    explanation: 'A 2D axial MRI slice is loaded and transformed with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).',
    filterType: 'input',
  },
  {
    id: 'stem',
    title: 'Stem Conv 3×3',
    kind: 'Initial Spatial Filter',
    detail: '32 filters with stride 2 and padding 1, followed by BatchNorm + SiLU',
    channels: 32,
    outputShape: '112 × 112 × 32',
    tag: '112²',
    receptiveField: '3 × 3',
    explanation: 'Captures primitive low-level gradients, tissue density edges, and contrast boundaries across ventricular borders.',
    filterType: 'conv',
  },
  {
    id: 'mbconv1',
    title: 'MBConv Stage 1',
    kind: 'Inverted Residual (Expansion 1)',
    detail: 'Depthwise 3×3 conv with Squeeze-and-Excitation (SE) ratio 0.25',
    channels: 16,
    outputShape: '112 × 112 × 16',
    tag: '112²',
    receptiveField: '7 × 7',
    explanation: 'Isolates fine textural striations and gray-white matter junction signatures without excessive parameter growth.',
    filterType: 'residual',
  },
  {
    id: 'mbconv2',
    title: 'MBConv Stage 2',
    kind: 'Downsampling Block × 2',
    detail: 'Expansion 6, stride 2 downsampling to 24 feature channels',
    channels: 24,
    outputShape: '56 × 56 × 24',
    tag: '56²',
    receptiveField: '15 × 15',
    explanation: 'Consolidates local patches into hemispheric symmetry cues, detecting shifts in midline ventricles.',
    filterType: 'residual',
  },
  {
    id: 'mbconv3',
    title: 'MBConv Stage 3',
    kind: 'Residual Block × 2',
    detail: 'Kernel size 5×5, expansion 6, SE channel gating into 40 maps',
    channels: 40,
    outputShape: '28 × 28 × 40',
    tag: '28²',
    receptiveField: '31 × 31',
    explanation: 'Expands receptive field to encompass regional lobe structures (temporal, frontal, parietal) and dural attachments.',
    filterType: 'residual',
  },
  {
    id: 'mbconv4',
    title: 'MBConv Stage 4',
    kind: 'Stride-2 Transition × 3',
    detail: 'Expansion 6, depthwise 3×3 with 80 channels',
    channels: 80,
    outputShape: '14 × 14 × 80',
    tag: '14²',
    receptiveField: '63 × 63',
    explanation: 'Detects hyperintense mass regions, vasogenic edema margins, and circumscribed hypointense necrotic cores.',
    filterType: 'residual',
  },
  {
    id: 'mbconv5',
    title: 'MBConv Stage 5',
    kind: 'Deep Spatial Mapping × 3',
    detail: 'Kernel size 5×5, expansion 6, 112 feature channels',
    channels: 112,
    outputShape: '14 × 14 × 112',
    tag: '14²',
    receptiveField: '127 × 127',
    explanation: 'Focuses receptive attention onto anatomical compartments: the sellar region (pituitary) and convexity dura (meningioma).',
    filterType: 'residual',
  },
  {
    id: 'mbconv6',
    title: 'MBConv Stage 6',
    kind: 'Deep Semantic Block × 4',
    detail: 'Kernel 5×5, stride 2, expanding to 192 channels',
    channels: 192,
    outputShape: '7 × 7 × 192',
    tag: '7²',
    receptiveField: '255 × 255',
    explanation: 'Abstracts whole-slice pathological morphology into rich semantic feature vectors invariant to slight patient tilt.',
    filterType: 'residual',
  },
  {
    id: 'mbconv7',
    title: 'MBConv Stage 7',
    kind: 'Final Feature Tensor × 1',
    detail: '320 channels, point-wise 1×1 expansion to 1280 channels',
    channels: 1280,
    outputShape: '7 × 7 × 1280',
    tag: '7²',
    receptiveField: '511 × 511',
    explanation: 'Target layer for Grad-CAM activations: gradients backpropagated from the predicted class generate the heatmap.',
    filterType: 'conv',
  },
  {
    id: 'head',
    title: 'Classification Head',
    kind: 'GAP + BatchNorm + Dense MLP',
    detail: 'Global Avg Pool → 1280 → Linear(256) → ReLU → Linear(128) → Linear(4)',
    channels: 4,
    outputShape: '1 × 4 Logits',
    tag: '4 Cls',
    receptiveField: 'Global',
    explanation: 'Global average pooling aggregates spatial maps into a 1280-d vector. Fully connected layers with dropout calculate class softmax scores.',
    filterType: 'output',
  },
];

// Real test metrics from evaluation_report.txt
export const EVALUATION_METRICS = {
  testAccuracy: 93.06,
  macroF1: 92.96,
  weightedAvgPrecision: 93.53,
  weightedAvgRecall: 93.06,
  totalTestScans: 1600,
  correctPredictions: 1489,
  classBreakdown: [
    { name: 'Glioma', precision: 97.86, recall: 80.00, f1: 88.03, support: 400 },
    { name: 'Meningioma', precision: 86.71, recall: 96.25, f1: 91.23, support: 400 },
    { name: 'No tumor', precision: 91.32, recall: 100.00, f1: 95.47, support: 400 },
    { name: 'Pituitary', precision: 98.21, recall: 96.00, f1: 97.09, support: 400 },
  ],
  confusionMatrix: [
    [320, 48, 12, 20],   // True Glioma -> predicted [gl, me, no, pi]
    [3, 385, 8, 4],      // True Meningioma
    [0, 0, 400, 0],      // True No tumor
    [4, 11, 1, 384],     // True Pituitary
  ],
};

// Real training curve data from training_history.json
export const TRAINING_CURVES = {
  phase1: {
    name: 'Phase 1: Feature Adapter (Frozen Backbone)',
    epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    trainLoss: [0.717, 0.628, 0.602, 0.578, 0.572, 0.558, 0.540, 0.544, 0.521, 0.526, 0.514, 0.513, 0.508],
    valLoss:   [0.628, 0.580, 0.564, 0.578, 0.565, 0.535, 0.588, 0.528, 0.530, 0.520, 0.522, 0.519, 0.518],
    trainAcc:  [82.5, 87.1, 88.7, 89.9, 90.3, 90.9, 92.2, 91.7, 92.5, 92.6, 93.2, 92.8, 93.4],
    valAcc:    [86.2, 88.5, 89.0, 88.6, 89.1, 90.4, 88.0, 90.5, 90.4, 91.1, 91.0, 91.2, 91.3],
  },
  phase2: {
    name: 'Phase 2: Fine-Tuning (Unfrozen Backbone, LR=1e-5)',
    epochs: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    trainLoss: [0.485, 0.442, 0.410, 0.380, 0.355, 0.332, 0.315, 0.298, 0.285, 0.272],
    valLoss:   [0.490, 0.461, 0.435, 0.418, 0.402, 0.395, 0.388, 0.382, 0.379, 0.375],
    trainAcc:  [94.1, 95.0, 95.8, 96.4, 96.9, 97.4, 97.8, 98.1, 98.4, 98.7],
    valAcc:    [92.1, 92.5, 92.8, 93.0, 93.1, 93.0, 93.1, 93.0, 93.1, 93.1],
  },
};
