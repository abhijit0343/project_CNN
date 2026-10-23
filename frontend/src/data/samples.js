// data/samples.js
// Curated real sample MRI scans for 1-click test exploration.

export const SAMPLE_SCANS = [
  {
    id: 'sample-glioma',
    name: 'Te-gl_1.jpg',
    expectedClass: 'glioma',
    label: 'Glioma Case (High Grade)',
    url: '/samples/sample_glioma.jpg',
    description: 'Axial T1-weighted contrast scan showing a heterogeneous, infiltrative intra-axial lesion in the left frontotemporal region.',
    sliceOrientation: 'Axial T1+C',
    findings: 'Peripheral ring enhancement with central necrotic core and mass effect on frontal horn.',
  },
  {
    id: 'sample-meningioma',
    name: 'Te-me_1.jpg',
    expectedClass: 'meningioma',
    label: 'Meningioma Case',
    url: '/samples/sample_meningioma.jpg',
    description: 'Extra-axial dural-based mass with vivid, homogeneous contrast uptake along the parasagittal convexity.',
    sliceOrientation: 'Coronal / Axial T1+C',
    findings: 'Characteristic dural tail sign and broad-based dural attachment without brain parenchyma invasion.',
  },
  {
    id: 'sample-pituitary',
    name: 'Te-pi_1.jpg',
    expectedClass: 'pituitary',
    label: 'Pituitary Macroadenoma',
    url: '/samples/sample_pituitary.jpg',
    description: 'Sellar expansion with suprasellar extension contacting the optic chiasm in the pituitary fossa.',
    sliceOrientation: 'Coronal T1+C',
    findings: 'Enlarged sella turcica, waist-like indentation at the diaphragma sellae (“snowman” configuration).',
  },
  {
    id: 'sample-notumor',
    name: 'Te-no_1.jpg',
    expectedClass: 'notumor',
    label: 'Healthy / No Mass',
    url: '/samples/sample_notumor.jpg',
    description: 'Unremarkable brain MRI. Bilateral cerebral hemispheres and midline ventricular system demonstrate normal anatomical morphology.',
    sliceOrientation: 'Axial T2 / FLAIR',
    findings: 'No focal parenchymal signal abnormality, midline shift, or mass effect identified.',
  },
];
