// scanService.js
// Business logic for MRI scan analysis.
// Invokes the trained NeuroScan CNN directly via Python child_process,
// loading neuroscan_model.pkl from the models directory.

const { execFile } = require('child_process');
const path         = require('path');
const fs           = require('fs');

/**
 * Predicts the tumour class for an uploaded MRI image.
 *
 * @param {Express.Multer.File} file - The uploaded file object from multer
 * @returns {Promise<Object>} Prediction result formatted for frontend
 */
const analyzeImage = async (file) => {
  return new Promise((resolve, reject) => {
    console.log('[scanService] analyzeImage called with:', file.originalname);

    // Absolute paths
    const rootDir    = path.resolve(__dirname, '../../../');
    const scriptPath = path.resolve(rootDir, 'model/src/predict.py');
    const pklPath    = path.resolve(rootDir, 'model/src/models/neuroscan_model.pkl');
    const ptPath     = path.resolve(rootDir, 'model/src/models/neuroscan_final.pt');

    // Prefer pickle bundle if it exists, otherwise fallback to .pt checkpoint
    const modelToUse = fs.existsSync(pklPath) ? pklPath : ptPath;

    // Detect Python executable (venv if available, otherwise system python)
    const venvPythonWin = path.resolve(rootDir, 'venv/Scripts/python.exe');
    const pythonBin     = fs.existsSync(venvPythonWin) ? venvPythonWin : 'python';

    const args = [
      scriptPath,
      file.path,
      '--model', modelToUse,
      '--top-k', '4',
      '--json',
    ];

    console.log(`[scanService] Running: ${pythonBin} ${args.join(' ')}`);

    execFile(pythonBin, args, { maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        console.error('[scanService] Python execution error:', stderr || error.message);
        return reject(new Error('Model inference failed during processing.'));
      }

      try {
        // Robustly locate JSON object in stdout even if warnings were printed
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('Predictor did not output valid JSON: ' + stdout);
        }

        const data = JSON.parse(jsonMatch[0]);

        resolve({
          filename   : file.originalname,
          prediction : data.label,
          confidence : data.confidence,
          all_scores : data.scores,
          top_k      : data.top_k,
          model_used : path.basename(modelToUse),
          message    : 'Analysis complete.',
        });
      } catch (parseErr) {
        console.error('[scanService] Failed to parse model output:', stdout);
        reject(new Error('Failed to parse prediction result.'));
      }
    });
  });
};

/**
 * Retrieves a previously analysed scan result by ID.
 *
 * @param {string} id - Scan identifier
 * @returns {Promise<Object|null>} Scan data or null
 */
const getScanById = async (id) => {
  console.log('[scanService] getScanById called with:', id);
  return null;
};

module.exports = { analyzeImage, getScanById };
