// scanController.js
// Handles incoming HTTP requests for MRI scan operations.
// Business logic lives in scanService.js.

const scanService = require('../services/scanService');

/**
 * POST /api/scans/upload
 * Accepts a single MRI image, forwards it to the model service,
 * and returns the prediction result.
 */
const uploadScan = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded.' });
    }

    const result = await scanService.analyzeImage(req.file);
    return res.status(200).json({ success: true, data: result });
  } catch (err) {
    console.error('[scanController] uploadScan error:', err.message);
    return res.status(500).json({ error: 'Failed to process scan.' });
  }
};

/**
 * GET /api/scans/:id
 * Retrieve a previously analysed scan result by its ID.
 */
const getScanById = async (req, res) => {
  try {
    const { id } = req.params;
    const result  = await scanService.getScanById(id);

    if (!result) {
      return res.status(404).json({ error: 'Scan not found.' });
    }

    return res.status(200).json({ success: true, data: result });
  } catch (err) {
    console.error('[scanController] getScanById error:', err.message);
    return res.status(500).json({ error: 'Failed to retrieve scan.' });
  }
};

module.exports = { uploadScan, getScanById };
