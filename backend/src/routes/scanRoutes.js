const express = require('express');
const router  = express.Router();
const scanController = require('../controllers/scanController');
const upload         = require('../middleware/uploadMiddleware');

// Accept uploaded file under field name 'image' (frontend), 'mri', or any
const uploadFlexible = (req, res, next) => {
  upload.any()(req, res, (err) => {
    if (err) return next(err);
    if (req.files && req.files.length > 0) {
      req.file = req.files[0];
    }
    next();
  });
};

// POST /api/scans/upload or POST /api/predict — upload an MRI scan for analysis
router.post('/upload', uploadFlexible, scanController.uploadScan);
router.post('/', uploadFlexible, scanController.uploadScan);

// GET  /api/scans/:id     — get a previous scan result by ID
router.get('/:id', scanController.getScanById);

module.exports = router;
