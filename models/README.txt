meter_seg.onnx + meter_rec.onnx: counter locator and digit reader trained on the Kaggle Water Meters dataset.
meter_model.json: digit count, decimal places and the held-out accuracy. meter_split.json: which photos were train/val/test.
Retrain with training/train_meter_models.py (see README). Without these files the app falls back to RapidOCR.
