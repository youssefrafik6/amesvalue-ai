# Verification record

The local run used the provided Excel workbook directly through pandas and xlrd. The separate R run read it with readxl and reproduced the original stepwise fit, selected terms, VIF sensitivity and sample predictions. The historical workflow generated 31 figure pages. Test metrics were recalculated from holdout predictions for all 16 candidates under both protocols.

Ruff and pytest passed on Python 3.11 and 3.13 on Apple Silicon (six tests each). The tests exercise all API endpoints, invalid property input, calibration ranks, parcel isolation, workbook shape and all ten Streamlit pages. A public-artifact smoke test verifies prediction with the workbook and all record-level results absent. Docker Compose built both services and the running API returned a verified estimate and bounds. Screenshots were taken from the running Streamlit app.

A warm test of 60 single-record predictions on Apple Silicon measured 3.80 ms median and 4.28 ms p95 inside the prediction engine. Those times exclude network and startup overhead. Training and batch inference times for each candidate appear in the model comparison CSV.

Community Cloud deployment and hosted GitHub CI runs await publication. The local R environment was snapshotted with renv. Restoring the lock in a fresh R environment remains unverified.
