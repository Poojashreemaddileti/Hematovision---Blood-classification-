# HematoVision — Blood Cell Classification

HematoVision is a Flask + TensorFlow web application for blood-cell image classification.

## Features

- Existing trained `trained_model.h5` inference pipeline retained.
- Upload JPG, JPEG, PNG or WEBP blood-cell images.
- Classifies images into four supported cell classes.
- Displays prediction and confidence immediately.
- Responsive medical-AI interface retained.
- No MySQL or TiDB connection.
- No database.
- No prediction history.
- Uploaded images are processed in memory and are not saved by the application.
- `/health` endpoint included.
- Gunicorn, Docker and Render deployment files included.

## Local run

```cmd
python app.py
```

Open:

`http://127.0.0.1:5000`

## Production

Docker:

```cmd
docker build -t hematovision .
docker run -p 8080:8080 hematovision
```

## Model

The existing class order is preserved:

`eosinophil, lymphocyte, monocyte, neutrophil`

The system is a project-level AI classifier and should not be treated as a standalone clinical diagnosis.
