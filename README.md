# Predictive Maintenance System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-CI%2FCD-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-success)

</p>

---

## Overview

Predictive Maintenance System is an end-to-end Machine Learning project designed to identify abnormal machine behavior before equipment failure occurs. The system analyzes sensor readings collected from industrial machines, detects anomalies using unsupervised learning algorithms, and helps organizations reduce downtime, maintenance costs, and unexpected equipment failures.

The project demonstrates a complete machine learning workflow, including data preprocessing, anomaly detection, model evaluation, REST API development, Docker containerization, and CI/CD integration.

---

## Problem Statement

Unexpected equipment failures can significantly impact manufacturing operations by causing production delays, increasing maintenance costs, and reducing operational efficiency.

Traditional maintenance approaches such as reactive and scheduled maintenance are often inefficient because they either respond after failures occur or replace components that are still functioning properly.

The objective of this project is to develop an intelligent predictive maintenance system capable of identifying abnormal machine behavior before failures occur.

---

## Solution

This project applies unsupervised machine learning algorithms to analyze industrial sensor data and detect anomalous operating conditions. The trained models identify machines that deviate from normal behavior, allowing maintenance teams to perform proactive maintenance and reduce operational risks.

The prediction service is deployed through FastAPI, making it easy to integrate into industrial monitoring systems.

---

## Features

- Industrial sensor data preprocessing
- Automated anomaly detection
- Machine health monitoring
- Data visualization
- Feature engineering
- REST API using FastAPI
- Docker support
- GitHub Actions CI/CD pipeline
- Modular project architecture
- Production-ready deployment

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest, DBSCAN |
| API Development | FastAPI |
| Model Serialization | Joblib / Pickle |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Development Environment | PyCharm |

---

## Project Structure

```text
Predictive-Maintenance
│
├── .github
│   └── workflow
│       └── ci-cd.yml
│
├── model
│   ├── isolation_forest.pkl
│   ├── dbscan_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── app.py
├── main.py
├── preprocessing.py
├── data.csv
├── dbscan_results.csv
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Dataset

The project uses industrial equipment sensor data collected from machines operating under different conditions.

Example features include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type
- Sensor Measurements

**Target**

- Machine Health Status
- Anomaly Detection

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Predictive-Maintenance.git
```

Navigate to the project

```bash
cd Predictive-Maintenance
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server

```bash
uvicorn app:app --reload
```

Access the API documentation

```
http://127.0.0.1:8000/docs
```

---

## Project Workflow

```text
Industrial Sensor Data
          │
          ▼
Data Cleaning
          │
          ▼
Feature Engineering
          │
          ▼
Data Preprocessing
          │
          ▼
Model Training
          │
          ├──────────────┐
          ▼              ▼
Isolation Forest     DBSCAN
          │              │
          └──────┬───────┘
                 ▼
Anomaly Detection
                 │
                 ▼
REST API Deployment
                 │
                 ▼
Production Monitoring
```

---

## Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Duplicate removal
- Feature scaling
- Data normalization
- Feature transformation
- Data validation

---

## Machine Learning Models

The project utilizes unsupervised learning algorithms for anomaly detection:

- Isolation Forest
- DBSCAN Clustering

These algorithms identify abnormal machine behavior without requiring labeled failure data.

---

## Model Evaluation

The system evaluates anomaly detection performance using:

- Anomaly Score Analysis
- Cluster Distribution
- Outlier Detection
- Visual Inspection
- Prediction Consistency

---

## API Example

### Request

```json
{
  "air_temperature": 298.5,
  "process_temperature": 309.4,
  "rotational_speed": 1500,
  "torque": 45.2,
  "tool_wear": 120
}
```

### Response

```json
{
  "prediction": "Normal",
  "anomaly_score": 0.12
}
```

---

## Docker Support

Build the Docker image

```bash
docker build -t predictive-maintenance .
```

Run the container

```bash
docker run -p 8000:8000 predictive-maintenance
```

---

## CI/CD Pipeline

The repository includes a GitHub Actions workflow that automates:

- Dependency installation
- Environment setup
- Project validation
- Build verification
- Continuous Integration
- Deployment readiness

---

## Business Applications

This solution can be used in:

- Manufacturing Industries
- Smart Factories
- Industrial IoT Platforms
- Equipment Monitoring Systems
- Automotive Manufacturing
- Oil and Gas Industry
- Power Plants
- Predictive Maintenance Platforms

---

## Skills Demonstrated

- Python Programming
- Data Preprocessing
- Feature Engineering
- Anomaly Detection
- Unsupervised Machine Learning
- Isolation Forest
- DBSCAN Clustering
- FastAPI Development
- REST API Development
- Docker
- GitHub Actions
- CI/CD
- Model Deployment

---

## Future Improvements

Future enhancements include:

- Real-time sensor monitoring
- IoT device integration
- MLflow experiment tracking
- Cloud deployment on AWS, Azure, or GCP
- Interactive monitoring dashboard
- Automated model retraining
- Explainable AI for anomaly detection
- Streaming data support using Kafka

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Author

**Farsan K**

Aspiring Data Scientist | Machine Learning Engineer

**GitHub:** https://github.com/Farsan-k

**LinkedIn:** https://www.linkedin.com/in/farsank/
