# 🚀 Global Enterprise RFM Customer Intelligence Platform

A next-generation, production-ready Machine Learning and Customer Segmentation solution built with Python, Streamlit, and Plotly. This platform combines RFM (Recency, Frequency, Monetary) analysis with advanced unsupervised clustering (K-Means) and Log-Transform feature scaling to help enterprises identify customer personas and trigger automated marketing playbooks.

---

## ✨ Key Features

- **📊 Executive Overview Dashboard:** High-level executive KPIs including Total Portfolio Volume, Mean Recency, Purchase Frequency, and Monetary metrics alongside interactive segment share visualizations.
- **📈 Interactive Analytics Suite:** Deep-dive Plotly visuals featuring Log-Transformed RFM Scatter plots, Recency Distribution Histograms, Outlier Boxplots per persona, and Pearson Correlation Heatmaps.
- **⚡ Real-Time Inference & Playbook Engine:** Instantly score individual customer behavior (Recency, Frequency, Monetary value) to assign automated AI-driven marketing actions and perks.
- **📁 Batch CSV Bulk Scoring & Mapping:** Upload custom customer datasets with smart column auto-detection and manual mapping configuration to process and download bulk segmentations.
- **📉 ML Optimization (The Elbow Method):** Scientific validation interface visualizing inertia curves to confirm the optimal cluster count ($K=4$).
- **📋 Audit Telemetry:** Tracks system state, deployment architectures, and engine configurations.

---

## 🛠️ Tech Stack & Architecture

- **Frontend & UI:** Streamlit (with custom dark-mode enterprise CSS theme)
- **Machine Learning:** Scikit-Learn (K-Means Clustering), NumPy, Pandas
- **Data Visualization:** Plotly
- **Data Transformation:** Log Transformations (`np.log1p`) for skewed distribution correction

---

## 📂 Project Structure

```text
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── model.pkl              # Trained K-Means clustering model (if applicable)
└── README.md              # Project documentation