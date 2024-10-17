 # Laptop Prices Analysis and Prediction

### Project Overview

This project focuses on analyzing and predicting laptop prices using the dataset sourced from [Kaggle Laptop Prices Dataset](https://www.kaggle.com/datasets/owm4096/laptop-prices). The project is aimed at understanding the underlying factors that affect laptop pricing, performing various experiments with machine learning models using MLflow, and deploying a full end-to-end machine learning pipeline.

---

## Goals

1. **Exploratory Data Analysis (EDA):**  
   - Understand the data through visualization and statistical analysis.
   - Uncover insights into the features that drive laptop prices.

2. **Model Training & Experimentation:**  
   - Train various machine learning models.
   - Use **MLflow** to track and compare different experiments.
   - Choose the best-performing model based on evaluation metrics.

3. **Pipeline Deployment:**  
   - Develop a robust machine learning pipeline.
   - Automate data preprocessing, model training, evaluation, and prediction.
   - Ensure reproducibility of the results.

---

## Dataset Information

- **Source:** [Kaggle Laptop Prices Dataset](https://www.kaggle.com/datasets/owm4096/laptop-prices)
- **Description:** The dataset contains various features related to laptops (brand, RAM, storage, GPU, etc.) along with their prices.

---

## Project Structure

```bash
.
├── data/                 # Directory containing the dataset
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── src/                  # Source code for pipeline, data processing, and model training
├── mlruns/               # MLflow directory for tracking experiments (default one)
├── predict_pricing/      # MLflow directory for tracking pricing experiments
├── README.md             # Project documentation
└── requirements.txt      # List of project dependencies
```

## Installation

To run this project, ensure you have Python 3.10 or higher installed on your machine. Follow the steps below to set up the environment and install the required dependencies:

1. **Clone the Repository:**
   Clone this repository to your local machine.
   ```bash
   git clone <repository-link>
