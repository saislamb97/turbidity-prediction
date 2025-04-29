# TurbidityPrediction

Comprehensive project for predicting and optimizing turbidity reduction during water treatment using Machine Learning and Optimization Techniques.

---

## 📌 Project Overview

This project focuses on predicting and optimizing the **turbidity reduction** in electrocoagulation-based water treatment systems. Two major versions of the analysis were developed:

- **Version 1:** Initial exploration, cleaning, analysis, and modeling.
- **Version 2:** Enhanced modeling, feature engineering, optimization with Harmony Search, scenario simulation, and Power BI visualization.

Both versions aim to support real-time decision-making for operational settings (Voltage, Time, Number of Electrodes) to achieve efficient water treatment.

---

## 📁 Project Structure

| Folder/File | Description |
|:------------|:------------|
| `electrocoagulation_v1/` | Version 1: cleaned CSV file, Python scripts, generated graphs, and final report. |
| `electrocoagulation_v2/` | Version 2: enhanced version with Jupyter notebook (`.ipynb`), Power BI dashboard, cleaned CSV, Python scripts, advanced graphs, and final report. |
| `raw_data_v1.xlsx` | Raw experimental data used for Version 1 analysis. |
| `raw_data_v2.xlsx` | Cleaned and preprocessed experimental data (Power BI processed) used for Version 2. |
| `Research_Proposal.pdf` | Research proposal outlining project objectives, methodology, and significance. |
| `README.md` | Full project documentation (this file). |
| `requirements.txt` | List of required Python packages. |
| `.gitignore` | Git configuration to exclude temporary files, virtual environments, and other non-essential files from version control. |

---

## 🚀 Version Highlights

### Version 1 (`electrocoagulation_v1/`)
- **Data Cleaning:** Removal of invalid data entries.
- **Run Segmentation:** Based on experimental timing.
- **Exploratory Data Analysis (EDA):** Histograms, scatter plots.
- **Machine Learning Model:**
  - Baseline Linear Regression.
  - Tuned Random Forest Regressor.
- **Outputs:**
  - Turbidity at 5 minutes.
  - Fastest time to achieve minimum turbidity.
- **Deliverable:** Final analytical report.

### Version 2 (`electrocoagulation_v2/`)
- **Advanced Data Engineering:**
  - New feature: `Composite Energy = Voltage × Time × No. of Electrodes`.
- **Modeling:**
  - Random Forest and Support Vector Regressor (SVR).
  - Grouped Cross-Validation (by experiment) to prevent leakage.
- **Optimization:**
  - Hyperparameter tuning using GridSearchCV.
  - Further fine-tuning with **Harmony Search metaheuristic**.
- **Scenario Simulation:**
  - Low, Medium, High resource usage scenarios.
  - Energy-turbidity trade-off visualization.
- **Dashboard:**
  - Power BI file with interactive visualizations.
- **Deliverable:** Full professional report and visualization dashboard.

---

## 📊 Key Technologies

- **Python:** Data analysis, modeling, optimization.
- **Pandas, NumPy:** Data manipulation.
- **Matplotlib, Seaborn:** Data visualization.
- **Scikit-learn:** Machine learning modeling and validation.
- **Harmony Search:** Metaheuristic optimization algorithm.
- **Power BI:** Interactive dashboard for data visualization.

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saislamb97/turbidity-prediction.git
   cd turbidity-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open files:**
   - Jupyter notebooks: `electrocoagulation_v2/Electrocoagulation.ipynb`
   - Power BI dashboard: `electrocoagulation_v2/cleanup_data.pbix`

4. **Run the analysis scripts** to reproduce results or customize the models.

---

## 📚 How to Use

- Analyze Version 1 outputs for basic trends and early modeling.
- Explore Version 2 for:
  - Advanced feature engineering.
  - Machine learning predictions.
  - Optimization insights (Harmony Search).
  - Scenario simulations and operational recommendations.
- Use the Power BI dashboard to interactively explore the dataset and model results.

---

## 📝 Research Proposal

- Full proposal is available in `Research_Proposal.pdf`.
- Includes project motivation, objectives, methodology, expected outcomes.

---

## 📈 Results Summary

- **Best Model:** Tuned Random Forest Regressor (V2).
- **Test R²:** 0.6470
- **Turbidity vs Energy Correlation:** r = -0.742 (strong inverse relationship).
- **Optimization:** Harmony Search improved model performance slightly (MSE reduced from 138.45 to 135.27).
- **Scenario Insights:**
  - High-resource scenario achieved lowest turbidity fastest.
  - Medium-resource scenario offered a good balance of energy usage and treatment performance.

---

## ⚡ Future Work

- Extend to live IoT sensor integration.
- Real-time dashboard updates.
- Include additional features (e.g., pH, conductivity) in prediction models.
- Deploy models as an API service for industrial monitoring systems.

---

## 📄 License

This project is provided for research and educational purposes. Please credit appropriately if used.

---

_"Turning raw water data into optimized, actionable intelligence."_ 💧⚡
