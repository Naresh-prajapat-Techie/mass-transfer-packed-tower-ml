# 🧪 Mass Transfer Simulation & Machine Learning for Packed Absorption Towers

A complete project integrating chemical engineering mass-transfer modeling, numerical simulation, and machine learning to predict packed tower performance parameters.

---

## 📌 Table of Contents

<a href="#overview">Overview</a>

<a href="#problem-statement">Problem Statement</a>

<a href="#dataset">Dataset</a>

<a href="#tools--technologies">Tools & Technologies</a>

<a href="#project-structure">Project Structure</a>

<a href="#simulation-methodology">Simulation Methodology</a>

<a href="#exploratory-data-analysis">Exploratory Data Analysis</a>

<a href="#machine-learning-models--results">Machine Learning Models & Results</a>

<a href="#how-to-run-this-project">How to Run This Project</a>

<a href="#engineering-insights">Engineering Insights</a>

<a href="#future-work">Future Work</a>

<a href="#author--contact">Author & Contact</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

This project focuses on the performance analysis of a countercurrent packed absorption tower, a key unit operation in chemical process industries.

A full numerical simulator was developed using Python to model:

Gas–liquid mass transfer

Equilibrium relations

NTU/HTU calculations

Outlet gas & liquid composition profiles

A synthetic dataset (20–50k samples) was generated using these physics-based simulations.
Machine learning models were then trained to predict kLa, NTU, HTU, and outlet concentrations, enabling data-driven performance evaluation.

---

<h2><a class="anchor" id="problem-statement"></a>Problem Statement</h2>

Packed towers are widely used for:

Gas absorption

Stripping operations

Pollution control

Solvent recovery

However:

Experiments are costly

Operating conditions vary widely

kLa and NTU/HTU correlations are complex and nonlinear

This project solves these challenges by:

Simulating thousands of operating conditions

Generating realistic process data

Training ML models to predict tower behavior

Enabling faster design and optimization decisions

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

The dataset is synthetically generated using numerical simulation.

Each row includes:

Input Variables

Gas flow rate (G)

Liquid flow rate (L)

Column height (H)

Column diameter (D)

Packing surface area (aₛ)

Inlet gas & liquid compositions

Equilibrium constant (m)

Mass-transfer coefficient assumptions

Output Variables

kLa (volumetric mass-transfer coefficient)

NTU (Number of Transfer Units)

HTU (Height of Transfer Unit)

Y_out, X_out (outlet compositions)

Dataset example is stored in:
`/data/packed_synthetic_20.csv`

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

Python (NumPy, Pandas, SciPy, Scikit-Learn)

Matplotlib & Seaborn (data visualization)

ODE Solvers for mass transfer calculations

GitHub for documentation & version control

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

mass-transfer-packed-tower-ml/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/                                # Core simulation + ML model
│   └── packed_absorption_project.py
│
├── data/
│   └── packed_synthetic_20.csv          # Example dataset
│
├── models/
│   ├── lin_model_packed.pkl            # Linear Regression model
│   └── poly2_model_packed.pkl          # Polynomial Regression model
│
├── outputs/
│   └── predictions_packed.csv          # Model predictions
│
├── notebooks/                          # Optional notebooks for EDA
├── docs/                               # Report, diagrams, presentation


---

<h2><a class="anchor" id="simulation-methodology"></a>Simulation Methodology</h2>
✔ Mass Transfer Basis

Two-film theory

Differential gas-phase balance

Equilibrium relation 
𝑌
∗
=
𝑚
𝑋
Y
∗
=mX

Variation along column height

✔ Numerical Solution

ODE solver integrates concentration changes

Computes NTU, HTU, kLa

Adds controlled noise to mimic real experimental variation

✔ Data Generation

Monte Carlo sampling of 20–50k operating cases

Produces a rich dataset for ML training

---

<h2><a class="anchor" id="exploratory-data-analysis"></a>Exploratory Data Analysis (EDA)</h2>

Insights from dataset:

Higher gas flow increases NTU but may reduce HTU

Larger packing area increases kLa

Strong nonlinear relation between flow rates and tower efficiency

Outliers correspond to extreme or near-flooding conditions

Visualizations include:

Correlation heatmaps

kLa vs flow rates

NTU–HTU behavior

Residual plots

---

<h2><a class="anchor" id="machine-learning-models--results"></a>Machine Learning Models & Results</h2>
Models Used:

Linear Regression

Polynomial Regression (Degree 2) – Best performer

Sample Results:
Linear Regression:
R² = 0.936
RMSE = 0.0916

Polynomial Regression:
R² = 0.913
RMSE = 0.106

Observations:

Linear model performs surprisingly well

Polynomial model captures deeper nonlinear behavior

Residuals show low bias and good spread

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

Clone the repository:

git clone https://github.com/yourusername/mass-transfer-packed-tower-ml.git
cd mass-transfer-packed-tower-ml

Install requirements:

pip install -r requirements.txt

Run the simulation + ML pipeline:

python src/packed_absorption_project.py

View outputs in:

/data/

/models/

/outputs/

---

<h2><a class="anchor" id="engineering-insights"></a>Engineering Insights</h2>

From simulations and ML modeling:

Increasing gas flow improves mass transfer but increases pressure drop

HTU decreases with higher kLa → more efficient tower

Larger packing surface area significantly enhances absorption

ML models can approximate tower behavior without extensive experimentation

---

<h2><a class="anchor" id="future-work"></a>Future Work</h2>

Future enhancements can include:

Multi-component absorption

Gas absorption with reaction

Flooding & loading predictions

Deep learning models (ANNs)

Real experimental data integration

Streamlit web app for real-time prediction

---

<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

Naresh Prajapat
Chemical Engineering Student

📧 Email: 223120@student.nitandhra.ac.in





