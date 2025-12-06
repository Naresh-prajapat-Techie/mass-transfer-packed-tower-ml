# packed_absorption_project.py
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
import os, joblib

OUT = "packed_outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------
# 1) EQUILIBRIUM RELATION : Y* = m X
# ------------------------------------------
def equilibrium_Ystar(X, m):
    return m * X


# ---------------------------------------------------------
# 2) PACKED TOWER DIFFERENTIAL MODEL (COUNTERCURRENT ODEs)
# ---------------------------------------------------------
def simulate_packed_countercurrent(G, L, Y_in, X_in, H, D, a_s, kGa, m, n_points=200):
    """
    Simple mass transfer model:
    dY/dz = -(kGa / G)(Y - Y*)
    dX/dz = +(kLa / L)(Y - Y*)
    kLa approximated using kGa and alpha factor
    """

    alpha = 0.9  # split factor
    kGa_val = kGa
    kLa_val = kGa_val * (1 - alpha) / alpha

    def odes(z, y):
        Y, X = y
        Ystar = equilibrium_Ystar(X, m)
        dYdz = -(kGa_val / G) * (Y - Ystar)
        dXdz =  (kLa_val / L) * (Y - Ystar)
        return [dYdz, dXdz]

    sol = solve_ivp(
        odes, (0, H), [Y_in, X_in],
        t_eval=np.linspace(0, H, n_points),
        max_step=H/50
    )

    Y_out = sol.y[0, -1]
    X_out = sol.y[1, -1]

    # Approximate NTU & HTU
    NtOG = (kGa_val / G) * H
    HtOG = H / (NtOG + 1e-12)

    return {
        "Y_out": float(Y_out),
        "X_out": float(X_out),
        "NtOG": float(NtOG),
        "HtOG": float(HtOG),
        "kLa": float(kLa_val)
    }


# ---------------------------------------------------------
# 3) GENERATE SYNTHETIC PACKED-TOWER DATASET
# ---------------------------------------------------------
def sample_and_generate(n_samples=50, random_state=1):
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n_samples):
        D  = rng.uniform(0.1, 1.0)        # column diameter (m)
        H  = rng.uniform(0.5, 10.0)       # height (m)
        a_s = rng.uniform(50, 400)        # packing surface area (m²/m³)
        G  = rng.uniform(0.1, 10.0)       # gas molar flow rate (mol/s)
        L  = rng.uniform(0.1, 50.0)       # liquid molar flow (mol/s)
        Y_in = rng.uniform(0.001, 0.2)    # mole fraction (gas)
        X_in = rng.uniform(0.0, 0.05)     # mole fraction (liquid)
        m  = rng.uniform(0.5, 5.0)        # equilibrium slope

        Ug = G / (np.pi * (D/2)**2)       # superficial gas velocity

        # simple empirical kGa correlation
        kGa = 0.1 * (Ug**0.7) * (a_s**0.3) * (H**-0.05) * rng.uniform(0.8, 1.2)

        result = simulate_packed_countercurrent(
            G=G, L=L, Y_in=Y_in, X_in=X_in,
            H=H, D=D, a_s=a_s, kGa=kGa, m=m
        )

        # add experimental noise
        kLa_meas = result["kLa"] * rng.normal(1.0, 0.08)  # 8% noise

        rows.append({
            "D": D, "H": H, "a_s": a_s, "G": G, "L": L,
            "Ug": Ug, "m": m,
            "Y_in": Y_in, "X_in": X_in,
            "kLa_true": result["kLa"],
            "kLa_meas": kLa_meas,
            "Y_out": result["Y_out"],
            "X_out": result["X_out"],
            "NtOG": result["NtOG"],
            "HtOG": result["HtOG"]
        })

    return pd.DataFrame(rows)


# Generate dataset
N_SAMPLES = 20
print("\nGenerating", N_SAMPLES, "samples...")
df = sample_and_generate(n_samples=N_SAMPLES, random_state=42)
df.to_csv(f"{OUT}/packed_synthetic_{N_SAMPLES}.csv", index=False)
print("Dataset saved to:", OUT)


# ---------------------------------------------------------
# 4) FEATURE ENGINEERING
# ---------------------------------------------------------
df["Re_packed"] = (df["G"] * df["D"]) / 1e-3   # surrogate
df["Sc"] = 1.0                                  # constant for simplicity

features = ["D", "H", "a_s", "G", "L", "Ug", "m", "Re_packed"]
X = df[features]
y = df["kLa_meas"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# ---------------------------------------------------------
# 5) LINEAR REGRESSION
# ---------------------------------------------------------
model_lin = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

model_lin.fit(X_train, y_train)
pred_lin = model_lin.predict(X_test)

rmse_lin = np.sqrt(mean_squared_error(y_test, pred_lin))
print("\nLinear Regression:")
print("R² =", r2_score(y_test, pred_lin))
print("RMSE =", rmse_lin)


# ---------------------------------------------------------
# 6) POLYNOMIAL REGRESSION (Degree 2)
# ---------------------------------------------------------
poly = Pipeline([
    ("poly", PolynomialFeatures(2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

poly.fit(X_train, y_train)
pred_poly = poly.predict(X_test)

rmse_poly = np.sqrt(mean_squared_error(y_test, pred_poly))
print("\nPolynomial Regression (degree 2):")
print("R² =", r2_score(y_test, pred_poly))
print("RMSE =", rmse_poly)


# ---------------------------------------------------------
# 7) SAVE MODELS + PREDICTION DATA
# ---------------------------------------------------------
joblib.dump(model_lin, f"{OUT}/lin_model_packed.pkl")
joblib.dump(poly, f"{OUT}/poly2_model_packed.pkl")

pd.DataFrame({
    "y_test": y_test,
    "pred_lin": pred_lin,
    "pred_poly": pred_poly
}).to_csv(f"{OUT}/predictions_packed.csv", index=False)

print("\nSaved all outputs in:", OUT)

