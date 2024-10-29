"""
Code that does Bayesian Model Averaging in the quantile regression model

Adopted from paper : https://doi.org/10.1016/j.ijforecast.2016.07.005
Title : Quantile regression forecasts of inflation under model uncertainty

@author: fawdywahyu18
"""

import numpy as np
from scipy.stats import invgamma, gamma
from numpy.polynomial.polynomial import Polynomial
import scipy.io
import pandas as pd

# ==============| Define Functions |=============

def transx(x, tcode):
    # Transform series based on transformation code `tcode`
    small = 1.0e-40
    relvarm = 0.00000075
    relvarq = 0.000625
    n = len(x)
    y = np.zeros(n)

    if tcode == 1:
        y = x
    elif tcode == 2:
        y[1:] = x[1:] - x[:-1]
    elif tcode == 3:
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    elif tcode == 4:
        if np.min(x) < small:
            y = np.nan
        else:
            y = np.log(x)
    elif tcode == 5:
        if np.min(x) < small:
            y = np.nan
        else:
            x = np.log(x)
            y[1:] = x[1:] - x[:-1]
    elif tcode == 6:
        if np.min(x) < small:
            y = np.nan
        else:
            x = np.log(x)
            y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    elif tcode == 7:
        if np.min(x) < small:
            y = np.nan
        else:
            x = np.log(x)
            y, _ = detrend1(x, relvarm)
    elif tcode == 8:
        if np.min(x) < small:
            y = np.nan
        else:
            x = np.log(x)
            y, _ = detrend1(x, relvarq)
    elif tcode == 16:
        if np.min(x) < small:
            y = np.nan
        else:
            x = np.log(x)
            y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    elif tcode == 17:
        if np.min(x) < small:
            y = np.nan
        else:
            x = np.log(x)
            y[13:] = x[13:] - x[12:-1] - x[1:-12] + x[:-13]
    else:
        y = np.nan

    return y

def detrend1(x, order=1):
    t = np.arange(len(x))
    p = Polynomial.fit(t, x, order)
    trend = p(t)
    detrended_x = x - trend
    return detrended_x, trend

def yfcsta(y, tcode, nph):
    n = len(y)
    yf = np.zeros(n)

    if tcode == 1 or tcode == 4:
        yf[:n - nph] = y[nph:]
    elif tcode == 2 or tcode == 5:
        for t in range(n - nph):
            yf[t] = np.sum(y[t + 1:t + nph + 1])
    elif tcode == 3 or tcode == 6:
        for t in range(n - nph):
            yf[t] = np.sum(np.cumsum(y[t + 1:t + nph + 1]))
    else:
        raise ValueError("Invalid Transformation Code in yfcsta")

    yf = yf / nph
    if tcode == 1 or tcode == 4:
        yf = yf * nph

    return yf

def mlag2(X, p):
    
    Xlag = np.zeros_like(X, dtype=float)
    Xlag[p:] = X[:-p]

    return Xlag

def draw_ig(mu, lam):
    v0 = np.random.randn()**2
    x1 = mu + (0.5 * (mu**2) * v0) / lam - (0.5 * mu / lam) * np.sqrt(4 * mu * lam * v0 + (mu**2) * (v0**2))
    x2 = (mu**2) / x1
    p1_v0 = mu / (mu + x1)

    if np.random.rand() > p1_v0:
        return x1
    else:
        return x2

# ==============| Main Code Block |=============

# Load data

# List nama sheet (sesuai kolom kedua cell_array)
names = [
    'CPI', 'IPM', 'HSTARTS', 'CUM', 'M1', 'RCOND', 'RCONS', 
    'RG', 'RINVBF', 'ROUTPUT', 'RUC', 'ULC', 'WSD', 
    'DYS (BAA-AAA)', 'NAPM', 'NAPMII', 'NAPMNOI'
]

# List angka arbitrer
tcode_numbers = [5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 1, 1, 1, 1]

# Buat cell_array kosong
cell_array = []

# Load data dari file Excel sesuai nama sheet di kolom kedua cell_array
file_path = "ALL_DATA.xlsx"  # Ganti dengan path file Anda
for i, sheet_name in enumerate(names):
    # Baca data dari sheet yang sesuai, abaikan kolom pertama dan baris pertama
    data_load = pd.read_excel(file_path, sheet_name=sheet_name) # Salah cara load DATA
    data_load = data_load.iloc[:, 1:].to_numpy()  # Hapus kolom pertama setelah memuat
    # Bulatkan data hingga dua digit desimal
    data_load = np.round(data_load, 2)

    
    # Tambahkan row ke cell_array
    cell_array.append([data_load, sheet_name, tcode_numbers[i]])

data = cell_array

# -----------------| USER INPUT |-----------------
nsave = 2000
nburn = 2000
ntot = nsave + nburn
iter = 500
lags = 2
nfore = 1
# ------------------------------------------------

# Transform to stationarity
all_series = []
for i in range(len(data)):
    tcode = data[i][2]
    if tcode == 5:
        all_series.append(400 * transx(data[i][0][:, -1], tcode))
    elif tcode == 4 or tcode == 2:
        all_series.append(4 * transx(data[i][0][:, -1], tcode))
    else:
        all_series.append(data[i][0][:, -1])

all_series = np.column_stack(all_series) #Nilai masih berbeda di bagian ini.

# Specify which series are LHS and RHS
Yraw_full = yfcsta(all_series[:, 0], data[0][2], nfore)
Yraw_full_lags = all_series[:, 0]
Zraw = all_series[:, 1:]
Yraw_full = Yraw_full[:-nfore]
Yraw_full_lags = Yraw_full_lags[:-nfore]
Zraw = Zraw[:-nfore, :]

ylag = mlag2(Yraw_full_lags, lags - 1)
if lags > 0:
    Xraw = np.column_stack([np.ones((ylag.shape[0] - lags + 1, 1)), Yraw_full_lags[lags-1:], ylag[lags-1:], Zraw[lags-1:]])
    Yraw = Yraw_full[lags-1:]
else:
    Xraw = np.column_stack([np.ones((ylag.shape[0], 1)), Zraw])
    Yraw = Yraw_full

y = Yraw
x = Xraw
T, p = x.shape

# ==============| Get OLS coefficients
beta_OLS = np.linalg.lstsq(x, y, rcond=None)[0]
sigma_OLS = np.sum((y - x @ beta_OLS)**2) / (T - p - 1)

# ==============| Define priors
V = 9 * np.eye(p)
Vinv = np.linalg.inv(V)
a0, b0 = 0.1, 0.1

# ==============| Initialize vectors
quant = np.arange(5, 100, 5) / 100
n_q = len(quant)
beta = np.zeros((p, n_q))
z = np.ones((T, n_q))
sigma2 = np.zeros((1, n_q))
theta = np.zeros((1, n_q))
tau_sq = np.zeros((1, n_q))
beta_draws = np.zeros((p, n_q, nsave))

for irep in range(ntot):
    if irep % iter == 0:
        print(f'Iteration {irep}')

    for q in range(n_q):
        tau_sq[:, q] = 2 / (quant[q] * (1 - quant[q]))
        theta[:, q] = (1 - 2 * quant[q]) / (quant[q] * (1 - quant[q]))

        a1 = a0 + 3 * T / 2
        sse = (y - x @ beta[:, q] - theta[:, q] * z[:, q]) ** 2
        a2 = b0 + np.sum(sse / (2 * z[:, q] * tau_sq[:, q])) + np.sum(z[:, q])
        sigma2[0, q] = 1 / gamma.rvs(a1, scale=1 / a2)

        U = np.diag(1 / (np.sqrt(sigma2[0, q]) * tau_sq[:, q] * z[:, q]))
        y_tilde = y - theta[:, q] * z[:, q]
        xsq = x.T @ U
        V_beta = np.linalg.inv(xsq @ x + Vinv)
        miu_beta = V_beta @ (xsq @ y_tilde)
        beta[:, q] = miu_beta + np.linalg.cholesky(V_beta).T @ np.random.randn(p)

        for t in range(T):
            k1 = np.sqrt(theta[:, q]**2 + 2 * tau_sq[:, q]) / np.abs(y[t] - x[t] @ beta[:, q])
            k2 = (theta[:, q]**2 + 2 * tau_sq[:, q]) / tau_sq[:, q]
            z[t, q] = max(1 / draw_ig(k1, k2), 1e-4)

    if irep >= nburn:
        beta_draws[:, :, irep - nburn] = beta

