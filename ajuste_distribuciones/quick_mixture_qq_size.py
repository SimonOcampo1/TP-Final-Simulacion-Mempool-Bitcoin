"""
VISUALIZACIÓN Q-Q EXACTO - MODELO MEZCLA SEGMENTADA
Universidad Tecnológica Nacional - FRLP
Autor: Simón Tadeo Ocampo
Año: 2025

Genera Q-Q plot exacto para el modelo de mezcla segmentada de tamaños
de transacción (size_bytes) con cutoff fijo=300 bytes.

Objetivo: validar bondad de ajuste de la distribución mezcla mediante
inversión numérica de la CDF combinada.

Escenarios:
 - Baja congestión: small=johnsonsu, large=johnsonsu
 - Alta congestión: small=johnsonsu, large=pareto

Salida: qq_exacto_segmentado_size_{scenario}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


CSV_FILE = 'Toma de Datos/mempool_data_final_20250919_054047.csv'  # Baja congestión
VAR_COLUMN = 'size_bytes'
CUTOFF = 300

DIST_SMALL = 'johnsonsu'
DIST_LARGE = 'johnsonsu'

OUTPUT_FIG = f"qq_exacto_segmentado_size_{'high' if DIST_LARGE=='pareto' else 'low'}.png"

N_POINTS = 140
P_MIN, P_MAX = 0.01, 0.99
MAX_ITERS_BINSEARCH = 55
TOL = 1e-8
FIG_SIZE_SQ = (6.2, 6.2)


def fit_distribution(data: np.ndarray, dist_name: str):
    """Ajusta distribución específica a datos usando MLE"""
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    return dist, params


def mixture_cdf(x, w_s, dist_s, ps, dist_l, pl):
    """CDF de la mezcla: w_s*F_small(x) + (1-w_s)*F_large(x)"""
    return w_s * dist_s.cdf(x, *ps) + (1 - w_s) * dist_l.cdf(x, *pl)


def mixture_ppf(p, w_s, dist_s, ps, dist_l, pl, lo, hi):
    """
    Función cuantil (PPF) de la mezcla mediante búsqueda binaria.
    Encuentra x tal que F_mix(x) = p
    """
    for _ in range(MAX_ITERS_BINSEARCH):
        mid = 0.5 * (lo + hi)
        Fm = mixture_cdf(mid, w_s, dist_s, ps, dist_l, pl)
        
        if abs(Fm - p) < TOL:
            return mid
        
        if Fm < p:
            lo = mid
        else:
            hi = mid
    
    return 0.5 * (lo + hi)


def main():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE)
    
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontró el CSV: {ruta}")
    
    df = pd.read_csv(ruta)
    
    if VAR_COLUMN not in df.columns:
        raise ValueError(f"Columna {VAR_COLUMN} no está en el dataset")
    
    data = df[VAR_COLUMN].values
    data = data[np.isfinite(data) & (data > 0)]
    data.sort()
    
    seg_small = data[data < CUTOFF]
    seg_large = data[data >= CUTOFF]
    
    if len(seg_small) == 0 or len(seg_large) == 0:
        raise ValueError("Segmentos vacíos: revisar cutoff o datos")
    
    w_small = len(seg_small) / len(data)
    w_large = 1 - w_small
    
    dist_s, ps = fit_distribution(seg_small, DIST_SMALL)
    dist_l, pl = fit_distribution(seg_large, DIST_LARGE)
    
    lo = min(seg_small.min(), seg_large.min())
    hi = np.quantile(data, 0.9995)
    
    probs = np.linspace(P_MIN, P_MAX, N_POINTS)
    theo_vals = np.array([
        mixture_ppf(p, w_small, dist_s, ps, dist_l, pl, lo, hi)
        for p in probs
    ])
    emp_vals = np.quantile(data, probs)
    
    plt.figure(figsize=FIG_SIZE_SQ)
    plt.scatter(theo_vals, emp_vals, s=18, alpha=0.65, label='Datos vs Mezcla')
    
    mn = min(theo_vals.min(), emp_vals.min())
    mx = max(theo_vals.max(), emp_vals.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1.2, label='Identidad')
    
    plt.title(f"Q-Q Exacto Mezcla size_bytes (cutoff={CUTOFF})")
    plt.xlabel('Cuantiles Teóricos Mezcla')
    plt.ylabel('Cuantiles Empíricos')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=160)
    plt.close()
    print(f"✅ Figura guardada: {OUTPUT_FIG}")
    
    print("\n--- RESUMEN ---")
    print(f"Archivo: {CSV_FILE}")
    print(f"Segmento pequeño: n={len(seg_small)} ({w_small:.2%}) Dist={DIST_SMALL} params={tuple(round(p,5) for p in ps)}")
    print(f"Segmento grande:  n={len(seg_large)} ({w_large:.2%}) Dist={DIST_LARGE} params={tuple(round(p,5) for p in pl)}")


if __name__ == '__main__':
    main()
