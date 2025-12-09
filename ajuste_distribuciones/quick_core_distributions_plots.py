"""
VISUALIZACIÓN RÁPIDA - DISTRIBUCIONES CORE
Universidad Tecnológica Nacional - FRLP
Autor: Simón Tadeo Ocampo
Año: 2025

Genera figuras individuales (histograma + PDF global, zoom y Q-Q) para:
 - Intervalos entre llegadas (inter_arrival_seconds)
 - Fee rate (fee_rate_sats_per_byte)
 - Tamaños de transacción (size_bytes) - modelo único

Escenarios: baja y alta congestión (dos CSV distintos con mismo esquema temporal)

Distribuciones utilizadas:
 - Intervalos: Inverse Gamma (invgamma)
 - Fee rate low: Johnson SU (johnsonsu)
 - Fee rate high: Log-Normal (lognorm)
 - Tamaños: Johnson SU (johnsonsu) - modelo único para ambos escenarios

Salida (por escenario low/high):
  interarrivals_{scenario}.png / interarrivals_{scenario}_zoom.png / interarrivals_{scenario}_qq.png
  feerate_{scenario}.png / feerate_{scenario}_zoom.png / feerate_{scenario}_qq.png
  size_single_{scenario}.png / size_single_{scenario}_zoom.png / size_single_{scenario}_qq.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


SCENARIOS = [
    {
        'name': 'low',
        'csv': 'Toma de Datos/mempool_data_final_20250919_054047.csv',
        'fee_dist': 'johnsonsu'
    },
    {
        'name': 'high',
        'csv': 'Toma de Datos/mempool_data_final_20250922_110158.csv',
        'fee_dist': 'lognorm'
    }
]

BINS_MAIN = 120
BINS_ZOOM = 100
ZOOM_MAX_INTER = 3.0
ZOOM_MAX_FEE = 400
ZOOM_MAX_SIZE = 1200
ALPHA_HIST = 0.55
FIG_SIZE_SQ = (6.2, 6.2)
QQ_PROBS = np.linspace(0.01, 0.99, 110)


def _fit(dist_name, data):
    """Ajusta distribución a datos usando MLE"""
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    return dist, params


def _qq_plot(data, dist, params, title, outfile):
    """Genera gráfico Q-Q para validación de ajuste"""
    probs = QQ_PROBS
    theo = dist.ppf(probs, *params)
    emp = np.quantile(data, probs)
    
    plt.figure(figsize=FIG_SIZE_SQ)
    plt.scatter(theo, emp, s=18, alpha=0.65, label='Datos')
    
    mn = min(theo.min(), emp.min())
    mx = max(theo.max(), emp.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1.2)
    
    plt.title(title)
    plt.xlabel('Cuantiles Teóricos')
    plt.ylabel('Cuantiles Empíricos')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=170)
    plt.close()
    print(f"  ✅ Guardado {outfile}")


def _hist_plots(data, dist, params, variable_label, prefix, zoom_max):
    """Genera histograma + PDF en vista global y con zoom"""
    
    x = np.linspace(data.min(), np.quantile(data, 0.995), 1000)
    pdf = dist.pdf(x, *params)
    
    plt.figure(figsize=FIG_SIZE_SQ)
    plt.hist(data, bins=BINS_MAIN, density=True, alpha=ALPHA_HIST, 
             color='#A9CCE3', edgecolor='black', label='Datos')
    plt.plot(x, pdf, 'r-', lw=2, label=f'Ajuste {dist.name}')
    plt.title(f'{variable_label} - vista global')
    plt.xlabel(variable_label)
    plt.ylabel('Densidad')
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out1 = f"{prefix}.png"
    plt.savefig(out1, dpi=170)
    plt.close()
    print(f"  ✅ Guardado {out1}")
    
    data_zoom = data[data <= zoom_max]
    xz = np.linspace(data.min(), min(zoom_max, np.quantile(data_zoom, 0.995) if len(data_zoom) > 10 else zoom_max), 1000)
    pdf_z = dist.pdf(xz, *params)
    
    plt.figure(figsize=FIG_SIZE_SQ)
    plt.hist(data_zoom, bins=BINS_ZOOM, density=True, alpha=ALPHA_HIST, 
             color='#A9CCE3', edgecolor='black', label=f'Datos <= {zoom_max}')
    plt.plot(xz, pdf_z, 'r-', lw=2, label=f'Ajuste {dist.name}')
    plt.title(f'{variable_label} - zoom <= {zoom_max}')
    plt.xlabel(variable_label)
    plt.ylabel('Densidad')
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out2 = f"{prefix}_zoom.png"
    plt.savefig(out2, dpi=170)
    plt.close()
    print(f"  ✅ Guardado {out2}")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    for sc in SCENARIOS:
        csv_path = os.path.join(base, sc['csv'])
        
        if not os.path.isfile(csv_path):
            print(f"[ERROR] Falta {csv_path}, omitiendo {sc['name']}")
            continue
        
        print(f"\n=== Escenario {sc['name']} ===")
        df = pd.read_csv(csv_path)
        
        if 'timestamp_utc' not in df.columns:
            print('[ERROR] No timestamp_utc: no se pueden derivar intervalos')
            continue
        
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df = df.sort_values('timestamp_utc')
        df['inter_arrival_seconds'] = df['timestamp_utc'].diff().dt.total_seconds()
        
        inter = df['inter_arrival_seconds'].values
        inter = inter[np.isfinite(inter) & (inter > 0)]
        
        fee = df['fee_rate_sats_per_byte'].values
        fee = fee[np.isfinite(fee) & (fee > 0)]
        
        sz = df['size_bytes'].values
        sz = sz[np.isfinite(sz) & (sz > 0)]
        
        print(f"\n[INTER-ARRIVALS: {len(inter)} datos]")
        d_inter, p_inter = _fit('invgamma', inter)
        _hist_plots(inter, d_inter, p_inter, 'Inter-arrival (seg)', f'interarrivals_{sc["name"]}', ZOOM_MAX_INTER)
        _qq_plot(inter, d_inter, p_inter, f'Q-Q Inter-arrivals ({sc["name"]})', f'interarrivals_{sc["name"]}_qq.png')
        
        print(f"\n[FEE RATE: {len(fee)} datos]")
        d_fee, p_fee = _fit(sc['fee_dist'], fee)
        _hist_plots(fee, d_fee, p_fee, 'Fee Rate (sats/byte)', f'feerate_{sc["name"]}', ZOOM_MAX_FEE)
        _qq_plot(fee, d_fee, p_fee, f'Q-Q Fee Rate ({sc["name"]})', f'feerate_{sc["name"]}_qq.png')
        
        print(f"\n[SIZE: {len(sz)} datos]")
        d_sz, p_sz = _fit('johnsonsu', sz)
        _hist_plots(sz, d_sz, p_sz, 'Tamaño (bytes)', f'size_single_{sc["name"]}', ZOOM_MAX_SIZE)
        _qq_plot(sz, d_sz, p_sz, f'Q-Q Tamaños ({sc["name"]})', f'size_single_{sc["name"]}_qq.png')


if __name__ == '__main__':
    main()
