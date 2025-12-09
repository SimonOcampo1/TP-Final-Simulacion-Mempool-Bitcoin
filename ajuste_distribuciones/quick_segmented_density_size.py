"""
VISUALIZACIÓN DENSIDAD - MODELO SEGMENTADO
Universidad Tecnológica Nacional - FRLP
Autor: Simón Tadeo Ocampo
Año: 2025

Genera gráficos del modelo segmentado de tamaños (size_bytes) para
baja y alta congestión con cutoff fijo=300 bytes:
 - Histograma + PDFs de cada segmento
 - Curva mezcla
 - Línea vertical del cutoff
 - Zoom doble: panorámico y zoom en rango bajo

Escenarios:
  LOW: mempool_data_final_20250919_054047.csv (johnsonsu + johnsonsu)
  HIGH: mempool_data_final_20250922_110158.csv (johnsonsu + pareto)

Salida: 
  segmentado_size_low.png, segmentado_size_low_zoom.png
  segmentado_size_high.png, segmentado_size_high_zoom.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


CUTOFF = 300
BINS_MAIN = 140
BINS_ZOOM = 120
ZOOM_MAX = 1200
ALPHA_HIST = 0.55
FIG_SIZE_SQ = (6.2, 6.2)

SCENARIOS = [
    {
        'name': 'low',
        'csv': 'Toma de Datos/mempool_data_final_20250919_054047.csv',
        'dist_small': 'johnsonsu',
        'dist_large': 'johnsonsu'
    },
    {
        'name': 'high',
        'csv': 'Toma de Datos/mempool_data_final_20250922_110158.csv',
        'dist_small': 'johnsonsu',
        'dist_large': 'pareto'
    }
]


def fit(data, dist_name):
    """Ajusta distribución a datos usando MLE"""
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    return dist, params


def build_and_plot(data_all, scenario):
    """
    Construye y grafica modelo segmentado para un escenario.
    Genera figura global y figura con zoom.
    """
    data = data_all[(data_all > 0) & np.isfinite(data_all)]
    data.sort()
    
    seg_s = data[data < CUTOFF]
    seg_l = data[data >= CUTOFF]
    
    if len(seg_s) == 0 or len(seg_l) == 0:
        print(f"[WARN] Segmentos vacíos en {scenario['name']} - omitiendo.")
        return
    
    w_s = len(seg_s) / len(data)
    w_l = 1 - w_s
    
    ds, ps = fit(seg_s, scenario['dist_small'])
    dl, pl = fit(seg_l, scenario['dist_large'])
    
    x_main = np.linspace(data.min(), np.quantile(data, 0.995), 1200)
    x_zoom = np.linspace(data.min(), min(ZOOM_MAX, np.quantile(data, 0.99)), 1000)
    
    pdf_s_main = ds.pdf(x_main, *ps)
    pdf_l_main = dl.pdf(x_main, *pl)
    pdf_mix_main = w_s * pdf_s_main + w_l * pdf_l_main
    
    pdf_s_zoom = ds.pdf(x_zoom, *ps)
    pdf_l_zoom = dl.pdf(x_zoom, *pl)
    pdf_mix_zoom = w_s * pdf_s_zoom + w_l * pdf_l_zoom
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_SQ)
    ax.hist(data, bins=BINS_MAIN, density=True, alpha=ALPHA_HIST, 
            color='#A9CCE3', edgecolor='black', label='Datos')
    ax.plot(x_main, pdf_s_main, color='#1E8449', lw=2, 
            label=f"Seg<={CUTOFF} ({scenario['dist_small']})")
    ax.plot(x_main, pdf_l_main, color='#CB4335', lw=2, 
            label=f"Seg>{CUTOFF} ({scenario['dist_large']})")
    ax.plot(x_main, pdf_mix_main, 'k--', lw=2, label='Mezcla')
    ax.axvline(CUTOFF, color='black', ls=':', lw=1.8, label='Cutoff')
    
    ax.set_title(f"Tamaños segmentado ({scenario['name']}) - vista global")
    ax.set_xlabel('Tamaño (bytes)')
    ax.set_ylabel('Densidad')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    
    out_main = f"segmentado_size_{scenario['name']}.png"
    fig.savefig(out_main, dpi=170)
    plt.close()
    print(f"✅ Guardado {out_main} (w_small={w_s:.2%}, w_large={w_l:.2%})")
    
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE_SQ)
    ax2.hist(data[data <= ZOOM_MAX], bins=BINS_ZOOM, density=True, alpha=ALPHA_HIST, 
             color='#A9CCE3', edgecolor='black', label='Datos (<= zoom)')
    ax2.plot(x_zoom, pdf_s_zoom, color='#1E8449', lw=2, 
             label=f"Seg<={CUTOFF} ({scenario['dist_small']})")
    ax2.plot(x_zoom, pdf_l_zoom, color='#CB4335', lw=2, 
             label=f"Seg>{CUTOFF} ({scenario['dist_large']})")
    ax2.plot(x_zoom, pdf_mix_zoom, 'k--', lw=2, label='Mezcla')
    ax2.axvline(CUTOFF, color='black', ls=':', lw=1.8, label='Cutoff')
    ax2.set_xlim(data.min(), ZOOM_MAX)
    
    ax2.set_title(f"Tamaños segmentado ({scenario['name']}) - zoom <= {ZOOM_MAX} bytes")
    ax2.set_xlabel('Tamaño (bytes)')
    ax2.set_ylabel('Densidad')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    
    out_zoom = f"segmentado_size_{scenario['name']}_zoom.png"
    fig2.savefig(out_zoom, dpi=170)
    plt.close()
    print(f"✅ Guardado {out_zoom}")
    
    print(f"Parámetros seg pequeño ({scenario['name']}): {tuple(round(p,5) for p in ps)}")
    print(f"Parámetros seg grande  ({scenario['name']}): {tuple(round(p,5) for p in pl)}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for sc in SCENARIOS:
        csv_path = os.path.join(base_dir, sc['csv'])
        
        if not os.path.isfile(csv_path):
            print(f"[ERROR] No existe {csv_path}, omitiendo escenario {sc['name']}")
            continue
        
        df = pd.read_csv(csv_path)
        
        if 'size_bytes' not in df.columns:
            print(f"[ERROR] Falta columna size_bytes en {sc['csv']}")
            continue
        
        print(f"\n=== Escenario {sc['name']} ===")
        build_and_plot(df['size_bytes'].values, sc)


if __name__ == '__main__':
    main()
