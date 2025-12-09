"""
MODELO DE SIMULACIÓN DE EVENTOS DISCRETOS - MEMPOOL DE BLOCKCHAIN
Universidad Tecnológica Nacional - FRLP
Autor: Simón Tadeo Ocampo
Año: 2025

Modelo de simulación de la mempool de Bitcoin utilizando SimPy.
Implementa procesamiento por lotes con disciplina de cola por prioridad.

Factores experimentales:
- Lambda: [0.0005, 0.0015, 0.004] (modificador de tasa de llegada)
- Escenario: [1, 2] (Baja/Alta congestión)
- Capacidad: [0.5M, 1.0M, 2.0M] bytes por bloque
- Réplicas: 10 por configuración
Total: 180 corridas (3 × 2 × 3 × 10)
"""

import simpy
import random
import numpy as np
import pandas as pd
from datetime import datetime


def johnson_su(n, gamma, delta, xi, lam):
    """Generador vectorizado de distribución Johnson SU"""
    z = np.random.normal(0, 1, n)
    return xi + lam * np.sinh((z - gamma) / delta)


def generar_trafico(n, escenario, factor_lambda):
    """
    Genera vectores de tráfico basados en distribuciones empíricas ajustadas.
    
    Args:
        n: Número de transacciones a generar
        escenario: 1 (baja congestión) o 2 (alta congestión)
        factor_lambda: Multiplicador de intervalos (< 1 aumenta carga, > 1 reduce carga)
    
    Returns:
        dict: Diccionario con arrays 'inter', 'fees', 'sizes'
    """
    data = {}
    
    if escenario == 1:
        shape, scale_p, loc = 0.521501, 0.00781445, -0.00150761
        raw_inter = (scale_p / np.random.gamma(shape, 1, n)) + loc
        data['inter'] = np.maximum(raw_inter * factor_lambda, 0.0001)
        
        data['fees'] = np.maximum(johnson_su(n, -2.8088, 0.796842, 0.399676, 0.0399706), 1)
        data['sizes'] = np.maximum(johnson_su(n, -1.813859, 0.681927, 187.508001, 8.750103), 60)
        
    else:
        shape, scale_p, loc = 0.565951, 0.00880769, -0.00133821
        raw_inter = (scale_p / np.random.gamma(shape, 1, n)) + loc
        data['inter'] = np.maximum(raw_inter * factor_lambda, 0.0001)
        
        data['fees'] = np.maximum(johnson_su(n, -3.59146, 0.977982, 0.335296, 0.0572708), 1)
        
        is_small = np.random.rand(n) <= 0.552
        sizes = np.zeros(n)
        
        idx_s = np.where(is_small)[0]
        if len(idx_s) > 0:
            sizes[idx_s] = johnson_su(len(idx_s), 0.601, 0.3624, 223, 0.4938)
            
        idx_l = np.where(~is_small)[0]
        if len(idx_l) > 0:
            b_par, loc_par, scale_par = 1.633, -73.46, 316.5
            u = np.random.rand(len(idx_l))
            sizes[idx_l] = loc_par + scale_par / (u**(1/b_par))
            
        data['sizes'] = np.maximum(sizes, 60)
        
    return data


def simulacion_mempool(env, params, stats):
    """
    Simulación de la dinámica de la mempool con procesamiento por lotes.
    
    Args:
        env: Entorno SimPy
        params: Diccionario con Lambda, Escenario, Capacidad
        stats: Diccionario para almacenar resultados
    """
    n_gen = 500000
    trafico = generar_trafico(n_gen, params['Escenario'], params['Lambda'])
    
    interarribos = trafico['inter']
    fees = trafico['fees']
    sizes = trafico['sizes']
    
    avg_size = np.mean(sizes)
    lambda_efectiva = 1.0 / np.mean(interarribos)
    txs_esperadas_600s = lambda_efectiva * 600.0
    capacidad_bloque_txs = params['Capacidad'] / avg_size
    rho_esperado = txs_esperadas_600s / capacidad_bloque_txs
    
    print(f"  λ_eff={lambda_efectiva:.4f} txs/s, E[txs/bloque]={txs_esperadas_600s:.1f}, Cap={capacidad_bloque_txs:.1f} txs, ρ={rho_esperado:.4f}")
    
    if rho_esperado > 2.0:
        print(f"  [ABORTANDO] Rho = {rho_esperado:.4f} > 2.0 (sobresaturación crítica)")
        return
    
    if rho_esperado < 0.1:
        print(f"  [ADVERTENCIA] Rho = {rho_esperado:.4f} < 0.1 (subsaturado)")
    elif rho_esperado > 1.5:
        print(f"  [ADVERTENCIA] Rho = {rho_esperado:.4f} > 1.5 (alta saturación)")
    
    warmup_time = 18000
    stop_time = 90000
    block_interval = 600
    
    mempool = {}
    next_tx_id = 0
    
    stats['total_bloques_minados'] = 0
    stats['total_txs_procesadas'] = 0
    
    def llegadas():
        nonlocal next_tx_id
        for i in range(len(interarribos)):
            yield env.timeout(interarribos[i])
            
            if env.now >= stop_time:
                break
            
            mempool[next_tx_id] = {
                'arrival_time': env.now,
                'fee': fees[i],
                'size': sizes[i]
            }
            next_tx_id += 1
    
    def minado():
        bloque_num = 0
        while True:
            yield env.timeout(block_interval)
            bloque_num += 1
            
            if env.now >= stop_time:
                break
            
            if len(mempool) == 0:
                continue
            
            txs_ordenadas = sorted(
                [(tx_id, tx['fee'], tx['size'], tx['arrival_time']) 
                 for tx_id, tx in mempool.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            bloque_bytes = 0
            txs_a_procesar = []
            
            for tx_id, fee, size, arrival_time in txs_ordenadas:
                if bloque_bytes + size <= params['Capacidad']:
                    bloque_bytes += size
                    wait_time = env.now - arrival_time
                    
                    if env.now > warmup_time:
                        stats['waits'].append(wait_time)
                        stats['total_txs_procesadas'] += 1
                    
                    txs_a_procesar.append(tx_id)
            
            for tx_id in txs_a_procesar:
                del mempool[tx_id]
            
            if env.now > warmup_time:
                stats['total_bloques_minados'] += 1
    
    def monitor_cola():
        while True:
            yield env.timeout(60)
            if env.now > warmup_time:
                stats['queue_len'].append(len(mempool))
    
    env.process(llegadas())
    env.process(minado())
    env.process(monitor_cola())
    
    env.run(until=stop_time)


# === Ejecución del Experimento Factorial ===

lambdas = [0.0005, 0.0015, 0.004]
escenarios = [1, 2]
capacidades = [0.5e6, 1.0e6, 2.0e6]
replicas = 10

total_runs = len(lambdas) * len(escenarios) * len(capacidades) * replicas
resultados_finales = []
counter = 0

SEMILLA_BASE = 42
print(f"Semilla aleatoria base: {SEMILLA_BASE}")
print(f"Iniciando {total_runs} simulaciones...")
print(f"{'='*80}\n")

for l in lambdas:
    for e in escenarios:
        for c in capacidades:
            for r in range(replicas):
                counter += 1
                
                semilla_replica = SEMILLA_BASE + counter
                np.random.seed(semilla_replica)
                random.seed(semilla_replica)
                
                stats = {
                    'waits': [], 
                    'queue_len': [],
                    'total_bloques_minados': 0,
                    'total_txs_procesadas': 0
                }
                params = {'Lambda': l, 'Escenario': e, 'Capacidad': c, 'Replica': r+1}
                
                if counter == 1 or counter % 10 == 0 or counter == total_runs:
                    print(f"[{counter}/{total_runs}] λ={l}, Esc={e}, Cap={c/1e6:.1f}MB, Rep={r+1}")
                
                env = simpy.Environment()
                
                try:
                    simulacion_mempool(env, params, stats)
                except Exception as e:
                    print(f"  [ERROR] Simulación falló: {e}")
                
                w_avg = np.mean(stats['waits']) if stats['waits'] else 0
                w_max = np.max(stats['waits']) if stats['waits'] else 0
                w_p95 = np.percentile(stats['waits'], 95) if stats['waits'] else 0
                
                lq_avg = np.mean(stats['queue_len']) if stats['queue_len'] else 0
                lq_max = np.max(stats['queue_len']) if stats['queue_len'] else 0
                
                time_window = 90000 - 18000
                n_txs = stats['total_txs_procesadas']
                throughput = n_txs / time_window if time_window > 0 else 0
                
                resultados_finales.append({
                    'Lambda_Factor': l,
                    'Escenario': e,
                    'Capacidad_Bytes': c,
                    'Replica': r+1,
                    'W_Tiempo_Esp_Promedio': w_avg,
                    'W_Tiempo_Esp_Maximo': w_max,
                    'W_Tiempo_Esp_P95': w_p95,
                    'Lq_Cola_Promedio': lq_avg,
                    'Lq_Cola_Maximo': lq_max,
                    'Throughput': throughput,
                    'N_Transacciones': n_txs,
                    'N_Bloques_Minados': stats['total_bloques_minados']
                })


df = pd.DataFrame(resultados_finales)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
nombre_archivo = f'Resultados_Simulacion_{timestamp}.csv'
df.to_csv(nombre_archivo, index=False)

print(f"\n{'='*80}")
print(f"¡SIMULACIÓN COMPLETADA!")
print(f"{'='*80}")
print(f"\nArchivo generado: '{nombre_archivo}'")
print(f"Total de corridas: {len(df)}")

print(f"\n{'='*80}")
print(f"RESUMEN EJECUTIVO DE RESULTADOS")
print(f"{'='*80}")

print("\n--- MÉTRICAS GLOBALES ---")
print(f"Tiempo de espera promedio:   {df['W_Tiempo_Esp_Promedio'].mean():.2f} seg (±{df['W_Tiempo_Esp_Promedio'].std():.2f})")
print(f"Tiempo de espera máximo:     {df['W_Tiempo_Esp_Maximo'].max():.2f} seg")
print(f"Longitud de cola promedio:   {df['Lq_Cola_Promedio'].mean():.2f} txs (±{df['Lq_Cola_Promedio'].std():.2f})")
print(f"Longitud de cola máxima:     {df['Lq_Cola_Maximo'].max():.0f} txs")
print(f"Throughput promedio:         {df['Throughput'].mean():.4f} txs/seg")

print("\n--- POR FACTOR LAMBDA ---")
for factor in sorted(df['Lambda_Factor'].unique()):
    subset = df[df['Lambda_Factor'] == factor]
    print(f"\nFactor λ = {factor}:")
    print(f"  W promedio: {subset['W_Tiempo_Esp_Promedio'].mean():.2f} seg")
    print(f"  Lq promedio: {subset['Lq_Cola_Promedio'].mean():.2f} txs")
    print(f"  Throughput: {subset['Throughput'].mean():.4f} txs/seg")

print("\n--- POR ESCENARIO ---")
for esc in sorted(df['Escenario'].unique()):
    subset = df[df['Escenario'] == esc]
    nombre_esc = "Baja Congestión" if esc == 1 else "Alta Congestión"
    print(f"\nEscenario {esc} ({nombre_esc}):")
    print(f"  W promedio: {subset['W_Tiempo_Esp_Promedio'].mean():.2f} seg")
    print(f"  Lq promedio: {subset['Lq_Cola_Promedio'].mean():.2f} txs")
    print(f"  Throughput: {subset['Throughput'].mean():.4f} txs/seg")

print("\n--- POR CAPACIDAD ---")
for cap in sorted(df['Capacidad_Bytes'].unique()):
    subset = df[df['Capacidad_Bytes'] == cap]
    print(f"\nCapacidad {cap/1e6:.1f} MB:")
    print(f"  W promedio: {subset['W_Tiempo_Esp_Promedio'].mean():.2f} seg")
    print(f"  Lq promedio: {subset['Lq_Cola_Promedio'].mean():.2f} txs")
    print(f"  Throughput: {subset['Throughput'].mean():.4f} txs/seg")

print("\n--- CONFIGURACIONES EXTREMAS ---")
idx_max_w = df['W_Tiempo_Esp_Promedio'].idxmax()
print(f"\nMayor tiempo de espera:")
print(f"  λ={df.loc[idx_max_w, 'Lambda_Factor']}, Escenario={df.loc[idx_max_w, 'Escenario']}, " +
      f"Cap={df.loc[idx_max_w, 'Capacidad_Bytes']/1e6:.1f}MB")
print(f"  W = {df.loc[idx_max_w, 'W_Tiempo_Esp_Promedio']:.2f} seg, Lq = {df.loc[idx_max_w, 'Lq_Cola_Promedio']:.2f} txs")

idx_max_lq = df['Lq_Cola_Maximo'].idxmax()
print(f"\nMayor longitud de cola:")
print(f"  λ={df.loc[idx_max_lq, 'Lambda_Factor']}, Escenario={df.loc[idx_max_lq, 'Escenario']}, " +
      f"Cap={df.loc[idx_max_lq, 'Capacidad_Bytes']/1e6:.1f}MB")
print(f"  Lq_max = {df.loc[idx_max_lq, 'Lq_Cola_Maximo']:.0f} txs")

print(f"\n{'='*80}")
print("Simulación completada exitosamente.")
print(f"{'='*80}\n")
