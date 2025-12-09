# Simulación de la Dinámica de una Mempool de Blockchain

## Descripción

Investigación académica sobre la modelización y simulación de eventos discretos aplicada a la mempool de Bitcoin. Este trabajo implementa un modelo completo desde la captura de datos empíricos hasta el análisis estadístico de resultados mediante diseño experimental factorial.

**Paper:** *Simulación de la Dinámica de una Mempool de Blockchain: Un Enfoque Basado en Eventos Discretos*

**Autor:** Simón Tadeo Ocampo  
**Institución:** Universidad Tecnológica Nacional - Facultad Regional La Plata  
**Materia:** Simulación  
**Año:** 2025

---

## 🎯 Objetivos del Estudio

- Modelar el proceso de llegada de transacciones mediante ajuste robusto de distribuciones empíricas
- Implementar un modelo de simulación de eventos discretos de la mempool con procesamiento por lotes
- Evaluar el impacto de factores operativos (tasa de llegada, distribución de comisiones, capacidad del bloque) sobre métricas de rendimiento
- Identificar condiciones que conducen a congestión de red

---

## 📂 Estructura del Proyecto

```
TP-Final-Simulacion/
│
├── 📄 README.md                                    # Este archivo
├── 📄 requirements.txt                             # Dependencias Python
├── 📊 Simulacion de la Dinamica de una Mempool de Blockchain - Simon Ocampo.pdf
├── 🎥 Video Presentación Mempool Blockchain - STO.mp4
├── 📊 Presentacion TP Final Simulacion - STO.pdf
│
├── 📂 Datos empíricos (raíz del proyecto)
│   ├── mempool_capture.py                          # Script de captura WebSocket
│   ├── mempool_data_final_20250919_054047_low.csv  # Dataset baja congestión
│   └── mempool_data_final_20250922_110158_high.csv # Dataset alta congestión
│
├── 📂 Ajuste de distribuciones (raíz del proyecto)
│   ├── fitter_code.py                              # Framework robusto completo
│   ├── quick_core_distributions_plots.py           # Generador de figuras base
│   ├── quick_mixture_qq_size.py                    # Q-Q plots de mezclas
│   └── quick_segmented_density_size.py             # Densidades segmentadas
│
└── 📂 Modelo de simulación (raíz del proyecto)
    ├── modelo_sim.py                               # Simulación SimPy (180 corridas)
    ├── anova_completo.py                           # Análisis ANOVA factorial
    └── Resultados_Simulacion_Corregido_20251207_202227.csv
```

**Nota:** Por simplicidad, todos los archivos de código están en la raíz del repositorio para acceso directo.

---

## 🛠️ Herramientas y Tecnologías

### Captura de Datos
- **Python 3.10** - Lenguaje principal
- **websockets** - Conexión en tiempo real con blockchain.info
- **asyncio** - Manejo asíncrono de eventos

### Ajuste de Distribuciones
- **NumPy** - Computación numérica
- **SciPy** - Ajuste de distribuciones estadísticas
- **pandas** - Manipulación de datos
- **sklearn** - Validación cruzada k-fold
- **matplotlib** - Visualización

**Framework desarrollado:**
- Validación cruzada 3-fold para generalización
- Bootstrap (1000 iteraciones) para estabilidad paramétrica
- Criterios de información múltiples (BIC, AIC, HQIC)
- Detección automática de overfitting
- Segmentación discreta para variables heterogéneas

### Modelado y Simulación
- **SimPy** - Framework de simulación de eventos discretos
- **NumPy** - Generación de variables aleatorias
- Implementación de:
  - Proceso de llegadas (Inverse Gamma)
  - Distribuciones fee rate (Johnson SU)
  - Distribuciones de tamaños (Johnson SU / Pareto segmentado)
  - Cola de prioridad por comisión
  - Procesamiento por lotes cada 600 segundos

### Análisis Estadístico
- **SciPy.stats** - ANOVA factorial, tests post-hoc
- **pandas** - Manipulación de resultados
- Diseño experimental: 3×2×3 con 10 réplicas (180 corridas)
- Cálculo de tamaños de efecto (η²)
- Test de Tukey para comparaciones múltiples

### Documentación
- **LaTeX** - Redacción científica con template `joas`
- **BibLaTeX** - Gestión de referencias bibliográficas
- **TikZ/PGFPlots** - Diagramas técnicos

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/SimonOcampo1/TP-Final-Simulacion.git
cd TP-Final-Simulacion
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Capturar datos empíricos (opcional)
```bash
cd datos_empiricos
python mempool_capture.py 1.0  # Captura por 1 hora
```

### 4. Ajustar distribuciones
```bash
cd ajuste_distribuciones
python fitter_code.py --csv ../datos_empiricos/mempool_data_low.csv --column size_bytes --segment
python quick_core_distributions.py  # Generar figuras
```

### 5. Ejecutar simulación
```bash
cd modelo_simulacion
python modelo_sim.py  # 180 corridas, ~2-3 horas
```

### 6. Analizar resultados
```bash
python anova_completo.py
python analisis_final.py
```

---

## 📊 Resultados Principales

- **13.7M transacciones** procesadas en total
- **180 configuraciones** experimentales evaluadas
- **Rango de ρ:** 0.0001 a 2.3 (subsaturado → sobresaturado)
- **Tiempo de espera promedio:** 107 minutos (máx: 12.9 horas)
- **Cola promedio:** 13,996 transacciones (máx: 300,928)

### Efectos Estadísticos (ANOVA)
- **Lambda (tasa de llegada):** F=11.32, p<0.001, η²=0.140
- **Escenario (congestión):** F=29.15, p<0.001, η²=0.172
- **Capacidad (bloque):** F=16.29, p<0.001, η²=0.190

**Todos los factores altamente significativos** explicando el 50.2% de la varianza total.

---

## 📄 Citas y Referencias

Si utilizas este trabajo en tu investigación, por favor cita:

```bibtex
@techreport{ocampo2025mempool,
  author = {Ocampo, Simón Tadeo},
  title = {Simulación de la Dinámica de una Mempool de Blockchain: Un Enfoque Basado en Eventos Discretos},
  institution = {Universidad Tecnológica Nacional - Facultad Regional La Plata},
  year = {2025},
  type = {Trabajo Final de Simulación}
}
```

---

## 📧 Contacto

**Simón Tadeo Ocampo**  
📧 simontadeoocampo@alu.frlp.utn.edu.ar  
🔗 [GitHub](https://github.com/SimonOcampo1)

---

## 📝 Licencia

Este proyecto es material académico desarrollado para fines educativos en el marco de la materia Simulación (UTN FRLP).

---

## 🙏 Agradecimientos

- **Prof. Francisco Roqué** - Guía y orientación metodológica
- **Blockchain.com** - API pública de datos de mempool
- **Comunidad Python científico** - NumPy, SciPy, SimPy
