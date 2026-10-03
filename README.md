# Programa Ordenador de Ecuaciones (POE) - Clone Interactivo

Aplicación interactiva desarrollada en **Python** y **Streamlit** para la simulación, estructuración y resolución secuencial de balances de materia y energía en ingeniería de procesos.

## 🌟 Características
- **Parsing Dinámico de Ecuaciones:** Lectura e interpretación en tiempo real mediante `SymPy`.
- **Análisis de Grados de Libertad (GL):** Cálculo automático de $GL = N_{incógnitas} - N_{ecuaciones}$.
- **Matriz de Incidencia:** Representación tabular del acoplamiento entre ecuaciones y variables libres.
- **Secuenciación de Cálculo:** Algoritmo de reducción acíclica basado en la lógica de Lee-Christensen-Rudd / Steward.
- **Exportación:** Descarga inmediata del reporte de resultados en formato CSV.

## 🚀 Instalación y Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/poe-solver.git](https://github.com/TU_USUARIO/poe-solver.git)
   cd poe-solver
