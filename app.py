import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import streamlit as st
import sympy as sp

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption(
    "Secuenciación automática, cálculo de Subsistemas/Iteración, Memoria de "
    "Cálculo y resolución numérica interactiva."
)

# 1. ENTRADA DE ECUACIONES
st.sidebar.header("1. Definición del Sistema")
default_eqs = (
    "F1 + F2 = F3\n"
    "F1 * 0.1 + F2 * 0.4 = F3 * x3\n"
    "x3 + y3 = 1\n"
    "F1 = 100"
)
raw_eqs = st.sidebar.text_area(
    "Ingrese las ecuaciones (una por línea):", value=default_eqs, height=250
)

def parse_equations(raw_text):
    eq_lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    parsed = []
    errors = []
    for i, line in enumerate(eq_lines):
        eq_name = f"Ec{i+1}"
        try:
            if "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    lhs = sp.sympify(parts[0])
                    rhs = sp.sympify(parts[1])
                    parsed.append((eq_name, sp.Eq(lhs, rhs)))
                else:
                    errors.append(f"Línea {i+1} ({eq_name}): Múltiples signos '='.")
            else:
                expr = sp.sympify(line)
                parsed.append((eq_name, sp.Eq(expr, 0)))
        except Exception as e:
            errors.append(f"Línea {i+1} ({eq_name}): Error de sintaxis - {str(e)}")
    return parsed, errors

parsed_eqs, parse_errors = parse_equations(raw_eqs)

if parse_errors:
    for err in parse_errors:
        st.error(err)

if parsed_eqs:
    all_symbols_set = set()
    for _, eq in parsed_eqs:
        all_symbols_set.update(eq.free_symbols)
    all_symbols = sorted(list(all_symbols_set), key=lambda s: s.name)
    var_names = [s.name for s in all_symbols]

    st.sidebar.header("2. Especificación de Datos")
    known_selected = st.sidebar.multiselect(
        "Seleccione variables conocidas (Datos):", var_names
    )

    known_vars = {}
    for var in known_selected:
        val = st.sidebar.number_input(
            f"Valor para {var}:", value=1.0000, step=0.1, format="%.6f"
        )
        known_vars[sp.Symbol(var)] = val

    # ANÁLISIS DE GRADOS DE LIBERTAD (GL)
    total_vars = len(all_symbols)
    num_specified = len(known_vars)
    unknown_vars = [s for s in all_symbols if s not in known_vars]
    num_unknowns = len(unknown_vars)
    num_eqs = len(parsed_eqs)
    dof = num_unknowns - num_eqs

    st.subheader("📌 Análisis del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables Totales", total_vars)
    c2.metric("Datos", num_specified)
    c3.metric("Incógnitas Libres", num_unknowns)
    c4.metric("Grados de Libertad (GL)", dof)

    if dof > 0:
        st.warning(f"⚠ **Sistema Subdeterminado (GL = {dof})**")
    elif dof < 0:
        st.error(f"❌ **Sistema Sobredeterminado (GL = {dof})**")
    else:
        st.success("✅ **Sistema Determinado (GL = 0)**")

    # =========================================================================
    # MOTORES DE SECUENCIACIÓN ROBUSTOS
    # =========================================================================
    def engine_iteracion(parsed, known_dict):
        simulated_subs = known_dict.copy()
        remaining_eqs = parsed.copy()
        seq_steps = []
        memoria = []
        tear_counter, ver_counter, step = 0, 0, 1

        while remaining_eqs:
            found_step = False
            for idx, (eq_name, eq) in enumerate(remaining_eqs):
                free_vars = [s for s in eq.free_symbols if s not in simulated_subs]
                if len(free_vars) == 1:
                    target_var = free_vars[0]
                    seq_steps.append({
                        "Bloque": eq_name,
                        "Resuelve": target_var.name,
                        "Tipo": "Secuencial",
                        "POE_Format": f"{eq_name}\n1 {target_var.name}",
                    })
                    memoria.append(
                        f"**Paso {step}:** De `{eq_name}` se despeja `{target_var.name}`."
                    )
                    simulated_subs[target_var] = 1.0
                    remaining_eqs.pop(idx)
                    found_step = True
                    step += 1
                    break
                elif len(free_vars) == 0:
                    seq_steps.append({
                        "Bloque": eq_name,
                        "Resuelve": f"Ver{ver_counter}",
                        "Tipo": "Verificación",
                        "POE_Format": f"{eq_name}\n1 Ver{ver_counter}",
                    })
                    memoria.append(f"**Paso {step}:** `{eq_name}` actúa como Verificadora.")
                    ver_counter += 1
                    remaining_eqs.pop(idx)
                    found_step = True
                    step += 1
                    break

            if not found_step and remaining_eqs:
                rem_vars_counts = {}
                for _, eq in remaining_eqs:
                    for v in eq.free_symbols:
                        if v not in simulated_subs:
                            rem_vars_counts[v] = rem_vars_counts.get(v, 0) + 1
                if rem_vars_counts:
                    tear_var = max(rem_vars_counts, key=rem_vars_counts.get)
                    seq_steps.append({
                        "Bloque": f"ItVer{tear_counter}",
                        "Resuelve": tear_var.name,
                        "Tipo": "Corte (ItVer)",
                        "POE_Format": f"ItVer{tear_counter}\n1 {tear_var.name}",
                    })
                    memoria.append(
                        f"**Paso {step}:** Ciclo detectado. Corte en `{tear_var.name}`"
                        f" (`ItVer{tear_counter}`)."
                    )
                    simulated_subs[tear_var] = 1.0
                    tear_counter += 1
                    step += 1
                else:
                    break
        return seq_steps, memoria, tear_counter

    def engine_subsistemas(parsed, known_dict):
        simulated_subs = known_dict.copy()
        remaining_eqs = parsed.copy()
        seq_steps = []
        memoria = []
        step, sub_tear_counter = 1, 0

        while remaining_eqs:
            found_step = False
            for idx, (eq_name, eq) in enumerate(remaining_eqs):
                free_vars = [s for s in eq.free_symbols if s not in simulated_subs]
                if len(free_vars) == 1:
                    target_var = free_vars[0]
                    seq_steps.append({
                        "Bloque": eq_name,
                        "Resuelve": target_var.name,
                        "Tipo": "Secuencial",
                        "POE_Format": f"{eq_name}\n1 {target_var.name}",
                    })
                    memoria.append(
                        f"**Paso {step}:** Resolución directa de `{eq_name}` para"
                        f" `{target_var.name}`."
                    )
                    simulated_subs[target_var] = 1.0
                    remaining_eqs.pop(idx)
                    found_step = True
                    step += 1
                    break

            if not found_step and remaining_eqs:
                comp_eqs = [eq[0] for eq in remaining_eqs]
                comp_vars_set = set()
                for _, eq in remaining_eqs:
                    for v in eq.free_symbols:
                        if v not in simulated_subs:
                            comp_vars_set.add(v.name)
                comp_vars = list(comp_vars_set)

                if comp_eqs and comp_vars:
                    seq_steps.append({
                        "Bloque": f"Subsistema Global ({len(comp_eqs)} eqs)",
                        "Resuelve": f"{len(comp_vars)} variables acopladas",
                        "Tipo": "Subsistema / Lazo de Recirculación",
                        "POE_Format": f"Subsistema_{sub_tear_counter}\nEqs: {len(comp_eqs)}",
                    })
                    memoria.append(
                        f"**Paso {step}:** Bloque acoplado masivo detectado con"
                        f" {len(comp_eqs)} ecuaciones y {len(comp_vars)} incógnitas."
                    )

                    for v_name in comp_vars:
                        simulated_subs[sp.Symbol(v_name)] = 1.0
                    break
                else:
                    break
        return seq_steps, memoria

    seq_iter, mem_iter, tear_count = engine_iteracion(parsed_eqs, known_vars)
    seq_sub, mem_sub = engine_subsistemas(parsed_eqs, known_vars)

    st.divider()

    # =========================================================================
    # SECCIÓN INTERACTIVA DE ELECCIÓN Y RESOLUCIÓN NUMÉRICA
    # =========================================================================
    st.subheader("🎯 Configuración de Resolución Numérica")
    metodo_elegido = st.selectbox(
        "¿Cómo deseas que el POE resuelva el sistema numéricamente a partir de la"
        " secuencia analizada?",
        ("Subsistemas (Bloque Acoplado)", "Iteración (Corte Secuencial)"),
    )

    ejecutar_solver = st.button("🚀 Ejecutar Resolución Numérica del Sistema")

    resultados_reales = known_vars.copy()
    solve_success = False

    if ejecutar_solver:
        with st.spinner(f"Calculando valores usando el método de {metodo_elegido} (con restricciones físicas)..."):
            eqs_for_solve = [eq[1].subs(known_vars) for eq in parsed_eqs]
            vars_to_solve = [s for s in all_symbols if s not in known_vars]

            if vars_to_solve:
                try:
                    exprs = [eq.lhs - eq.rhs for eq in eqs_for_solve]
                    f_lambdified = sp.lambdify(vars_to_solve, exprs, "numpy")

                    def sistema_residual(x_vals):
                        res = f_lambdified(*x_vals)
                        if isinstance(res, (int, float)):
                            return [res]
                        return np.array(res, dtype=float).flatten()

                    # 1. ESTABLECER LÍMITES FÍSICOS (BOUNDS) OPTIMIZADOS
                    bounds_lower = []
                    bounds_upper = []
                    x0 = []

                    for s in vars_to_solve:
                        nombre = s.name.lower()
                        # Si es fracción molar/másica (empieza con x, y, z)
                        if nombre.startswith('x') or nombre.startswith('y') or nombre.startswith('z'):
                            bounds_lower.append(0.0)       # Mínimo 0
                            bounds_upper.append(1.0)       # Máximo 1
                            x0.append(0.5)                 # Arranca en el medio
                        # Si es un caudal, masa, volumen o moles (f, m, w, v, n, l)
                        elif nombre.startswith('f') or nombre.startswith('m') or nombre.startswith('w') or nombre.startswith('v') or nombre.startswith('n') or nombre.startswith('l'):
                            bounds_lower.append(0.0)       # Ningún caudal puede ser negativo
                            bounds_upper.append(np.inf)    # Sin límite superior
                            x0.append(50.0)                # Estimación estándar
                        # Otras variables genéricas (Temperaturas, Entalpías, Calor, etc.)
                        else:
                            bounds_lower.append(-np.inf)   # Permite valores negativos
                            bounds_upper.append(np.inf)    # Sin límite superior
                            x0.append(10.0)

                    limites = (bounds_lower, bounds_upper)

                    # 2. RESOLVER CON LEAST SQUARES
                    sol = least_squares(
                        sistema_residual, 
                        x0, 
                        bounds=limites, 
                        method='trf', 
                        max_nfev=5000
                    )

                    # Tolerancia relajada a 1e-4 para permitir cierre en procesos industriales
                    if sol.cost < 1e-4:
                        for i, sym in enumerate(vars_to_solve):
                            resultados_reales[sym] = sol.x[i]
                        solve_success = True
                        st.success("¡Sistema resuelto con éxito respetando las restricciones físicas!")
                    else:
                        st.warning(
                            "El método convergió a un punto, pero hay un error residual alto. "
                            "Esto significa que los Datos ingresados podrían estar forzando un escenario "
                            "termodVeo que estás armando un secuenciador de ecuaciones (POE) en Streamlit con lógica de cortes y subsistemas, excelente para resolver balances de materia y flujos de procesos. 

Tu mensaje se cortó justo al final ("a ese codigo modificarle..."). **¿Qué cambio específico o nueva funcionalidad necesitas aplicarle?**

Para orientarnos, acá te dejo algunas cosas que podríamos ajustarle según el enfoque que le estés dando:

*   **Parseo y Termodinámica:** ¿Querés que el sistema reconozca variables de estado adicionales (como Temperatura o Presión) y les asigne límites (bounds) distintos a los de las fracciones molares ($x, y, z$) o caudales ($F$)?
*   **Gestión de Datos:** ¿Necesitás conectar el output numérico o la memoria de cálculo para guardarlos en una base de datos (como Firestore) o exportarlos como un reporte?
*   **Robustez del Solver:** ¿El `least_squares` se está quedando trabado en mínimos locales con algún sistema de ecuaciones fuertemente no lineal o iterativo?
*   **Interfaz de Usuario:** ¿Querés reestructurar las pestañas o agregar métricas de rendimiento del cálculo?

Comentame cuál es el objetivo o el error que te está arrojando y rearmamos esa parte del código.
