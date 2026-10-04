import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import root
import streamlit as st
import sympy as sp

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption(
    "Secuenciación automática, cálculo de Subsistemas/Iteración, Memoria de"
    " Cálculo y resolución numérica interactiva."
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
        # En lugar de fallar si no es estrictamente cuadrado, agrupamos el bloque residual completo
        comp_eqs = [eq[0] for eq in remaining_eqs]
        comp_vars_set = set()
        for _, eq in remaining_eqs:
          for v in eq.free_symbols:
            if v not in simulated_subs:
              comp_vars_set.add(v.name)
        comp_vars = list(comp_vars_set)

        if comp_eqs and comp_vars:
          eq_names = ", ".join(comp_eqs[:5]) + (
              "..." if len(comp_eqs) > 5 else ""
          )
          var_names = ", ".join(comp_vars[:5]) + (
              "..." if len(comp_vars) > 5 else ""
          )

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
    with st.spinner(f"Calculando valores usando {metodo_elegido}..."):
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

          # Estimación inicial inteligente (evita ceros en divisiones y respeta fracciones)
          x0 = [0.2 if "x" in s.name else 50.0 for s in vars_to_solve]
          sol = root(sistema_residual, x0, method="hybr", options={"maxiter": 1000})

          if sol.success:
            for i, sym in enumerate(vars_to_solve):
              resultados_reales[sym] = sol.x[i]
            solve_success = True
            st.success("¡Sistema resuelto numéricamente con éxito!")
          else:
            st.warning(
                "El método numérico no convergió por completo con la"
                " estimación inicial actual. Intenta ajustar los valores"
                " conocidos (Datos)."
            )
        except Exception as e:
          st.error(f"Error crítico en la ejecución numérica: {str(e)}")

  # TABS DE VISUALIZACIÓN
  tab_seq, tab_reporte, tab_memoria, tab_res = st.tabs([
      "🔄 Matrices",
      "📄 Reportes POE",
      "📝 Memoria",
      "🚀 Resultados numéricos",
  ])

  with tab_seq:
    col1, col2 = st.columns(2)
    with col1:
      st.subheader("Secuencia por Iteración")
      if seq_iter:
        st.dataframe(
            pd.DataFrame(seq_iter)[["Bloque", "Resuelve", "Tipo"]],
            use_container_width=True,
        )
    with col2:
      st.subheader("Secuencia por Subsistemas")
      if seq_sub:
        st.dataframe(
            pd.DataFrame(seq_sub)[["Bloque", "Resuelve", "Tipo"]],
            use_container_width=True,
        )

  with tab_reporte:
    col1, col2 = st.columns(2)
    with col1:
      poe_iter = "\n\n".join([step["POE_Format"] for step in seq_iter])
      st.text_area("Copia formato Iteración:", value=poe_iter, height=400)
    with col2:
      poe_sub = "\n\n".join([step["POE_Format"] for step in seq_sub])
      st.text_area("Copia formato Subsistemas:", value=poe_sub, height=400)

  with tab_memoria:
    col1, col2 = st.columns(2)
    with col1:
      for linea in mem_iter:
        st.markdown(linea)
    with col2:
      for linea in mem_sub:
        st.markdown(linea)

  with tab_res:
    st.subheader("Valores Numéricos de las Variables")
    results = []
    for sym in all_symbols:
      val = resultados_reales.get(sym, "Pendiente de ejecutar")
      is_specified = sym in known_vars
      val_formatted = (
          f"{float(val):.4f}"
          if isinstance(val, (int, float, sp.Float, np.number))
          else str(val)
      )
      results.append({
          "Variable": sym.name,
          "Estado": "Dato" if is_specified else "Calculada",
          "Valor": val_formatted,
      })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
