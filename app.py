import streamlit as st
import sympy as sp
import pandas as pd

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption("Análisis de grados de libertad, secuenciación automática, detección de variables de corte y resolución de balances.")

# 1. ENTRADA DE ECUACIONES
st.sidebar.header("1. Definición del Sistema")
default_eqs = (
    "F1 + F2 = F3\n"
    "F1 * 0.1 + F2 * 0.4 = F3 * x3\n"
    "x3 + y3 = 1\n"
    "F1 = 100"
)
raw_eqs = st.sidebar.text_area(
    "Ingrese las ecuaciones (una por línea):",
    value=default_eqs,
    height=180
)

# Parsing de ecuaciones
eq_lines = [line.strip() for line in raw_eqs.split("\n") if line.strip()]
parsed_eqs = []
parse_errors = []

for i, line in enumerate(eq_lines):
    try:
        if "=" in line:
            parts = line.split("=")
            if len(parts) == 2:
                lhs = sp.sympify(parts[0])
                rhs = sp.sympify(parts[1])
                parsed_eqs.append(sp.Eq(lhs, rhs))
            else:
                parse_errors.append(f"Línea {i+1}: Múltiples signos '=' detectados.")
        else:
            expr = sp.sympify(line)
            parsed_eqs.append(sp.Eq(expr, 0))
    except Exception as e:
        parse_errors.append(f"Línea {i+1} ('{line}'): Error de sintaxis - {str(e)}")

if parse_errors:
    for err in parse_errors:
        st.error(err)

if parsed_eqs:
    # Extracción de variables únicas
    all_symbols = sorted(list(set().union(*[eq.free_symbols for eq in parsed_eqs])), key=lambda s: s.name)
    var_names = [s.name for s in all_symbols]

    # 2. ESPECIFICACIÓN DE VARIABLES
    st.sidebar.header("2. Especificación de Datos")
    known_selected = st.sidebar.multiselect("Seleccione variables conocidas (Datos):", var_names)
    
    known_vars = {}
    for var in known_selected:
        val = st.sidebar.number_input(f"Valor para {var}:", value=1.0000, step=0.1, format="%.4f")
        known_vars[sp.Symbol(var)] = val

    # 3. ANÁLISIS DE GRADOS DE LIBERTAD (GL)
    total_vars = len(all_symbols)
    num_specified = len(known_vars)
    unknown_vars = [s for s in all_symbols if s not in known_vars]
    num_unknowns = len(unknown_vars)
    num_eqs = len(parsed_eqs)
    dof = num_unknowns - num_eqs

    st.subheader("📌 Análisis del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables Totales", total_vars)
    c2.metric("Variables Especificadas", num_specified)
    c3.metric("Incógnitas", num_unknowns)
    c4.metric("Grados de Libertad (GL)", dof)

    # Diagnóstico GL
    if dof > 0:
        st.warning(f"⚠️ **Sistema Subdeterminado (GL = {dof})**: Debe fijar {dof} variable(s) adicional(es).")
    elif dof < 0:
        st.error(f"❌ **Sistema Sobredeterminado (GL = {dof})**: Hay {-dof} ecuación(es) de más o redundantes.")
    else:
        st.success("✅ **Sistema Determinado (GL = 0)**: Listo para ordenar y resolver.")

    st.divider()

    # 4. MATRIZ DE INCIDENCIA
    st.subheader("📊 Matriz de Incidencia (Ecuaciones vs. Incógnitas Libres)")
    inc_data = []
    for i, eq in enumerate(parsed_eqs):
        row = {"Ecuación": f"Eq {i+1}"}
        eq_vars = eq.free_symbols
        for sym in unknown_vars:
            row[sym.name] = 1 if sym in eq_vars else 0
        inc_data.append(row)

    df_inc = pd.DataFrame(inc_data).set_index("Ecuación")
    st.dataframe(df_inc, use_container_width=True)

    # 5. SECUENCIACIÓN Y RESOLUCIÓN CON DETECCIÓN DE CORTE
    tab_seq, tab_res = st.tabs(["🔄 Secuencia de Cálculo y Rasgado", "🚀 Ejecutar y Exportar"])

    substitutions = known_vars.copy()
    remaining_eqs = parsed_eqs.copy()
    sequence_steps = []
    step_num = 1
    stalled = False

    # Paso secuencial acyclico (Lee-Christensen-Rudd / Steward)
    while remaining_eqs:
        found_step = False
        for idx, eq in list(enumerate(remaining_eqs)):
            eq_sub = eq.subs(substitutions)
            free_vars = eq_sub.free_symbols
            
            if len(free_vars) == 1:
                target_var = list(free_vars)[0]
                sol = sp.solve(eq_sub, target_var)
                val = sol[0].evalf() if sol else sp.nan
                
                sequence_steps.append({
                    "Paso": step_num,
                    "Tipo": "Secuencial",
                    "Ecuación Evaluada": f"Eq {parsed_eqs.index(eq)+1}",
                    "Variable Resuelta": target_var.name,
                    "Valor Calculado": f"{float(val):.4f}" if isinstance(val, (int, float, sp.Float)) else str(val)
                })
                
                substitutions[target_var] = val
                remaining_eqs.pop(idx)
                found_step = True
                step_num += 1
                break
            elif len(free_vars) == 0:
                remaining_eqs.pop(idx)
                found_step = True
                break

        if not found_step and remaining_eqs:
            stalled = True
            break

    with tab_seq:
        st.subheader("Secuencia de Resolución Determinada")
        if sequence_steps:
            st.table(pd.DataFrame(sequence_steps))

        if stalled:
            # Análisis del subsistema acoplado
            sub_remaining_eqs = [eq.subs(substitutions) for eq in remaining_eqs]
            rem_vars = sorted(list(set().union(*[eq.free_symbols for eq in sub_remaining_eqs])), key=lambda s: s.name)
            
            # Ranking de candidatos a variable de corte (frecuencia en el subsistema)
            var_counts = {v.name: sum(1 for eq in sub_remaining_eqs if v in eq.free_symbols) for v in rem_vars}
            sorted_candidates = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)

            st.warning(f"⚠️ **Ciclo Detectado / Sistema Acoplado**: Quedan {len(sub_remaining_eqs)} ecuaciones simultáneas y {len(rem_vars)} incógnitas por resolver.")
            
            st.markdown("### ✂️ Candidatas Recomendadas para Variable de Corte (Rasgado)")
            st.write("Si deseas romper el ciclo manualmente asignando un valor estimado, las mejores opciones (mayor impacto de desacoplamiento) son:")
            
            df_cand = pd.DataFrame(sorted_candidates, columns=["Variable Recomendada de Corte", "Frecuencia en Subsistema Acoplado"])
            st.dataframe(df_cand, use_container_width=True)

            # Resolución simultánea automática del subsistema acoplado
            st.markdown("### ⚡ Resolución Simultánea del Subsistema Acoplado")
            try:
                simultaneous_sols = sp.solve(sub_remaining_eqs, rem_vars, dict=True)
                if simultaneous_sols:
                    sol_dict = simultaneous_sols[0]
                    sim_steps = []
                    for var_sym, val_expr in sol_dict.items():
                        val_eval = val_expr.evalf() if hasattr(val_expr, 'evalf') else val_expr
                        substitutions[var_sym] = val_eval
                        sim_steps.append({
                            "Paso": f"Simultáneo {step_num}",
                            "Tipo": "Acoplado (Simultáneo)",
                            "Ecuación Evaluada": "Subsistema Completo",
                            "Variable Resuelta": var_sym.name,
                            "Valor Calculado": f"{float(val_eval):.4f}" if isinstance(val_eval, (int, float, sp.Float)) else str(val_eval)
                        })
                        step_num += 1
                    st.success("✅ **El subsistema acoplado fue resuelto simultáneamente con éxito.**")
                    st.table(pd.DataFrame(sim_steps))
                else:
                    st.error("No se pudo resolver simbólicamente el subsistema. Selecciona una de las variables recomendadas e ingresa un valor hipotético en el menú lateral.")
            except Exception as e:
                st.error(f"Error al resolver el subsistema simultáneo: {str(e)}")

    with tab_res:
        st.subheader("Resultados Consolidados del Balance")
        
        results = []
        for sym in all_symbols:
            val = substitutions.get(sym, "No resuelto")
            is_specified = sym in known_vars
            
            if isinstance(val, (int, float, sp.Float)):
                val_formatted = f"{float(val):.4f}"
            else:
                val_formatted = str(val)

            results.append({
                "Variable": sym.name,
                "Estado": "Especificada (Dato)" if is_specified else ("Calculada" if sym in substitutions else "Incierta"),
                "Valor": val_formatted
            })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Resultados en CSV",
            data=csv_data,
            file_name="reporte_poe_resultados.csv",
            mime="text/csv"
        )
