import streamlit as st
import sympy as sp
import pandas as pd

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption("Secuenciación automática con reporte nativo y resolución matemática exacta.")

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
    height=250
)

# Parsing de ecuaciones y etiquetado (Ec1, Ec2...)
eq_lines = [line.strip() for line in raw_eqs.split("\n") if line.strip()]
parsed_eqs = []
parse_errors = []

for i, line in enumerate(eq_lines):
    eq_name = f"Ec{i+1}"
    try:
        if "=" in line:
            parts = line.split("=")
            if len(parts) == 2:
                lhs = sp.sympify(parts[0])
                rhs = sp.sympify(parts[1])
                parsed_eqs.append((eq_name, sp.Eq(lhs, rhs)))
            else:
                parse_errors.append(f"Línea {i+1} ({eq_name}): Múltiples signos '=' detectados.")
        else:
            expr = sp.sympify(line)
            parsed_eqs.append((eq_name, sp.Eq(expr, 0)))
    except Exception as e:
        parse_errors.append(f"Línea {i+1} ({eq_name}) - '{line}': Error de sintaxis - {str(e)}")

if parse_errors:
    for err in parse_errors:
        st.error(err)

if parsed_eqs:
    # Extracción de variables únicas
    all_symbols = sorted(list(set().union(*[eq[1].free_symbols for eq in parsed_eqs])), key=lambda s: s.name)
    var_names = [s.name for s in all_symbols]

    # 2. ESPECIFICACIÓN DE DATOS
    st.sidebar.header("2. Especificación de Datos")
    known_selected = st.sidebar.multiselect("Seleccione variables conocidas (Datos):", var_names)
    
    known_vars = {}
    for var in known_selected:
        # Aumentamos la precisión a 6 decimales para balances rigurosos
        val = st.sidebar.number_input(f"Valor para {var}:", value=1.0000, step=0.1, format="%.6f")
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
    c2.metric("Datos", num_specified)
    c3.metric("Incógnitas Libres", num_unknowns)
    c4.metric("Grados de Libertad (GL)", dof)

    if dof > 0:
        st.warning(f"⚠️️ **Sistema Subdeterminado (GL = {dof})**: Faltan especificar {dof} variable(s).")
    elif dof < 0:
        st.error(f"❌ **Sistema Sobredeterminado (GL = {dof})**: Sobran {-dof} ecuación(es).")
    else:
        st.success("✅ **Sistema Determinado (GL = 0)**: Listo para resolver.")

    st.divider()

    tab_seq, tab_reporte, tab_res = st.tabs(["🔄 Matriz y Secuencia", "📄 Reporte Estilo POE", "🚀 Resultados"])

    # =========================================================================
    # MOTOR 1: GENERADOR DE SECUENCIA Y REPORTE POE (Rastreo)
    # =========================================================================
    simulated_subs = known_vars.copy()
    remaining_eqs_seq = parsed_eqs.copy()
    sequence_steps = []
    
    tear_counter = 0
    ver_counter = 0

    while remaining_eqs_seq:
        found_step = False
        
        for idx, (eq_name, eq) in list(enumerate(remaining_eqs_seq)):
            eq_sub = eq.subs(simulated_subs)
            free_vars = eq_sub.free_symbols
            
            if len(free_vars) == 1:
                target_var = list(free_vars)[0]
                sol = sp.solve(eq_sub, target_var)
                val = sol[0].evalf() if sol else 1.0
                
                sequence_steps.append({
                    "Bloque": eq_name,
                    "Resuelve": target_var.name,
                    "Tipo": "Secuencial",
                    "POE_Format": f"{eq_name}\n1  {target_var.name}"
                })
                
                simulated_subs[target_var] = val
                remaining_eqs_seq.pop(idx)
                found_step = True
                break
                
            elif len(free_vars) == 0:
                sequence_steps.append({
                    "Bloque": eq_name,
                    "Resuelve": f"Ver{ver_counter}",
                    "Tipo": "Verificación (Cierra ciclo)",
                    "POE_Format": f"{eq_name}\n1 Ver{ver_counter}"
                })
                ver_counter += 1
                remaining_eqs_seq.pop(idx)
                found_step = True
                break

        if not found_step and remaining_eqs_seq:
            sub_remaining_eqs = [eq.subs(simulated_subs) for name, eq in remaining_eqs_seq]
            rem_vars = list(set().union(*[eq.free_symbols for eq in sub_remaining_eqs]))
            
            if rem_vars:
                var_counts = {v: sum(1 for eq in sub_remaining_eqs if v in eq.free_symbols) for v in rem_vars}
                tear_var = max(var_counts, key=var_counts.get)
                
                sequence_steps.append({
                    "Bloque": f"ItVer{tear_counter}",
                    "Resuelve": tear_var.name,
                    "Tipo": "Corte / Rasgado",
                    "POE_Format": f"ItVer{tear_counter}\n1  {tear_var.name}"
                })
                
                # Esto es un señuelo numérico solo para destrabar el mapa de secuencia.
                simulated_subs[tear_var] = 1.0  
                tear_counter += 1
            else:
                break

    # =========================================================================
    # MOTOR 2: RESOLUCIÓN MATEMÁTICA EXACTA (Para la tabla final)
    # =========================================================================
    real_subs = known_vars.copy()
    eqs_for_solve = [eq[1].subs(real_subs) for eq in parsed_eqs]
    vars_to_solve = [s for s in all_symbols if s not in real_subs]
    
    calc_success = False
    
    if dof == 0:
        try:
            # Resuelve el sistema real en bloque sin las asignaciones artificiales
            sols = sp.solve(eqs_for_solve, vars_to_solve, dict=True)
            if sols:
                sol_dict = sols[0]
                for var_sym, val_expr in sol_dict.items():
                    try:
                        val_eval = val_expr.evalf() if hasattr(val_expr, 'evalf') else val_expr
                        real_subs[var_sym] = val_eval
                    except:
                        real_subs[var_sym] = val_expr
                calc_success = True
        except Exception as e:
            st.error(f"Error en resolución matemática exacta: {str(e)}")

    # =========================================================================
    # RENDERIZADO VISUAL
    # =========================================================================
    with tab_seq:
        st.subheader("Ruta Lógica de Resolución")
        if sequence_steps:
            df_seq = pd.DataFrame(sequence_steps)[["Bloque", "Resuelve", "Tipo"]]
            st.table(df_seq)

    with tab_reporte:
        st.subheader("Reporte Secuencial (Formato POE)")
        st.caption("Copia este bloque para pegarlo en tu informe.")
        poe_text = "\n\n".join([step["POE_Format"] for step in sequence_steps])
        st.text_area("Salida del POE:", value=poe_text, height=600)
        st.download_button(label="📄 Descargar reporte.txt", data=poe_text.encode('utf-8'), file_name="reporte_poe.txt", mime="text/plain")

    with tab_res:
        st.subheader("Resultados Consolidados")
        if not calc_success and dof == 0:
            st.warning("No se pudo calcular numéricamente el sistema completo. Revisa si hay ecuaciones o datos redundantes.")
        elif dof != 0:
            st.warning("Los resultados completos solo se muestran cuando GL = 0.")
        
        results = []
        for sym in all_symbols:
            # Aquí llamamos a real_subs (los resultados verídicos del motor 2)
            val = real_subs.get(sym, "No resuelto")
            is_specified = sym in known_vars
            
            if isinstance(val, (int, float, sp.Float)):
                val_formatted = f"{float(val):.4f}"
            else:
                val_formatted = str(val)

            results.append({
                "Variable": sym.name,
                "Estado": "Dato" if is_specified else "Calculada",
                "Valor": val_formatted
            })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Descargar Resultados en CSV", data=csv_data, file_name="reporte_poe_resultados.csv", mime="text/csv")
