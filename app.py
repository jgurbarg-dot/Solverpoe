import streamlit as st
import sympy as sp
import pandas as pd
import itertools

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption("Secuenciación automática, cálculo de Subsistemas/Iteración, Memoria de Cálculo y resolución exacta.")

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
        val = st.sidebar.number_input(f"Valor para {var}:", value=1.0000, step=0.1, format="%.6f")
        known_vars[sp.Symbol(var)] = val

    # 3. ANÁLISIS DE GRADOS DE LIBERTAD (GL) Y DEPENDENCIA
    total_vars = len(all_symbols)
    num_specified = len(known_vars)
    unknown_vars = [s for s in all_symbols if s not in known_vars]
    num_unknowns = len(unknown_vars)
    num_eqs = len(parsed_eqs)
    dof = num_unknowns - num_eqs

    dependencia_str = "Independiente"
    try:
        if num_eqs > 0 and num_unknowns > 0:
            exprs = [eq[1].lhs - eq[1].rhs for eq in parsed_eqs]
            J = sp.Matrix(exprs).jacobian(unknown_vars)
            
            temp_subs = known_vars.copy()
            for uv in unknown_vars:
                temp_subs[uv] = 1.0 
                
            J_num = J.subs(temp_subs)
            rango = J_num.rank()
            
            if rango < num_eqs:
                dependencia_str = "Dependiente (Existen ecuaciones redundantes)"
        elif num_eqs > num_unknowns:
             dependencia_str = "Dependiente (Sistema Sobredeterminado)"
    except Exception:
        dependencia_str = "No evaluado"

    st.subheader("📌 Análisis del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables Totales", total_vars)
    c2.metric("Datos", num_specified)
    c3.metric("Incógnitas Libres", num_unknowns)
    c4.metric("Grados de Libertad (GL)", dof)

    if dof > 0:
        st.warning(f"⚠ **Sistema Subdeterminado (GL = {dof})**: Faltan especificar {dof} variable(s).")
    elif dof < 0:
        st.error(f"❌ **Sistema Sobredeterminado (GL = {dof})**: Sobran {-dof} ecuación(es).")
    else:
        st.success("✅ **Sistema Determinado (GL = 0)**: Listo para resolver.")

    # =========================================================================
    # MOTOR 1: MODO ITERACIÓN (Corte / Rasgado con ItVer)
    # =========================================================================
    simulated_subs_iter = known_vars.copy()
    remaining_eqs_iter = parsed_eqs.copy()
    sequence_steps_iter = []
    memoria_iter = []
    
    tear_counter = 0
    ver_counter = 0
    step_iter = 1

    while remaining_eqs_iter:
        found_step = False
        
        for idx, (eq_name, eq) in list(enumerate(remaining_eqs_iter)):
            eq_sub = eq.subs(simulated_subs_iter)
            free_vars = eq_sub.free_symbols
            
            if len(free_vars) == 1:
                target_var = list(free_vars)[0]
                sol = sp.solve(eq_sub, target_var)
                val = sol[0].evalf() if sol else 1.0
                
                sequence_steps_iter.append({"Bloque": eq_name, "Resuelve": target_var.name, "Tipo": "Secuencial", "POE_Format": f"{eq_name}\n1  {target_var.name}"})
                memoria_iter.append(f"**Paso {step_iter}:** De la `{eq_name}` se despeja de forma directa `{target_var.name}`.")
                
                simulated_subs_iter[target_var] = val
                remaining_eqs_iter.pop(idx)
                found_step = True
                step_iter += 1
                break
                
            elif len(free_vars) == 0:
                sequence_steps_iter.append({"Bloque": eq_name, "Resuelve": f"Ver{ver_counter}", "Tipo": "Verificación", "POE_Format": f"{eq_name}\n1 Ver{ver_counter}"})
                memoria_iter.append(f"**Paso {step_iter}:** La `{eq_name}` cierra el balance. Se utiliza como Ecuación Verificadora (`Ver{ver_counter}`) para controlar la convergencia del ciclo.")
                
                ver_counter += 1
                remaining_eqs_iter.pop(idx)
                found_step = True
                step_iter += 1
                break

        if not found_step and remaining_eqs_iter:
            sub_remaining_eqs = [eq.subs(simulated_subs_iter) for name, eq in remaining_eqs_iter]
            rem_vars = list(set().union(*[eq.free_symbols for eq in sub_remaining_eqs]))
            
            if rem_vars:
                var_counts = {v: sum(1 for eq in sub_remaining_eqs if v in eq.free_symbols) for v in rem_vars}
                tear_var = max(var_counts, key=var_counts.get)
                
                sequence_steps_iter.append({"Bloque": f"ItVer{tear_counter}", "Resuelve": tear_var.name, "Tipo": "Corte (ItVer)", "POE_Format": f"ItVer{tear_counter}\n1  {tear_var.name}"})
                memoria_iter.append(f"**Paso {step_iter}:** Se detecta un ciclo. Se asume un valor inicial para `{tear_var.name}` convirtiéndola en Variable de Corte (`ItVer{tear_counter}`) para destrabar el sistema.")
                
                simulated_subs_iter[tear_var] = 1.0  
                tear_counter += 1
                step_iter += 1
            else:
                break

    # =========================================================================
    # MOTOR 1B: MODO SUBSISTEMAS (Resolución de bloques en cascada con ItVer)
    # =========================================================================
    simulated_subs_sub = known_vars.copy()
    remaining_eqs_sub = parsed_eqs.copy()
    sequence_steps_sub = []
    memoria_sub = []
    step_sub = 1
    sub_tear_counter = 0

    while remaining_eqs_sub:
        found_step = False
        
        # 1. Ecuaciones directas (1x1)
        for idx, (eq_name, eq) in list(enumerate(remaining_eqs_sub)):
            eq_sub = eq.subs(simulated_subs_sub)
            free_vars = list(eq_sub.free_symbols)
            
            if len(free_vars) == 1:
                target_var = free_vars[0]
                sol = sp.solve(eq_sub, target_var)
                val = sol[0].evalf() if sol else 1.0
                
                sequence_steps_sub.append({"Bloque": eq_name, "Resuelve": target_var.name, "Tipo": "Secuencial", "POE_Format": f"{eq_name}\n1  {target_var.name}"})
                memoria_sub.append(f"**Paso {step_sub}:** Resolución directa de `{eq_name}`. Se despeja `{target_var.name}`.")
                
                simulated_subs_sub[target_var] = val
                remaining_eqs_sub.pop(idx)
                found_step = True
                step_sub += 1
                break
                
        # 2. Búsqueda de Subsistemas acoplados (NxN)
        if not found_step and remaining_eqs_sub:
            for block_size in range(2, len(remaining_eqs_sub) + 1):
                for combo in itertools.combinations(remaining_eqs_sub, block_size):
                    combo_eqs = [eq.subs(simulated_subs_sub) for name, eq in combo]
                    combo_vars = list(set().union(*[e.free_symbols for e in combo_eqs]))
                    
                    if len(combo_vars) == block_size:
                        eq_names = ", ".join([name for name, eq in combo])
                        var_names = ", ".join([v.name for v in combo_vars])
                        
                        # FORMATO ESTILO ITVER PARA DELIMITAR SUBSISTEMAS EN POE
                        poe_lines = []
                        # Abrir bloque con ItVer
                        tear_v = combo_vars[0]
                        poe_lines.append(f"ItVer{sub_tear_counter}\n1  {tear_v.name}")
                        
                        # Ecuaciones intermedias
                        rem_vars = combo_vars[1:]
                        for j in range(block_size - 1):
                            poe_lines.append(f"{combo[j][0]}\n1  {rem_vars[j].name}")
                            
                        # Cerrar bloque con Verificador
                        poe_lines.append(f"{combo[-1][0]}\n1 Ver{sub_tear_counter}")
                        
                        formato_poe_bloque = "\n\n".join(poe_lines)
                        
                        sequence_steps_sub.append({
                            "Bloque": eq_names, 
                            "Resuelve": var_names, 
                            "Tipo": f"Subsistema {block_size}x{block_size}", 
                            "POE_Format": formato_poe_bloque
                        })
                        memoria_sub.append(f"**Paso {step_sub}:** Se detecta acoplamiento. Se resuelve el subsistema simultáneo formado por `{eq_names}` para hallar las incógnitas `{var_names}`. *(El reporte de texto lo estructura con `ItVer{sub_tear_counter}` y `Ver{sub_tear_counter}` para que POE identifique los límites del bloque)*.")
                        
                        # Asignar valores temporales para continuar descifrando la ruta
                        for v in combo_vars:
                            simulated_subs_sub[v] = 1.0
                        
                        remaining_eqs_sub = [item for item in remaining_eqs_sub if item not in combo]
                        sub_tear_counter += 1
                        found_step = True
                        step_sub += 1
                        break
                if found_step:
                    break
                    
            if not found_step:
                memoria_sub.append(f"**Paso {step_sub}:** ⚠ No se pudo agrupar el resto de ecuaciones en subsistemas cuadrados perfectos.")
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
        except Exception:
            pass

    # =========================================================================
    # DIAGNÓSTICO E INTERFAZ
    # =========================================================================
    tipo_resolucion = "Iteración" if tear_counter > 0 else "Subsistemas"
    st.info(f"**Diagnóstico Primario:** {tipo_resolucion} | **Dependencia:** {dependencia_str}")
    st.divider()

    tab_seq, tab_reporte, tab_memoria, tab_res = st.tabs(["🔄 Matriz y Secuencia", "📄 Reportes Estilo POE", "📝 Memoria de Cálculo", "🚀 Resultados"])

    with tab_seq:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Secuencia por Iteración")
            if sequence_steps_iter:
                st.dataframe(pd.DataFrame(sequence_steps_iter)[["Bloque", "Resuelve", "Tipo"]], use_container_width=True)
        with col2:
            st.subheader("Secuencia por Subsistemas")
            if sequence_steps_sub:
                st.dataframe(pd.DataFrame(sequence_steps_sub)[["Bloque", "Resuelve", "Tipo"]], use_container_width=True)

    with tab_reporte:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reporte MODO ITERACIÓN")
            poe_iter = "\n\n".join([step["POE_Format"] for step in sequence_steps_iter])
            st.text_area("Copia formato Iteración:", value=poe_iter, height=400)
            st.download_button("📄 Descargar Iteración.txt", data=poe_iter.encode('utf-8'), file_name="reporte_iteracion.txt")
        with col2:
            st.subheader("Reporte MODO SUBSISTEMAS")
            poe_sub = "\n\n".join([step["POE_Format"] for step in sequence_steps_sub])
            st.text_area("Copia formato Subsistemas:", value=poe_sub, height=400)
            st.download_button("📄 Descargar Subsistemas.txt", data=poe_sub.encode('utf-8'), file_name="reporte_subsistemas.txt")

    with tab_memoria:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Memoria Descriptiva (Iteración)")
            for linea in memoria_iter:
                st.markdown(linea)
        with col2:
            st.subheader("Memoria Descriptiva (Subsistemas)")
            for linea in memoria_sub:
                st.markdown(linea)

    with tab_res:
        st.subheader("Resultados Consolidados")
        if not calc_success and dof == 0:
            st.warning("No se pudo calcular numéricamente el sistema completo. Revisa si hay ecuaciones o datos redundantes.")
        elif dof != 0:
            st.warning("Los resultados completos solo se muestran cuando GL = 0.")
        
        results = []
        for sym in all_symbols:
            val = real_subs.get(sym, "No resuelto")
            is_specified = sym in known_vars
            
            if isinstance(val, (int, float, sp.Float)):
                val_formatted = f"{float(val):.4f}"
            else:
                val_formatted = str(val)

            results.append({"Variable": sym.name, "Estado": "Dato" if is_specified else "Calculada", "Valor": val_formatted})

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        st.download_button("📥 Descargar Resultados en CSV", data=df_results.to_csv(index=False).encode('utf-8'), file_name="resultados.csv")
