
import streamlit as st
import sympy as sp
import pandas as pd
import networkx as nx

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption("Secuenciación automática, cálculo de Subsistemas/Iteración, Memoria de Cálculo y resolución exacta (Optimizado para 300+ Ecuaciones).")

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

# Parseo protegido por caché para no recalcular 300 strings en cada clic
@st.cache_resource
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
    # Extracción eficiente de variables
    all_symbols_set = set()
    for _, eq in parsed_eqs:
        all_symbols_set.update(eq.free_symbols)
    all_symbols = sorted(list(all_symbols_set), key=lambda s: s.name)
    var_names = [s.name for s in all_symbols]

    st.sidebar.header("2. Especificación de Datos")
    known_selected = st.sidebar.multiselect("Seleccione variables conocidas (Datos):", var_names)
    
    known_vars = {}
    for var in known_selected:
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
        st.warning(f"⚠ **Sistema Subdeterminado (GL = {dof})**")
    elif dof < 0:
        st.error(f"❌ **Sistema Sobredeterminado (GL = {dof})**")
    else:
        st.success("✅ **Sistema Determinado (GL = 0)**")

    # =========================================================================
    # MOTOR 1: MODO ITERACIÓN (Corte / Rasgado con ItVer) - Optimizado
    # =========================================================================
    @st.cache_resource
    def engine_iteracion(parsed, known_dict):
        simulated_subs = known_dict.copy()
        remaining_eqs = parsed.copy()
        seq_steps = []
        memoria = []
        tear_counter, ver_counter, step = 0, 0, 1

        while remaining_eqs:
            found_step = False
            for idx, (eq_name, eq) in enumerate(remaining_eqs):
                # Evaluación rápida por sets en lugar de sustitución pesada sympy
                free_vars = [s for s in eq.free_symbols if s not in simulated_subs]
                
                if len(free_vars) == 1:
                    target_var = free_vars[0]
                    seq_steps.append({"Bloque": eq_name, "Resuelve": target_var.name, "Tipo": "Secuencial", "POE_Format": f"{eq_name}\n1  {target_var.name}"})
                    memoria.append(f"**Paso {step}:** De `{eq_name}` se despeja de forma directa `{target_var.name}`.")
                    simulated_subs[target_var] = 1.0 # Valor dummy para secuenciación rápida
                    remaining_eqs.pop(idx)
                    found_step = True
                    step += 1
                    break
                
                elif len(free_vars) == 0:
                    seq_steps.append({"Bloque": eq_name, "Resuelve": f"Ver{ver_counter}", "Tipo": "Verificación", "POE_Format": f"{eq_name}\n1 Ver{ver_counter}"})
                    memoria.append(f"**Paso {step}:** `{eq_name}` actúa como Verificadora (`Ver{ver_counter}`).")
                    ver_counter += 1
                    remaining_eqs.pop(idx)
                    found_step = True
                    step += 1
                    break

            if not found_step and remaining_eqs:
                # Detección de variable de corte (Tearing)
                rem_vars_counts = {}
                for _, eq in remaining_eqs:
                    for v in eq.free_symbols:
                        if v not in simulated_subs:
                            rem_vars_counts[v] = rem_vars_counts.get(v, 0) + 1
                
                if rem_vars_counts:
                    tear_var = max(rem_vars_counts, key=rem_vars_counts.get)
                    seq_steps.append({"Bloque": f"ItVer{tear_counter}", "Resuelve": tear_var.name, "Tipo": "Corte (ItVer)", "POE_Format": f"ItVer{tear_counter}\n1  {tear_var.name}"})
                    memoria.append(f"**Paso {step}:** Ciclo detectado. Corte en `{tear_var.name}` (`ItVer{tear_counter}`).")
                    simulated_subs[tear_var] = 1.0  
                    tear_counter += 1
                    step += 1
                else:
                    break
        return seq_steps, memoria, tear_counter

    # =========================================================================
    # MOTOR 1B: MODO SUBSISTEMAS - Optimizado con Grafos (Cero itertools)
    # =========================================================================
    @st.cache_resource
    def engine_subsistemas(parsed, known_dict):
        simulated_subs = known_dict.copy()
        remaining_eqs = parsed.copy()
        seq_steps = []
        memoria = []
        step, sub_tear_counter = 1, 0

        while remaining_eqs:
            found_step = False
            
            # 1. Resolver directas 1x1 primero (Heurística rápida)
            for idx, (eq_name, eq) in enumerate(remaining_eqs):
                free_vars = [s for s in eq.free_symbols if s not in simulated_subs]
                if len(free_vars) == 1:
                    target_var = free_vars[0]
                    seq_steps.append({"Bloque": eq_name, "Resuelve": target_var.name, "Tipo": "Secuencial", "POE_Format": f"{eq_name}\n1  {target_var.name}"})
                    memoria.append(f"**Paso {step}:** Resolución directa de `{eq_name}` para `{target_var.name}`.")
                    simulated_subs[target_var] = 1.0
                    remaining_eqs.pop(idx)
                    found_step = True
                    step += 1
                    break

            # 2. Análisis Topológico de Bloques (Sustituye itertools)
            if not found_step and remaining_eqs:
                # Crear grafo bipartito Ecuación <-> Variable
                B = nx.Graph()
                for name, eq in remaining_eqs:
                    B.add_node(name, bipartite=0)
                    free = [s for s in eq.free_symbols if s not in simulated_subs]
                    for v in free:
                        B.add_node(v.name, bipartite=1)
                        B.add_edge(name, v.name)
                
                # Obtener componentes conexos (Subsistemas aislados)
                components = list(nx.connected_components(B))
                
                # Procesar el primer componente válido como bloque
                for comp in components:
                    comp_eqs = [n for n in comp if n.startswith("Ec")]
                    comp_vars = [n for n in comp if not n.startswith("Ec")]
                    
                    if len(comp_eqs) > 1 and len(comp_eqs) == len(comp_vars):
                        eq_names = ", ".join(comp_eqs)
                        var_names = ", ".join(comp_vars)
                        block_size = len(comp_eqs)
                        
                        poe_lines = []
                        poe_lines.append(f"ItVer{sub_tear_counter}\n1  {comp_vars[0]}")
                        for j in range(block_size - 1):
                            poe_lines.append(f"{comp_eqs[j]}\n1  {comp_vars[j+1]}")
                        poe_lines.append(f"{comp_eqs[-1]}\n1 Ver{sub_tear_counter}")
                        
                        seq_steps.append({
                            "Bloque": eq_names, 
                            "Resuelve": var_names, 
                            "Tipo": f"Subsistema {block_size}x{block_size}", 
                            "POE_Format": "\n\n".join(poe_lines)
                        })
                        memoria.append(f"**Paso {step}:** Acoplamiento resuelto por Grafos. Subsistema de `{eq_names}` para `{var_names}`.")
                        
                        # Actualizar estado
                        for v_name in comp_vars:
                            simulated_subs[sp.Symbol(v_name)] = 1.0
                        remaining_eqs = [eq for eq in remaining_eqs if eq[0] not in comp_eqs]
                        
                        sub_tear_counter += 1
                        found_step = True
                        step += 1
                        break
                
                # Fallback si el grafo no cuadra perfectamente
                if not found_step:
                    memoria.append(f"**Paso {step}:** ⚠ El grafo residual no es cuadrado. Se requiere análisis avanzado (Sobredeterminado o Subdeterminado local).")
                    break

        return seq_steps, memoria

    # =========================================================================
    # MOTOR 2: RESOLUCIÓN MATEMÁTICA EXACTA (Bloqueada por Caché)
    # =========================================================================
    @st.cache_resource
    def engine_solve(parsed, known_dict, is_dof_zero):
        real_subs = known_dict.copy()
        calc_success = False
        if is_dof_zero:
            eqs_for_solve = [eq[1].subs(real_subs) for eq in parsed]
            vars_to_solve = [s for s in all_symbols if s not in real_subs]
            try:
                # Timeout implícito manejado por el usuario (puede demorar en sistemas masivos)
                sols = sp.solve(eqs_for_solve, vars_to_solve, dict=True)
                if sols:
                    for var_sym, val_expr in sols[0].items():
                        try:
                            real_subs[var_sym] = val_expr.evalf() if hasattr(val_expr, 'evalf') else val_expr
                        except:
                            real_subs[var_sym] = val_expr
                    calc_success = True
            except Exception:
                pass
        return real_subs, calc_success

    # Ejecución de motores
    with st.spinner("Mapeando topología del sistema..."):
        seq_iter, mem_iter, tear_count = engine_iteracion(parsed_eqs, known_vars)
        seq_sub, mem_sub = engine_subsistemas(parsed_eqs, known_vars)
    
    with st.spinner("Calculando solución exacta..."):
        resultados_reales, solve_success = engine_solve(parsed_eqs, known_vars, dof == 0)

    # =========================================================================
    # INTERFAZ
    # =========================================================================
    tipo_resolucion = "Iteración" if tear_count > 0 else "Subsistemas (Desacoplado)"
    st.info(f"**Diagnóstico Estructural:** {tipo_resolucion}")
    st.divider()

    tab_seq, tab_reporte, tab_memoria, tab_res = st.tabs(["🔄 Matrices", "📄 Reportes POE", "📝 Memoria", "🚀 Resultados numéricos"])

    with tab_seq:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Secuencia por Iteración")
            if seq_iter:
                st.dataframe(pd.DataFrame(seq_iter)[["Bloque", "Resuelve", "Tipo"]], use_container_width=True)
        with col2:
            st.subheader("Secuencia por Grafos (Subsistemas)")
            if seq_sub:
                st.dataframe(pd.DataFrame(seq_sub)[["Bloque", "Resuelve", "Tipo"]], use_container_width=True)

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
            for linea in mem_iter: st.markdown(linea)
        with col2:
            for linea in mem_sub: st.markdown(linea)

    with tab_res:
        st.subheader("Resultados Consolidados")
        if not solve_success and dof == 0:
            st.warning("SymPy no pudo calcular numéricamente el sistema completo (Demasiado complejo o redundante).")
        
        results = []
        for sym in all_symbols:
            val = resultados_reales.get(sym, "No resuelto")
            is_specified = sym in known_vars
            val_formatted = f"{float(val):.4f}" if isinstance(val, (int, float, sp.Float)) else str(val)
            results.append({"Variable": sym.name, "Estado": "Dato" if is_specified else "Calculada", "Valor": val_formatted})

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
