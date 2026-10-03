import streamlit as st
import sympy as sp
import pandas as pd

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption("Secuenciación automática con detección de variables de corte (ItVer), ecuaciones verificadoras (Ver) y reporte nativo.")

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
    all_symbols = sorted(list(set().union(*[eq.free_symbols for name, eq in parsed_eqs])), key=lambda s: s.name)
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
    c2.metric("Variables Especificadas (Datos)", num_specified)
    c3.metric("Incógnitas Libres", num_unknowns)
    c4.metric("Grados de Libertad (GL)", dof)

    if dof > 0:
        st.warning(f"⚠️ **Sistema Subdeterminado (GL = {dof})**: Faltan especificar {dof} variable(s).")
    elif dof < 0:
        st.error(f"❌ **Sistema Sobredeterminado (GL = {dof})**: Sobran {-dof} ecuación(es).")
    else:
        st.success("✅ **Sistema Determinado (GL = 0)**: Listo para ordenar.")

    st.divider()

    # 4. ALGORITMO DE SECUENCIACIÓN, RASGADO Y VERIFICACIÓN
    tab_seq, tab_reporte, tab_res = st.tabs(["🔄 Matriz y Secuencia", "📄 Reporte Estilo POE", "🚀 Resultados"])

    substitutions = known_vars.copy()
    remaining_eqs = parsed_eqs.copy()
    sequence_steps = []
    
    tear_counter = 0
    ver_counter = 0

    while remaining_eqs:
        found_step = False
        
        # Búsqueda secuencial
        for idx, (eq_name, eq) in list(enumerate(remaining_eqs)):
            eq_sub = eq.subs(substitutions)
            free_vars = eq_sub.free_symbols
            
            # Caso 1: Una incógnita -> Se calcula la variable
            if len(free_vars) == 1:
                target_var = list(free_vars)[0]
                sol = sp.solve(eq_sub, target_var)
                val = sol[0].evalf() if sol else sp.nan
                
                sequence_steps.append({
                    "Bloque": eq_name,
                    "Resuelve": target_var.name,
                    "Tipo": "Secuencial",
                    "POE_Format": f"{eq_name}\n1  {target_var.name}"
                })
                
                substitutions[target_var] = val
                remaining_eqs.pop(idx)
                found_step = True
                break
                
            # Caso 2: Cero incógnitas -> Ecuación Verificadora que cierra un ciclo
            elif len(free_vars) == 0:
                sequence_steps.append({
                    "Bloque": eq_name,
                    "Resuelve": f"Ver{ver_counter}",
                    "Tipo": "Verificación (Cierra ciclo)",
                    "POE_Format": f"{eq_name}\n1 Ver{ver_counter}"
                })
                ver_counter += 1
                remaining_eqs.pop(idx)
                found_step = True
                break

        # Caso 3: Atasco por ciclo -> Selección de Variable de Corte (ItVer)
        if not found_step and remaining_eqs:
            sub_remaining_eqs = [eq.subs(substitutions) for name, eq in remaining_eqs]
            rem_vars = list(set().union(*[eq.free_symbols for eq in sub_remaining_eqs]))
            
            if rem_vars:
                # Heurística: Elegir la variable que más se repite en el subsistema acoplado
                var_counts = {v: sum(1 for eq in sub_remaining_eqs if v in eq.free_symbols) for v in rem_vars}
                tear_var = max(var_counts, key=var_counts.get)
                
                sequence_steps.append({
                    "Bloque": f"ItVer{tear_counter}",
                    "Resuelve": tear_var.name,
                    "Tipo": "Corte / Rasgado",
                    "POE_Format": f"ItVer{tear_counter}\n1  {tear_var.name}"
                })
                
                # Asignar valor ficticio para continuar descifrando la ruta secuencial
                substitutions[tear_var] = 1.0  
                tear_counter += 1
            else:
                break

    # Pestaña 1: Visualización de la secuencia en tabla
    with tab_seq:
        st.subheader("Ruta Lógica de Resolución")
        if sequence_steps:
            df_seq = pd.DataFrame(sequence_steps)[["Bloque", "Resuelve", "Tipo"]]
            st.table(df_seq)

    # Pestaña 2: Reporte idéntico al POE (Formato de texto)
    with tab_reporte:
        st.subheader("Reporte Secuencial (Formato POE)")
        st.caption("Salida de texto puro con la secuencia de cálculo, variables de corte y verificadoras.")
        
        poe_text = "\n\n".join([step["POE_Format"] for step in sequence_steps])
        
        st.text_area("Copia este bloque:", value=poe_text, height=500)
        
        st.download_button(
            label="📄 Descargar reporte.txt",
            data=poe_text.encode('utf-8'),
            file_name="reporte_poe.txt",
            mime="text/plain"
        )

    # Pestaña 3: Tabla de resultados finales
    with tab_res:
        st.subheader("Resultados Consolidados")
        st.info("Nota: Si existen variables iterativas (ItVer), sus valores se asumieron temporalmente como 1.0 para definir el orden. El programa POE real utiliza algoritmos numéricos para encontrar el valor exacto de convergencia.")
        
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
                "Estado": "Dato" if is_specified else "Calculada / Asumida",
                "Valor": val_formatted
            })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
