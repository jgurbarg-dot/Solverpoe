import streamlit as st
import sympy as sp
import pandas as pd

st.set_page_config(
    page_title="Programa Ordenador de Ecuaciones (POE)",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Programa Ordenador de Ecuaciones (POE)")
st.caption("Secuenciación con variables de corte (ItVer), ecuaciones verificadoras (Ver) y reporte nativo.")

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

# Parsing de ecuaciones
eq_lines = [line.strip() for line in raw_eqs.split("\n") if line.strip()]
parsed_eqs = []
parse_errors = []
eq_dict = {}  # Para guardar el nombre Ec1, Ec2...

for i, line in enumerate(eq_lines):
    try:
        eq_name = f"Ec{i+1}"
        if "=" in line:
            parts = line.split("=")
            if len(parts) == 2:
                lhs = sp.sympify(parts[0])
                rhs = sp.sympify(parts[1])
                eq_obj = sp.Eq(lhs, rhs)
                parsed_eqs.append((eq_name, eq_obj))
                eq_dict[eq_name] = eq_obj
            else:
                parse_errors.append(f"Línea {i+1}: Múltiples signos '=' detectados.")
        else:
            expr = sp.sympify(line)
            eq_obj = sp.Eq(expr, 0)
            parsed_eqs.append((eq_name, eq_obj))
            eq_dict[eq_name] = eq_obj
    except Exception as e:
        parse_errors.append(f"{eq_name} ('{line}'): Error - {str(e)}")

if parse_errors:
    for err in parse_errors:
        st.error(err)

if parsed_eqs:
    all_symbols = sorted(list(set().union(*[eq.free_symbols for name, eq in parsed_eqs])), key=lambda s: s.name)
    var_names = [s.name for s in all_symbols]

    # 2. ESPECIFICACIÓN DE VARIABLES
    st.sidebar.header("2. Especificación de Datos")
    known_selected = st.sidebar.multiselect("Seleccione variables conocidas (Datos):", var_names)
    
    known_vars = {}
    for var in known_selected:
        val = st.sidebar.number_input(f"Valor para {var}:", value=1.0000, step=0.1, format="%.4f")
        known_vars[sp.Symbol(var)] = val

    # 3. ANÁLISIS DE GRADOS DE LIBERTAD
    unknown_vars = [s for s in all_symbols if s not in known_vars]
    dof = len(unknown_vars) - len(parsed_eqs)

    st.subheader("📌 Análisis del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables Totales", len(all_symbols))
    c2.metric("Datos", len(known_vars))
    c3.metric("Incógnitas", len(unknown_vars))
    c4.metric("Grados de Libertad", dof)

    st.divider()

    # 4. ALGORITMO DE SECUENCIACIÓN (Estilo POE)
    tab_seq, tab_reporte, tab_res = st.tabs(["🔄 Matriz y Secuencia", "📄 Reporte Estilo POE", "🚀 Resultados"])

    substitutions = known_vars.copy()
    remaining_eqs = parsed_eqs.copy()
    sequence_steps = []
    
    tear_counter = 0
    ver_counter = 0

    while remaining_eqs:
        found_step = False
        
        # Intentar resolver ecuaciones secuencialmente
        for idx, (eq_name, eq) in list(enumerate(remaining_eqs)):
            eq_sub = eq.subs(substitutions)
            free_vars = eq_sub.free_symbols
            
            # Caso 1: Ecuación con 1 sola incógnita (Calcula una variable)
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
                
            # Caso 2: Ecuación con 0 incógnitas (Verificadora de un ciclo)
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

        # Caso 3: Sistema Acoplado (Requiere Variable de Corte / Tear Variable)
        if not found_step and remaining_eqs:
            # Seleccionar la variable más repetida como variable de corte
            sub_remaining_eqs = [eq.subs(substitutions) for name, eq in remaining_eqs]
            rem_vars = list(set().union(*[eq.free_symbols for eq in sub_remaining_eqs]))
            
            if rem_vars:
                var_counts = {v: sum(1 for eq in sub_remaining_eqs if v in eq.free_symbols) for v in rem_vars}
                tear_var = max(var_counts, key=var_counts.get)
                
                sequence_steps.append({
                    "Bloque": f"ItVer{tear_counter}",
                    "Resuelve": tear_var.name,
                    "Tipo": "Variable de Corte (Rasgado)",
                    "POE_Format": f"ItVer{tear_counter}\n1  {tear_var.name}"
                })
                
                # Asignamos un valor simbólico/hipotético para que el programa siga analizando el camino
                substitutions[tear_var] = 1.0  
                tear_counter += 1
            else:
                break # Seguridad contra bucles infinitos

    with tab_seq:
        st.subheader("Ruta de Resolución (Orden de Cálculo)")
        if sequence_steps:
            df_seq = pd.DataFrame(sequence_steps)[["Bloque", "Resuelve", "Tipo"]]
            st.table(df_seq)

    with tab_reporte:
        st.subheader("Reporte Secuencial Formato POE")
        st.caption("Copia este bloque para pegarlo en tu informe o editor de texto.")
        
        # Generar texto idéntico al solicitado
        poe_text = "\n\n".join([step["POE_Format"] for step in sequence_steps])
        
        st.text_area("Salida del POE:", value=poe_text, height=600)
        
        st.download_button(
            label="📄 Descargar como reporte.txt",
            data=poe_text.encode('utf-8'),
            file_name="reporte_poe.txt",
            mime="text/plain"
        )

    with tab_res:
        st.subheader("Valores Finales (Iteración de prueba)")
        st.info("Nota: Los valores de las variables de corte (`ItVer`) fueron asumidos como 1.0 temporalmente para generar la secuencia. Para convergencia real se requiere un iterador numérico (ej. Wegstein/Newton-Raphson).")
        
        results = []
        for sym in all_symbols:
            val = substitutions.get(sym, "No resuelto")
            is_specified = sym in known_vars
            results.append({
                "Variable": sym.name,
                "Estado": "Especificada (Dato)" if is_specified else "Calculada/Asumida",
                "Valor": f"{float(val):.4f}" if isinstance(val, (int, float, sp.Float)) else str(val)
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)
