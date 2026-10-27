import io
import re
import ollama
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="GenAI Data Analyst | Automated EDA Studio",
    page_icon="📊",
    layout="wide",
)

# --- 1. Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "df_history" not in st.session_state:
    st.session_state.df_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "auto_insights" not in st.session_state:
    st.session_state.auto_insights = ""

# --- 2. Sidebar: File Loading & Dynamic Filtering ---
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", type=["csv", "xlsx"]
    )

    if uploaded_file is not None and st.session_state.df is None:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file, header=None)
            header_idx = 0
            for idx, row in df_raw.iterrows():
                string_count = sum(
                    isinstance(val, str) and len(val.strip()) > 0 for val in row
                )
                if string_count >= (len(row) * 0.5):
                    header_idx = idx
                    break
            df_raw.columns = [
                str(col).strip() for col in df_raw.iloc[header_idx]
            ]
            df_raw = df_raw.iloc[header_idx + 1 :].reset_index(drop=True)

        st.session_state.df = df_raw
        st.session_state.df_history.append(df_raw.copy())
        st.success(f"Loaded: {uploaded_file.name}")

    if st.session_state.df is not None:
        st.divider()
        st.subheader("🛠️ Version Control")
        if st.button("↩️ Undo Last Change"):
            if len(st.session_state.df_history) > 1:
                st.session_state.df_history.pop()
                st.session_state.df = st.session_state.df_history[-1].copy()
                st.success("Reverted to previous dataset state.")
                st.rerun()
            else:
                st.warning("No previous steps to revert.")

        st.divider()
        st.subheader("🔍 Interactive Filters")
        filtered_df = st.session_state.df.copy()

        # Dynamic categorical filter
        cat_cols = filtered_df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Filter Category Column", ["None"] + cat_cols)
            if selected_cat != "None":
                unique_vals = filtered_df[selected_cat].dropna().unique().tolist()
                selected_vals = st.multiselect(f"Select values for {selected_cat}", unique_vals, default=unique_vals[:5])
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[selected_cat].isin(selected_vals)]

        # Dynamic numeric range slider
        num_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            selected_num = st.selectbox("Filter Numeric Column", ["None"] + num_cols)
            if selected_num != "None":
                min_v = float(filtered_df[selected_num].min())
                max_v = float(filtered_df[selected_num].max())
                if min_v < max_v:
                    val_range = st.slider(f"Range for {selected_num}", min_v, max_v, (min_v, max_v))
                    filtered_df = filtered_df[(filtered_df[selected_num] >= val_range[0]) & (filtered_df[selected_num] <= val_range[1])]

        st.session_state.active_df = filtered_df

        st.divider()
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Current View (CSV)",
            data=csv_data,
            file_name="analyst_export.csv",
            mime="text/csv",
        )

# --- 3. Main Dashboard ---
if st.session_state.df is not None:
    df = st.session_state.get("active_df", st.session_state.df)

    st.title("📊 GenAI Data Analyst & Automated EDA Studio")
    st.caption("Automated profiling, continuous conversational intelligence, and local execution with LLaMA 3.2.")

    # KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows (Active)", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
    m4.metric("Missing Values", f"{df.isna().sum().sum():,}")

    # Tabs for EDA
    t_preview, t_schema, t_stats, t_missing, t_insights = st.tabs(
        ["🔍 Preview", "ℹ️ Schema & Info", "📈 Descriptive Stats", "⚠️ Missing Values", "⚡ Executive Insights"]
    )

    with t_preview:
        col_h, col_t = st.columns(2)
        with col_h:
            st.markdown("**First 5 Rows:**")
            st.dataframe(df.head(5), use_container_width=True)
        with col_t:
            st.markdown("**Last 5 Rows:**")
            st.dataframe(df.tail(5), use_container_width=True)

    with t_schema:
        info_df = pd.DataFrame({
            "Column Name": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True)

    with t_stats:
        num_c = df.select_dtypes(include="number").columns
        cat_c = df.select_dtypes(exclude="number").columns
        if len(num_c) > 0:
            st.markdown("**Numerical Feature Profiling:**")
            st.dataframe(df[num_c].describe(), use_container_width=True)
        if len(cat_c) > 0:
            st.markdown("**Categorical Feature Profiling:**")
            st.dataframe(df[cat_c].describe(), use_container_width=True)

    with t_missing:
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df["Missing %"] = ((missing_df["Missing Count"] / len(df)) * 100).round(2)
        missing_df = missing_df.sort_values(by="Missing Count", ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(missing_df, use_container_width=True)
        with c2:
            fig_miss = px.bar(
                missing_df[missing_df["Missing Count"] > 0],
                x="Column",
                y="Missing Count",
                text="Missing %",
                title="Missing Value Distribution",
                color="Missing Count",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    with t_insights:
        if st.button("✨ Generate Executive Business Insights", type="secondary"):
            with st.spinner("LLaMA 3.2 is synthesizing executive insights..."):
                summary_prompt = f"""
You are a Principal Data Analyst. Analyze this dataset profile:
Columns: {list(df.columns)}
Summary Stats: {df.describe(include='all').to_dict()}
Missing Values: {df.isnull().sum().to_dict()}

Provide an executive report:
1. Executive Summary (2-3 sentences)
2. Top 3 Strategic Patterns/Findings
3. Identified Risks or Anomalies
4. Actionable Business Recommendations
Format cleanly with Markdown bullet points and bolding.
"""
                resp = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                st.session_state.auto_insights = resp["message"]["content"]

        if st.session_state.auto_insights:
            st.markdown(st.session_state.auto_insights)

    st.divider()

    # --- 4. Conversational Chat & Transformation Studio ---
    st.subheader("💬 Multi-Turn Analyst Chat & Transformation Studio")
    st.caption("Ask questions, instruct data transformations, or request Plotly charts. Context is retained across turns.")

    # Render conversational chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "code" in msg:
                st.code(msg["code"], language="python")
            if "fig" in msg and msg["fig"] is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

    user_query = st.chat_input("Ask a question, request a plot, or transform the dataset...")

    if user_query:
        # Display user query
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build prompt with conversation history context
        schema_context = f"""
Current Columns: {list(df.columns)}
Data Types: {df.dtypes.to_dict()}
Summary Sample (3 rows): {df.head(3).to_dict()}
"""
        system_instruction = """
You are a Senior Data Analyst and Python/Streamlit expert working on `df`.
You have `px` (Plotly Express) and `pd` available in the execution environment.

Rules:
1. Explain your reasoning and conclusions first in crisp, readable points.
2. If data transformation is needed, write code inside a single ```python block mutating `df`.
3. If visualization is needed, generate a Plotly figure assigned to variable `fig`.
4. Do not re-import pandas, create synthetic sample data, or use plt.show().
5. If no code is required, answer the question directly.
"""
        messages_payload = [{"role": "system", "content": system_instruction}]
        for m in st.session_state.chat_history[-4:]:  # Keep recent context window
            messages_payload.append({"role": m["role"], "content": m["content"]})
        messages_payload.append({"role": "user", "content": f"Schema:\n{schema_context}\n\nUser Request: {user_query}"})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing with LLaMA 3.2..."):
                response = ollama.chat(model="llama3.2", messages=messages_payload)
                reply = response["message"]["content"]
                st.markdown(reply)

                # Code extraction & safe execution
                code_match = re.search(r"```python(.*?)```", reply, re.DOTALL)
                generated_fig = None
                executed_code = None

                if code_match:
                    executed_code = code_match.group(1).strip()
                    st.code(executed_code, language="python")
                    try:
                        sandbox = {"df": df.copy(), "pd": pd, "px": px}
                        exec(executed_code, {}, sandbox)

                        if "fig" in sandbox and sandbox["fig"] is not None:
                            generated_fig = sandbox["fig"]
                            st.plotly_chart(generated_fig, use_container_width=True)

                        if not sandbox["df"].equals(df):
                            st.session_state.df = sandbox["df"]
                            st.session_state.df_history.append(sandbox["df"].copy())
                            st.success("Dataset successfully modified!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")

                # Save turn to chat history
                history_entry = {"role": "assistant", "content": reply}
                if executed_code:
                    history_entry["code"] = executed_code
                if generated_fig:
                    history_entry["fig"] = generated_fig

                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append(history_entry)

else:
    st.info("👈 Upload an Excel or CSV file from the sidebar to start analysis.")