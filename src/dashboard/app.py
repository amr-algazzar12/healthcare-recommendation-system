"""
app.py — Streamlit Clinician Dashboard (Milestone 4)

Provides a clinical-grade UI for:
  - Patient search and profile viewing
  - Running recommendations on demand
  - Visualising recommendation scores and explanations
  - Exploring population-level condition and medication statistics

Run:
    streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    (launched by docker/app/entrypoint.sh)
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = os.environ.get("API_BASE_URL", "http://localhost:5050")
CH_HOST   = os.environ.get("CLICKHOUSE_HOST",      "clickhouse")
CH_PORT   = int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8123"))
CH_DB     = os.environ.get("CLICKHOUSE_DB",         "healthcare")
CH_USER   = os.environ.get("CLICKHOUSE_USER",        "healthcare_user")
CH_PASS   = os.environ.get("CLICKHOUSE_PASSWORD",    "ch_secret_2026")

st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_ch_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        database=CH_DB, username=CH_USER, password=CH_PASS,
        compress=False,
    )


@st.cache_data(ttl=300)
def fetch_population_stats() -> dict:
    client = get_ch_client()
    total  = client.query(f"SELECT count() FROM {CH_DB}.patient_features").result_rows[0][0]
    flags  = client.query_df(f"""
        SELECT
            sum(has_diabetes)        AS diabetes,
            sum(has_hypertension)    AS hypertension,
            sum(has_asthma)          AS asthma,
            sum(has_hyperlipidemia)  AS hyperlipidemia,
            sum(has_coronary_disease) AS coronary_disease
        FROM {CH_DB}.patient_features
    """).iloc[0].to_dict()

    age_stats = client.query_df(f"""
        SELECT
            round(avg(age), 1) AS mean_age,
            min(age)           AS min_age,
            max(age)           AS max_age
        FROM {CH_DB}.patient_features
    """).iloc[0].to_dict()

    return {"total": total, "flags": flags, "age": age_stats}


@st.cache_data(ttl=60)
def search_patients(query: str, limit: int = 50) -> pd.DataFrame:
    client = get_ch_client()
    safe   = query.replace("'", "''")
    return client.query_df(f"""
        SELECT pf.patient_id, pf.age, pf.gender_encoded,
               pf.num_conditions, pf.num_medications, pf.num_encounters,
               pf.has_diabetes, pf.has_hypertension, pf.has_asthma,
               pf.has_hyperlipidemia, pf.has_coronary_disease,
               p.first, p.last, p.city
        FROM {CH_DB}.patient_features pf
        LEFT JOIN {CH_DB}.patients p ON pf.patient_id = p.patient_id
        WHERE pf.patient_id LIKE '%{safe}%'
           OR p.last ILIKE '%{safe}%'
           OR p.first ILIKE '%{safe}%'
        LIMIT {limit}
    """)


def call_api(endpoint: str, method: str = "GET",
             payload: dict | None = None) -> dict | None:
    try:
        if method == "POST":
            r = requests.post(f"{API_BASE}{endpoint}",
                              json=payload, timeout=120)
        else:
            r = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def gender_label(code: int) -> str:
    return {0: "Female", 1: "Male"}.get(int(code), "Unknown")


def race_label(code: int) -> str:
    return {0: "White", 1: "Black", 2: "Asian",
            3: "Hispanic", 4: "Native", 5: "Other"}.get(int(code), "Unknown")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("Healthcare Rec System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔍 Patient Search",
         "💊 Recommendations", "📊 Analytics", "⚙️ System"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API status
    health = call_api("/health")
    if health and health.get("status") == "ok":
        st.success("API ✓ Online")
        ch_status = health.get("clickhouse", "unknown")
        if ch_status == "ok":
            st.success("ClickHouse ✓ Connected")
        else:
            st.warning(f"ClickHouse: {ch_status}")
    else:
        st.error("API ✗ Offline")

    st.markdown("---")
    model_info = call_api("/model/info")
    if model_info and model_info.get("status") == "ok":
        st.caption(f"Model v{model_info.get('model_version', '?')}  "
                   f"· {model_info.get('feature_count', '?')} features")
    else:
        st.caption("Model: not loaded")


# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("🏥 Healthcare Recommendation System")
    st.markdown(
        "Personalised medication recommendations powered by XGBoost + "
        "collaborative filtering on 11,679 synthetic patients."
    )

    stats = fetch_population_stats()
    total = stats["total"]
    flags = stats["flags"]
    age   = stats["age"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Patients",     f"{total:,}")
    col2.metric("Mean Age",           f"{age['mean_age']:.0f} yrs")
    col3.metric("Diabetic",           f"{flags['diabetes']:,.0f}",
                delta=f"{flags['diabetes']/total*100:.1f}%")
    col4.metric("Hypertensive",       f"{flags['hypertension']:,.0f}",
                delta=f"{flags['hypertension']/total*100:.1f}%")
    col5.metric("Coronary Disease",   f"{flags['coronary_disease']:,.0f}",
                delta=f"{flags['coronary_disease']/total*100:.1f}%")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Chronic condition prevalence")
        cond_names = {
            "diabetes": "Diabetes",
            "hypertension": "Hypertension",
            "asthma": "Asthma",
            "hyperlipidemia": "Hyperlipidemia",
            "coronary_disease": "Coronary Disease",
        }
        fig, ax = plt.subplots(figsize=(6, 3.5))
        values = [flags[k] for k in cond_names]
        colors = sns.color_palette("muted", len(cond_names))
        bars   = ax.barh(list(cond_names.values()), values, color=colors)
        ax.set_xlabel("Patients")
        ax.set_title("Chronic Condition Prevalence")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + total * 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val/total*100:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Pipeline status")
        milestones = {
            "M1 — Data Ingestion":       ("✅ Complete",  "green"),
            "M2 — Feature Engineering":  ("✅ Complete",  "green"),
            "M3 — Model Training":       ("✅ Complete",  "green"),
            "M4 — API + Dashboard":      ("🔄 Active",    "blue"),
        }
        for name, (status, color) in milestones.items():
            st.markdown(
                f"<div style='padding:6px 12px;margin:4px 0;"
                f"border-left:4px solid {color};"
                f"background:#f8f9fa;border-radius:4px'>"
                f"<strong>{name}</strong> &nbsp; {status}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.subheader("Quick recommend")
        quick_pid = st.text_input("Patient ID (UUID)", key="quick_pid",
                                  placeholder="0009ed02-6244-...")
        if st.button("Get Recommendations", key="quick_btn"):
            if quick_pid.strip():
                st.session_state["selected_patient"] = quick_pid.strip()
                st.session_state["nav_to_recs"] = True
                st.rerun()


# ── Page: Patient Search ──────────────────────────────────────────────────────

elif page == "🔍 Patient Search":
    st.title("🔍 Patient Search")

    search_query = st.text_input(
        "Search by Patient ID, first name, or last name",
        placeholder="Enter patient ID or name...",
    )

    col_filters = st.columns(5)
    filter_map  = {}
    flags_ui    = ["Diabetes", "Hypertension", "Asthma", "Hyperlipidemia", "Coronary Disease"]
    flag_cols   = ["has_diabetes", "has_hypertension", "has_asthma",
                   "has_hyperlipidemia", "has_coronary_disease"]

    for i, (label, col_name) in enumerate(zip(flags_ui, flag_cols)):
        with col_filters[i]:
            val = st.selectbox(label, ["Any", "Yes", "No"], key=f"flt_{col_name}")
            if val == "Yes":
                filter_map[col_name] = 1
            elif val == "No":
                filter_map[col_name] = 0

    if search_query.strip():
        df_patients = search_patients(search_query.strip())
    else:
        # Show all with filters
        client = get_ch_client()
        where_parts = [f"{k} = {v}" for k, v in filter_map.items()]
        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        df_patients = client.query_df(f"""
            SELECT pf.patient_id, pf.age, pf.gender_encoded,
                   pf.num_conditions, pf.num_medications, pf.num_encounters,
                   pf.has_diabetes, pf.has_hypertension, pf.has_asthma,
                   pf.has_hyperlipidemia, pf.has_coronary_disease,
                   p.first, p.last, p.city
            FROM {CH_DB}.patient_features pf
            LEFT JOIN {CH_DB}.patients p ON pf.patient_id = p.patient_id
            {where}
            ORDER BY pf.patient_id
            LIMIT 100
        """)

    if df_patients.empty:
        st.info("No patients found.")
    else:
        st.caption(f"Showing {len(df_patients)} patients")

        df_display = df_patients.copy()
        df_display["gender"] = df_display["gender_encoded"].apply(gender_label)
        for flag, label in zip(flag_cols, flags_ui):
            df_display[label] = df_display[flag].apply(lambda x: "✓" if x else "")

        display_cols = ["patient_id", "first", "last", "age", "gender",
                        "city", "num_conditions", "num_medications",
                        "num_encounters"] + flags_ui

        available = [c for c in display_cols if c in df_display.columns]

        selected = st.dataframe(
            df_display[available],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="patient_table",
        )

        if (selected and
                selected.get("selection") and
                selected["selection"].get("rows")):
            row_idx = selected["selection"]["rows"][0]
            pid     = df_patients.iloc[row_idx]["patient_id"]
            st.session_state["selected_patient"] = pid

            st.success(f"Selected patient: `{pid}`")
            if st.button("View Recommendations →"):
                st.session_state["nav_to_recs"] = True
                st.rerun()


# ── Page: Recommendations ─────────────────────────────────────────────────────

elif page == "💊 Recommendations":
    st.title("💊 Medication Recommendations")

    # Patient selector
    col1, col2 = st.columns([3, 1])
    with col1:
        default_pid = st.session_state.get("selected_patient", "")
        patient_id  = st.text_input(
            "Patient ID",
            value=default_pid,
            placeholder="Enter patient UUID...",
        )
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=10)

    run_col, reload_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button("🔮 Generate Recommendations", type="primary",
                            use_container_width=True)
    with reload_col:
        if st.button("🔄 Load Stored", use_container_width=True):
            if patient_id.strip():
                st.session_state["load_stored"] = True

    if run_btn and patient_id.strip():
        with st.spinner("Running recommendation model..."):
            result = call_api(
                "/recommend",
                method="POST",
                payload={"patient_id": patient_id.strip(), "top_k": int(top_k)},
            )

        if result and result.get("status") == "ok":
            st.session_state["last_recs"]    = result
            st.session_state["last_patient"] = patient_id.strip()
        else:
            msg = result.get("message", "Unknown error") if result else "No response"
            st.error(f"Error: {msg}")

    if st.session_state.get("load_stored") and patient_id.strip():
        result = call_api(f"/recommend/{patient_id.strip()}")
        if result and result.get("status") == "ok":
            st.session_state["last_recs"]    = result
            st.session_state["last_patient"] = patient_id.strip()
        st.session_state["load_stored"] = False

    # Display results
    recs_data = st.session_state.get("last_recs")
    if recs_data:
        recs = recs_data.get("recommendations", [])
        pid  = recs_data.get("patient_id", "")

        if not recs:
            st.info("No recommendations found for this patient.")
        else:
            st.markdown(f"### Results for `{pid[:8]}...`")
            st.caption(
                f"Model v{recs_data.get('model_version', '?')} · "
                f"{len(recs)} recommendations"
            )

            # Score chart
            df_recs = pd.DataFrame(recs)
            fig, ax  = plt.subplots(figsize=(10, max(3, len(df_recs) * 0.45)))
            colors   = [
                "#2ecc71" if s > 0.8 else
                "#f39c12" if s > 0.6 else "#e74c3c"
                for s in df_recs["score"]
            ]
            bars = ax.barh(
                [f"#{r} {n[:35]}" for r, n in
                 zip(df_recs["rank"], df_recs["treatment_name"])],
                df_recs["score"],
                color=colors[::-1],
            )
            ax.set_xlabel("Recommendation Score (XGBoost probability)")
            ax.set_title("Medication Recommendation Ranking")
            ax.set_xlim(0, 1.05)
            ax.invert_yaxis()
            for bar, val in zip(bars, df_recs["score"][::-1]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
            ax.axvline(x=0.5, color="gray", linestyle="--",
                       alpha=0.4, label="0.5 threshold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Detailed table
            st.markdown("#### Detailed results")
            for rec in recs:
                score = rec["score"]
                badge = ("🟢" if score > 0.8 else
                         "🟡" if score > 0.6 else "🔴")
                with st.expander(
                    f"{badge} #{rec['rank']}  {rec['treatment_name']}  "
                    f"— score: {score:.4f}"
                ):
                    col_a, col_b = st.columns([1, 2])
                    col_a.metric("RxNorm Code",  rec["treatment_code"])
                    col_a.metric("Confidence",   f"{score:.1%}")
                    col_b.markdown(f"**Explanation:** {rec['explanation']}")

            # Patient profile if available
            st.markdown("---")
            st.markdown("#### Patient profile")
            p_data = call_api(f"/patients/{pid}")
            if p_data and p_data.get("status") == "ok":
                feats = p_data.get("features", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Age",         feats.get("age", "—"))
                c2.metric("Gender",      gender_label(feats.get("gender_encoded", -1)))
                c3.metric("Race",        race_label(feats.get("race_encoded", -1)))
                c4.metric("Conditions",  feats.get("num_conditions", "—"))
                c1.metric("Medications", feats.get("num_medications", "—"))
                c2.metric("Encounters",  feats.get("num_encounters", "—"))

                flags_active = [
                    label for label, key in zip(
                        flags_ui,
                        ["has_diabetes", "has_hypertension", "has_asthma",
                         "has_hyperlipidemia", "has_coronary_disease"]
                    ) if feats.get(key)
                ]
                if flags_active:
                    st.markdown(
                        "**Chronic conditions:** " +
                        " · ".join(f"`{f}`" for f in flags_active)
                    )

                # Known medications
                meds = p_data.get("medications", [])
                if meds:
                    st.markdown(f"**Known medications ({len(meds)}):**")
                    med_df = pd.DataFrame(meds)[
                        ["medication_code", "medication_name", "last_prescribed"]
                    ]
                    st.dataframe(med_df, use_container_width=True, height=200)


# ── Page: Analytics ───────────────────────────────────────────────────────────

elif page == "📊 Analytics":
    st.title("📊 Population Analytics")

    client = get_ch_client()
    tab1, tab2, tab3 = st.tabs([
        "Conditions", "Medications", "Recommendations"
    ])

    with tab1:
        st.subheader("Top conditions by patient count")
        df_cond = client.query_df(f"""
            SELECT description,
                   count(DISTINCT patient_id) AS patient_count
            FROM {CH_DB}.conditions
            GROUP BY description
            ORDER BY patient_count DESC
            LIMIT 20
        """)
        if not df_cond.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(df_cond["description"][::-1],
                    df_cond["patient_count"][::-1],
                    color=sns.color_palette("muted"))
            ax.set_xlabel("Unique patients")
            ax.set_title("Top 20 Conditions")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(df_cond, use_container_width=True)

    with tab2:
        st.subheader("Top medications by patient count")
        df_meds = client.query_df(f"""
            SELECT description,
                   count(DISTINCT patient_id) AS patient_count
            FROM {CH_DB}.medications
            GROUP BY description
            ORDER BY patient_count DESC
            LIMIT 20
        """)
        if not df_meds.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(df_meds["description"][::-1],
                    df_meds["patient_count"][::-1],
                    color=sns.color_palette("muted", 20))
            ax.set_xlabel("Unique patients")
            ax.set_title("Top 20 Medications")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(df_meds, use_container_width=True)

    with tab3:
        st.subheader("Stored recommendations")
        total_recs = client.query(
            f"SELECT count() FROM {CH_DB}.recommendations"
        ).result_rows[0][0]

        if total_recs == 0:
            st.info(
                "No recommendations stored yet. "
                "Use the Recommendations page to generate some."
            )
        else:
            st.metric("Total stored recommendations", f"{total_recs:,}")

            df_rec_summary = client.query_df(f"""
                SELECT
                    model_version,
                    count(DISTINCT patient_id) AS unique_patients,
                    count(*) AS total_recs,
                    round(avg(score), 4) AS avg_score
                FROM {CH_DB}.recommendations
                GROUP BY model_version
            """)
            st.dataframe(df_rec_summary, use_container_width=True)

            df_top_recs = client.query_df(f"""
                SELECT treatment_name,
                       count(*) AS times_recommended,
                       round(avg(score), 4) AS avg_score
                FROM {CH_DB}.recommendations
                GROUP BY treatment_name
                ORDER BY times_recommended DESC
                LIMIT 15
            """)
            if not df_top_recs.empty:
                st.subheader("Most frequently recommended medications")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(df_top_recs["treatment_name"][::-1],
                        df_top_recs["times_recommended"][::-1],
                        color="steelblue")
                ax.set_xlabel("Times recommended")
                ax.set_title("Top Recommended Medications")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ── Page: System ──────────────────────────────────────────────────────────────

elif page == "⚙️ System":
    st.title("⚙️ System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API")
        health = call_api("/health")
        if health:
            st.json(health)
        else:
            st.error("API unreachable")

        st.subheader("Model")
        model_info = call_api("/model/info")
        if model_info:
            st.json(model_info)

        if st.button("🔄 Reload Model from HDFS"):
            with st.spinner("Reloading..."):
                result = call_api("/model/reload", method="POST")
            if result and result.get("status") == "ok":
                st.success(f"Model reloaded: v{result.get('model_version')}")
            else:
                st.error("Reload failed")

    with col2:
        st.subheader("ClickHouse tables")
        client = get_ch_client()
        table_counts = {}
        for tbl in ["patients", "conditions", "medications",
                     "observations", "encounters", "procedures",
                     "patient_features", "recommendations"]:
            try:
                n = client.query(
                    f"SELECT count() FROM {CH_DB}.{tbl}"
                ).result_rows[0][0]
                table_counts[tbl] = n
            except Exception:
                table_counts[tbl] = "error"

        df_tables = pd.DataFrame(
            list(table_counts.items()), columns=["Table", "Row Count"]
        )
        st.dataframe(df_tables, use_container_width=True, hide_index=True)

        st.subheader("Service URLs")
        urls = {
            "Flask API":          "http://localhost:5050",
            "Streamlit":          "http://localhost:8501",
            "MLflow":             "http://localhost:5001",
            "Airflow":            "http://localhost:8081",
            "Spark Master UI":    "http://localhost:8080",
            "HDFS Namenode UI":   "http://localhost:9870",
        }
        for name, url in urls.items():
            st.markdown(f"**{name}:** [{url}]({url})")
