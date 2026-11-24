
"""
Streamlit dashboard: live status + historical statistics + search.

Run with:
    streamlit run dashboard.py

This process is independent from main.py. It reads data from MongoDB;
the camera-processing application must run separately on a machine
with access to the camera.
"""

import os
import datetime

import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Object Detection Dashboard",
    page_icon="📷",
    layout="wide",
)


# --------------------------------------------------------------------------
# Load Streamlit Cloud Secrets BEFORE importing config or database.
# Locally, config.py will load credentials from the .env file.
# --------------------------------------------------------------------------
try:
    mongo_uri = st.secrets.get("MONGO_URI")
    mongo_db_name = st.secrets.get("MONGO_DB_NAME")
    admin_password_secret = st.secrets.get("ADMIN_PASSWORD")

except Exception:
    # Streamlit secrets may not be configured during local execution.
    mongo_uri = None
    mongo_db_name = None
    admin_password_secret = None


if mongo_uri:
    os.environ["MONGO_URI"] = str(mongo_uri)

if mongo_db_name:
    os.environ["MONGO_DB_NAME"] = str(mongo_db_name)

if admin_password_secret:
    os.environ["ADMIN_PASSWORD"] = str(admin_password_secret)


# Import project modules AFTER loading cloud secrets.
import config
from database import Database


# --------------------------------------------------------------------------
# Database Connection
# --------------------------------------------------------------------------
@st.cache_resource
def get_db():
    return Database()


db = get_db()


# --------------------------------------------------------------------------
# Cached Database Functions
# --------------------------------------------------------------------------
@st.cache_data(ttl=5)
def cached_get_live_state():
    return db.get_live_state()


@st.cache_data(ttl=5)
def cached_get_statistics():
    return db.get_statistics()


@st.cache_data(ttl=5)
def cached_search_detections(limit=500):
    return db.search_detections(limit=limit)


@st.cache_data(ttl=5)
def cached_search_scenes(limit=500):
    return db.search_scenes(limit=limit)


@st.cache_data(ttl=5)
def cached_get_relationship_totals():
    return db.get_relationship_totals()


# --------------------------------------------------------------------------
# Modern Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }

        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            text-align: center;
        }

        .stButton>button {
            border-radius: 6px;
            font-weight: 500;
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 0.4rem 1rem;
        }

        .stButton>button:hover {
            background-color: #2563eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Dashboard Header
# --------------------------------------------------------------------------
st.title("📷 " + config.PROJECT_NAME)


# --------------------------------------------------------------------------
# MongoDB Connection Status
# --------------------------------------------------------------------------
if not db.connected:
    st.error(
        "Not connected to MongoDB. Please check your MongoDB Atlas "
        "credentials and Network Access settings."
    )


# --------------------------------------------------------------------------
# Dashboard Tabs
# --------------------------------------------------------------------------
tab_live, tab_history, tab_scenes, tab_relationships, tab_search, tab_admin = st.tabs(
    [
        "🔴 Live",
        "📋 Detection History",
        "🏠 Scene History",
        "🔗 Relationships",
        "🔍 Search",
        "⚙️ Admin",
    ]
)


# ==========================================================================
# 1. LIVE STATUS AND STATISTICS
# ==========================================================================
@st.fragment(run_every="5s")
def render_live_panel():
    st.subheader("Current Status")
    live = cached_get_live_state()
    if live:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Scene",
            live.get("scene") or "—",
            f'{(live.get("scene_confidence") or 0) * 100:.0f}% confidence',
        )
        col2.metric(
            "Objects Visible",
            len(live.get("objects", [])),
        )
        col3.metric(
            "FPS",
            live.get("fps", 0),
        )
        st.write("**Currently Detected Objects**")
        st.write(
            ", ".join(live.get("objects", [])) or "None"
        )
        st.write("**Objects Together**")
        together = live.get("objects_together", [])
        st.write(
            " + ".join(together) if together else "—"
        )
        st.caption(f'Last updated: {live.get("updated_at")}')
        st.caption(
            f'Device: {live.get("device", "unknown")} | '
            f'Raw detections: {live.get("raw_detection_count", 0)} | '
            f'Body pose: {"on" if live.get("pose_enabled") else "off"} | '
            f'Hand landmarks: {"on" if live.get("hands_enabled") else "off"} | '
            f'Buffered DB writes: {live.get("pending_database_writes", 0)}'
        )
        raw_classes = live.get("raw_classes", [])
        if raw_classes:
            st.write("**Raw model classes (debug view)**")
            st.write(", ".join(raw_classes))
    else:
        st.info(
            "No live data yet. Start `python main.py` on a computer "
            "connected to the camera to begin streaming detections."
        )
    # Detection Statistics
    st.subheader("Detection Statistics")
    stats = cached_get_statistics()
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Detection Sessions",
            stats.get("total_detection_sessions", 0),
        )
        c2.metric(
            "Unique Objects",
            stats.get("unique_objects", 0),
        )
        c3.metric(
            "Most Detected Object",
            stats.get("most_detected_object") or "—",
        )
        c4.metric(
            "Longest Detection (s)",
            stats.get("longest_detection_seconds", 0),
        )
        c5, c6, c7 = st.columns(3)
        c5.metric(
            "Total Scene Sessions",
            stats.get("total_scene_sessions", 0),
        )
        c6.metric(
            "Most Common Scene",
            stats.get("most_common_scene") or "—",
        )
        c7.metric(
            "Longest Scene",
            f'{stats.get("longest_scene_name") or "—"} '
            f'({stats.get("longest_scene_seconds", 0)}s)',
        )
    else:
        st.info("No statistics yet.")

with tab_live:
    render_live_panel()

# ==========================================================================
# 2. DETECTION HISTORY
# ==========================================================================
with tab_history:

    st.subheader("Detection History")

    docs = cached_search_detections(limit=500)

    if docs:

        from pdf_exporter import generate_objects_pdf

        if st.button("📄 Generate & Download Objects PDF Report"):

            pdf_path = generate_objects_pdf(db)

            if pdf_path:

                with open(pdf_path, "rb") as pdf_file:

                    st.download_button(
                        label="📥 Click here to download PDF",
                        data=pdf_file,
                        file_name="detected_objects_report.pdf",
                        mime="application/pdf",
                    )

            else:

                st.warning("No records available to export.")

        st.markdown("<br>", unsafe_allow_html=True)

        df = pd.DataFrame(docs).drop(
            columns=["_id"],
            errors="ignore",
        )

        st.dataframe(df, width="stretch")

    else:

        st.info("No detection history yet.")


# ==========================================================================
# 3. SCENE HISTORY
# ==========================================================================
with tab_scenes:

    st.subheader("Scene History")

    docs = cached_search_scenes(limit=500)

    if docs:

        df = pd.DataFrame(docs).drop(
            columns=["_id"],
            errors="ignore",
        )

        st.dataframe(df, width="stretch")

    else:

        st.info("No scene history yet.")


# ==========================================================================
# 4. OBJECT RELATIONSHIPS
# ==========================================================================
with tab_relationships:

    st.subheader("Total Time Together (all-time)")

    totals = cached_get_relationship_totals()

    if totals:

        rows = [
            {
                "Objects": " + ".join(item["_id"]),
                "Total Time (s)": round(item["total_seconds"], 1),
                "Sessions": item["sessions"],
            }
            for item in totals
        ]

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
        )

    else:

        st.info("No relationship data yet.")


    # Recent Relationship Sessions
    st.subheader("Recent Relationship Sessions")

    docs = db.search_relationships(limit=200)

    if docs:

        df = pd.DataFrame(docs).drop(
            columns=["_id"],
            errors="ignore",
        )

        if "objects" in df.columns:

            df["Objects"] = df["objects"].apply(
                lambda objects: " + ".join(objects)
            )

            df = df.drop(columns=["objects"])

        st.dataframe(df, width="stretch")


# ==========================================================================
# 5. SEARCH DETECTION HISTORY
# ==========================================================================
with tab_search:

    st.subheader("Search Detection History")

    with st.form("search_form"):

        col1, col2, col3 = st.columns(3)

        object_name = col1.text_input("Object name")

        tracking_id = col2.text_input("Tracking ID")

        date = col3.date_input(
            "Date",
            value=None,
        )

        col4, col5 = st.columns(2)

        min_duration = col4.number_input(
            "Minimum duration (s)",
            min_value=0.0,
            value=0.0,
        )

        min_confidence = col5.slider(
            "Minimum confidence",
            0.0,
            1.0,
            0.0,
        )

        submitted = st.form_submit_button("Search")


    if submitted:

        date_str = (
            date.strftime("%Y-%m-%d")
            if isinstance(date, datetime.date)
            else None
        )

        results = db.search_detections(
            object_name=object_name or None,
            tracking_id=tracking_id or None,
            date=date_str,
            min_duration=min_duration or None,
            min_confidence=min_confidence or None,
        )

        st.write(
            f"{len(results)} result(s)"
        )

        if results:

            df = pd.DataFrame(results).drop(
                columns=["_id"],
                errors="ignore",
            )

            st.dataframe(df, width="stretch")


# ==========================================================================
# 6. ADMIN PANEL
# ==========================================================================
with tab_admin:

    st.subheader("Admin")

    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_password:

        st.warning(
            "Admin functions are disabled. Configure ADMIN_PASSWORD "
            "in Streamlit Cloud Secrets or your local .env file."
        )

    else:

        entered_password = st.text_input(
            "Enter Admin Password",
            type="password",
        )

        if entered_password and entered_password == admin_password:

            st.warning(
                "This permanently deletes all detection, scene, "
                "and relationship history."
            )

            if st.button("Clear All History"):

                if db.clear_history():

                    st.success("History cleared.")

                    st.cache_data.clear()

                else:

                    st.error(
                        "Failed to clear history (DB offline?)."
                    )

        elif entered_password:

            st.error("Incorrect admin password.")

        else:

            st.info(
                "Enter the admin password to access admin functions."
            )
