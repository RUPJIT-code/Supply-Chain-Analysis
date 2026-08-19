import pandas as pd
import requests
import streamlit as st

# ============================================================
# 1. CONFIG & CONSTANTS
# ============================================================

APP_TITLE = "Customer Segmentation"
APP_ICON = "📊"
API_URL = "http://127.0.0.1:8000"          # change if the backend is deployed elsewhere
REQUEST_TIMEOUT_HEALTH = 5                  # seconds
REQUEST_TIMEOUT_PREDICT = 10                # seconds

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Avg_Discount_Rate",
    "Unique_Product_Categories",
    "Total_Profit",
]

PAGES = {
    "Dashboard": "🏠",
    "Predict Customer": "🔮",
    "About Model": "🧠",
}

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ============================================================
# 2. STYLING
# ============================================================

def inject_custom_css() -> None:
    """Light visual polish on top of Streamlit's defaults."""
    st.markdown(
        """
        <style>
            /* Tighter, more deliberate spacing */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            /* Section headers */
            h1, h2, h3 {
                letter-spacing: -0.01em;
            }

            /* Metric cards */
            div[data-testid="stMetric"] {
                background: rgba(127, 127, 127, 0.06);
                border: 1px solid rgba(127, 127, 127, 0.15);
                border-radius: 10px;
                padding: 1rem 1rem 0.75rem 1rem;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                border-right: 1px solid rgba(127, 127, 127, 0.15);
            }

            .status-pill {
                display: inline-block;
                padding: 0.15rem 0.65rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            .status-pill.ok {
                background: rgba(16, 185, 129, 0.15);
                color: #059669;
            }
            .status-pill.error {
                background: rgba(239, 68, 68, 0.15);
                color: #dc2626;
            }

            .app-footer {
                text-align: center;
                color: rgba(127, 127, 127, 0.8);
                font-size: 0.8rem;
                padding-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 3. STATIC REFERENCE DATA
# ============================================================
# Descriptive text shown on the Dashboard / About pages. Actual
# predictions always come from the backend — this is only
# pre-computed summary info about the training data.

SEGMENT_SUMMARY = pd.DataFrame({
    "Cluster": [0, 1, 2, 3, 4],
    "Segment": [
        "High-Value Profitable",
        "At-Risk Valuable",
        "Recent Low-Value",
        "High-Revenue / Low-Profit",
        "Discount-Sensitive",
    ],
    "Customers": [4587, 3074, 4765, 4305, 3815],
    "Customer %": [22.33, 14.96, 23.19, 20.95, 18.57],
    "Profit Contribution %": [73.31, 11.73, 3.89, 8.32, 2.75],
})

RECOMMENDED_ACTIONS = pd.DataFrame({
    "Segment": SEGMENT_SUMMARY["Segment"],
    "Recommended Action": [
        "Retain & Reactivate — Use loyalty rewards and personalized offers.",
        "Win Back — Use targeted re-engagement and limited-time incentives.",
        "Increase Repeat Purchases — Use recommendations, cross-selling, and follow-ups.",
        "Improve Profitability — Review discounts, margins, and order economics.",
        "Reduce Discount Dependency — Use targeted promotions instead of blanket discounts."
    ],
})

FEATURE_MEANINGS = pd.DataFrame({
    "Feature": FEATURES,
    "Business Meaning": [
        "Days since last purchase",
        "Number of purchases",
        "Total customer spending",
        "Average discount received",
        "Number of product categories purchased",
        "Total profit generated",
    ],
})

FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": [0.261849, 0.209938, 0.124595, 0.168391, 0.100606, 0.134620],
}).sort_values("Importance", ascending=False)

KEY_METRICS = [
    ("Customer Segments", "5"),
    ("Largest Segment", "23.19%"),
    ("Highest Profit Share", "73.31%"),
    ("RF Accuracy", "98%"),
]

MODEL_METRICS = [
    ("Accuracy", "98%"),
    ("Cluster 0 F1", "0.96"),
    ("Cluster 2 F1", "0.99"),
    ("Cluster 4 F1", "0.99"),
]


# ============================================================
# 4. API CLIENT
# ============================================================

def check_api_health() -> tuple[bool, str]:
    """Ping the backend /health endpoint. Returns (ok, message)."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT_HEALTH)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok":
            return True, "Connected to backend"
        return False, data.get("detail", "Backend reported an error.")
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to the backend API at {API_URL}."
    except requests.exceptions.Timeout:
        return False, "The backend API timed out."
    except Exception as e:
        return False, f"Unexpected error contacting backend: {e}"


def get_prediction(payload: dict) -> tuple[bool, dict | str]:
    """Call the /predict endpoint. Returns (ok, data_or_error_message)."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_PREDICT)
        if response.status_code == 200:
            return True, response.json()
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return False, f"API error ({response.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to the backend API at {API_URL}."
    except requests.exceptions.Timeout:
        return False, "The prediction request timed out."
    except Exception as e:
        return False, f"Unexpected error: {e}"


# ============================================================
# 5. REUSABLE UI COMPONENTS
# ============================================================

def render_sidebar() -> tuple[str, bool]:
    """Renders the sidebar nav + live backend status. Returns (page, api_ok)."""
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption("RFM-based customer intelligence")

        st.divider()

        page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            format_func=lambda p: f"{PAGES[p]}  {p}",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Backend Status**")

        api_ok, api_message = check_api_health()
        pill_class = "ok" if api_ok else "error"
        pill_label = "● Online" if api_ok else "● Offline"
        st.markdown(
            f'<span class="status-pill {pill_class}">{pill_label}</span>',
            unsafe_allow_html=True,
        )
        st.caption(api_message)

        st.markdown(
            '<div class="app-footer">Powered by FastAPI + Random Forest</div>',
            unsafe_allow_html=True,
        )

    return page, api_ok


def render_metric_row(metrics: list[tuple[str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


# ============================================================
# 6. PAGE RENDERERS
# ============================================================

def render_dashboard_page() -> None:
    st.title("Customer Segmentation Dashboard")
    st.markdown(
        "This application segments customers using RFM and behavioral "
        "characteristics and provides actionable business recommendations. "
        "Predictions are served by a FastAPI backend running the trained "
        "Random Forest model."
    )

    st.divider()
    render_metric_row(KEY_METRICS)
    st.divider()

    render_section_header("Customer Segment Overview")
    st.dataframe(SEGMENT_SUMMARY, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        render_section_header("Customer Distribution")
        st.bar_chart(SEGMENT_SUMMARY.set_index("Segment")["Customer %"])
    with col2:
        render_section_header("Profit Contribution")
        st.bar_chart(SEGMENT_SUMMARY.set_index("Segment")["Profit Contribution %"])

    st.divider()

    render_section_header("Key Business Insight")
    st.info(
        "Revenue does not necessarily equal customer value.\n\n"
        "Cluster 0 generates approximately ₹2,474 average monetary value "
        "and ₹633.61 average profit. Cluster 3 generates approximately "
        "₹4,054 average monetary value but only ₹76.60 average profit.\n\n"
        "The highest-spending customers are not necessarily the most "
        "profitable ones."
    )

    render_section_header("Recommended Business Actions")
    st.dataframe(RECOMMENDED_ACTIONS, use_container_width=True, hide_index=True)


def render_prediction_form() -> dict | None:
    """Renders the input form. Returns the payload dict once submitted, else None."""
    with st.form("customer_prediction_form"):
        render_section_header("Customer Information")

        col1, col2 = st.columns(2)
        with col1:
            recency = st.number_input(
                "Recency (days)", min_value=0.0, value=100.0, step=1.0,
                help="Number of days since the customer's last purchase.",
            )
            frequency = st.number_input(
                "Frequency", min_value=1.0, value=2.0, step=1.0,
                help="Number of purchases made by the customer.",
            )
            monetary = st.number_input(
                "Monetary / Total Spending", min_value=0.0, value=500.0, step=50.0,
                help="Total monetary value generated by the customer.",
            )
        with col2:
            discount_rate = st.number_input(
                "Average Discount Rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                help="Average discount rate. Example: 10% = 0.10",
            )
            categories = st.number_input(
                "Unique Product Categories", min_value=1.0, value=2.0, step=1.0,
                help="Number of different product categories purchased.",
            )
            total_profit = st.number_input(
                "Total Profit", value=100.0, step=25.0,
                help="Total profit generated by the customer.",
            )

        submitted = st.form_submit_button("Predict Customer Segment", use_container_width=True)

    if not submitted:
        return None

    return {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "Avg_Discount_Rate": discount_rate,
        "Unique_Product_Categories": categories,
        "Total_Profit": total_profit,
    }


def render_prediction_result(payload: dict) -> None:
    with st.spinner("Getting prediction from backend..."):
        ok, result = get_prediction(payload)

    st.divider()

    if not ok:
        st.error(f"Prediction failed: {result}")
        return

    render_section_header("Prediction Result")
    st.success(f"Predicted Segment: **{result['segment_name']}**")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Cluster", result["cluster"])
    col2.metric("Model", "Random Forest")

    render_section_header("Customer Profile")
    st.write(result["description"])

    render_section_header("Recommended Business Action")
    st.info(result["recommendation"])

    render_section_header("Input Summary")
    display_input = pd.DataFrame([payload])
    display_input["Avg_Discount_Rate"] = display_input["Avg_Discount_Rate"] * 100
    display_input = display_input.rename(columns={
        "Avg_Discount_Rate": "Avg Discount (%)",
        "Unique_Product_Categories": "Product Categories",
    })
    st.dataframe(display_input, use_container_width=True, hide_index=True)


def render_predict_page(api_ok: bool) -> None:
    st.title("Predict Customer Segment")
    st.markdown(
        "Enter the customer's behavioral information below. This form calls "
        "the FastAPI backend, which applies the saved scaler and Random "
        "Forest model to predict the customer's segment and provide a "
        "recommended business action."
    )

    if not api_ok:
        st.warning(
            "The backend API is not reachable right now. Start it before "
            "making predictions (see sidebar for status)."
        )

    st.divider()

    payload = render_prediction_form()
    if payload is not None:
        render_prediction_result(payload)


def render_about_page() -> None:
    st.title("About the ML Model")
    st.markdown(
        "**Objective:** identify different customer behavior patterns and "
        "provide actionable business recommendations."
    )

    st.divider()

    render_section_header("System Architecture")
    st.code(
        "Streamlit Frontend (this app)\n"
        "          ↓  HTTP request (JSON)\n"
        "FastAPI Backend  (/predict)\n"
        "          ↓\n"
        "Load scaler.pkl → transform input\n"
        "          ↓\n"
        "Load random_forest.pkl → predict cluster\n"
        "          ↓\n"
        "Return cluster + segment + recommendation (JSON)\n"
        "          ↓\n"
        "Streamlit displays the result",
        language=None,
    )

    render_section_header("Machine Learning Pipeline")
    st.code(
        "Customer Transaction Data\n"
        "          ↓\n"
        "Feature Engineering\n"
        "          ↓\n"
        "RFM + Behavioral Features\n"
        "          ↓\n"
        "Feature Scaling\n"
        "          ↓\n"
        "K-Means Clustering\n"
        "          ↓\n"
        "5 Customer Segments\n"
        "          ↓\n"
        "Random Forest Classifier\n"
        "          ↓\n"
        "New Customer Segment Prediction\n"
        "          ↓\n"
        "Business Recommendation",
        language=None,
    )

    render_section_header("Features Used")
    st.dataframe(FEATURE_MEANINGS, use_container_width=True, hide_index=True)

    render_section_header("Random Forest Performance")
    render_metric_row(MODEL_METRICS)
    st.caption("The classifier predicts the K-Means-generated customer segments.")

    render_section_header("Feature Importance")
    st.bar_chart(FEATURE_IMPORTANCE.set_index("Feature"))

    st.info(
        "Feature importance indicates how useful each feature was to the "
        "Random Forest when distinguishing between the five K-Means-"
        "generated customer segments. It should not be interpreted as "
        "causal impact."
    )


# ============================================================
# 7. ROUTER
# ============================================================

def main() -> None:
    inject_custom_css()
    page, api_ok = render_sidebar()

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Predict Customer":
        render_predict_page(api_ok)
    elif page == "About Model":
        render_about_page()


if __name__ == "__main__":
    main()