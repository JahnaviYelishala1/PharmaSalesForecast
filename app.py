import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import calendar
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Pharma Sales Forecast",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("pharma_model.sav", "rb"))

# -----------------------------
# Label Encoders
# -----------------------------
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
drug_labels = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

le_weekday = LabelEncoder().fit(weekday_labels)
le_drug = LabelEncoder().fit(drug_labels)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("🔧 Controls")

year = st.sidebar.number_input("Select Year", min_value=2015, max_value=2030, value=2024)
month = st.sidebar.number_input("Select Month", min_value=1, max_value=12, value=1)

st.sidebar.markdown("---")
st.sidebar.write("Choose visual components:")
show_heatmap = st.sidebar.checkbox("Show Drug Correlation Heatmap", True)
show_daily = st.sidebar.checkbox("Show Daily Trends", True)
show_weekly = st.sidebar.checkbox("Show Weekly Breakdown", True)

st.sidebar.markdown("---")
run_prediction = st.sidebar.button("🔮 Run Forecast")

# -----------------------------
# Header
# -----------------------------
st.title("💊 Pharma Sales Forecast Dashboard")
st.write(f"Forecasting sales for **all drugs** for **{calendar.month_name[month]} {year}**.")

# -----------------------------
# Prediction Logic
# -----------------------------
if run_prediction:

    with st.spinner("⏳ Fetching predictions... Please wait..."):
        
        num_days = calendar.monthrange(year, month)[1]
        df_all_daily = pd.DataFrame()
        drug_totals = {}

        for drug in drug_labels:
            drug_encoded = le_drug.transform([drug])[0]

            daily_sales = []
            for day in range(1, num_days + 1):
                date = datetime(year, month, day)
                weekday = date.strftime("%A")
                weekday_encoded = le_weekday.transform([weekday])[0]

                daily_sum = 0
                for hour in range(24):
                    X = np.array([[year, month, hour, weekday_encoded, drug_encoded]])
                    daily_sum += model.predict(X)[0]

                daily_sales.append(daily_sum)

            df_temp = pd.DataFrame({
                "Day": range(1, num_days + 1),
                "Predicted_Sales": daily_sales,
                "Drug": drug
            })

            df_all_daily = pd.concat([df_all_daily, df_temp], ignore_index=True)
            drug_totals[drug] = sum(daily_sales)

        # Monthly summary
        df_monthly = pd.DataFrame({
            "Drug": list(drug_totals.keys()),
            "Predicted_Monthly_Quantity": list(drug_totals.values())
        }).sort_values("Predicted_Monthly_Quantity", ascending=False)

    st.success("✅ Forecast completed successfully!")


    # -----------------------------
    # Create Monthly Summary Table
    # -----------------------------
    df_monthly = pd.DataFrame({
        "Drug": list(drug_totals.keys()),
        "Predicted_Monthly_Quantity": list(drug_totals.values())
    }).sort_values("Predicted_Monthly_Quantity", ascending=False)

    # -----------------------------
    # TABS for Clean Navigation
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Monthly Summary", "📈 Weekly Breakdown", "📅 Daily Forecast", "🔥 Heatmap & Downloads"])

    # -----------------------------
    # TAB 1 — Monthly Summary
    # -----------------------------
    with tab1:
        st.subheader("📊 Monthly Sales for All Drugs")
        st.dataframe(df_monthly, use_container_width=True)

        fig_monthly_bar = px.bar(
            df_monthly,
            x="Drug",
            y="Predicted_Monthly_Quantity",
            title="Monthly Predicted Sales (All Drugs)",
            color="Drug",
            text_auto=".2f",
            height=500
        )
        st.plotly_chart(fig_monthly_bar, use_container_width=True)

        fig_monthly_pie = px.pie(
            df_monthly,
            names="Drug",
            values="Predicted_Monthly_Quantity",
            hole=0.45,
            title="Market Share of Drug Consumption"
        )
        st.plotly_chart(fig_monthly_pie, use_container_width=True)

    # -----------------------------
    # TAB 2 — Weekly Breakdown
    # -----------------------------
    with tab2:
        if show_weekly:

            df_all_daily["Week"] = pd.cut(
                df_all_daily["Day"],
                bins=[0, 7, 14, 21, num_days],
                labels=[1, 2, 3, 4]
            ).astype(int)

            df_weekly = df_all_daily.groupby(["Drug", "Week"])["Predicted_Sales"].sum().reset_index()

            st.subheader("📆 Weekly Forecast (4 Fixed Weeks)")
            st.dataframe(df_weekly, use_container_width=True)

            fig_week = px.bar(
                df_weekly,
                x="Week",
                y="Predicted_Sales",
                color="Drug",
                title="Weekly Sales Breakdown for All Drugs",
                barmode="group",
                height=550
            )
            st.plotly_chart(fig_week, use_container_width=True)

    # -----------------------------
    # TAB 3 — Daily Forecast
    # -----------------------------
    with tab3:
        if show_daily:
            st.subheader("📅 Daily Sales Forecast for All Drugs")
            fig_daily = px.line(
                df_all_daily,
                x="Day",
                y="Predicted_Sales",
                color="Drug",
                title="Daily Forecast Trends",
                markers=True,
                height=600
            )
            st.plotly_chart(fig_daily, use_container_width=True)

    # -----------------------------
    # TAB 4 — Heatmap + Downloads
    # -----------------------------
    with tab4:

        if show_heatmap:
            st.subheader("🔥 Drug-Drug Correlation Heatmap")

            pivot_df = df_all_daily.pivot_table(
                index="Day",
                columns="Drug",
                values="Predicted_Sales"
            )

            corr_matrix = pivot_df.corr()

            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Between Drug Sales"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Download CSV
        csv = df_all_daily.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Daily Forecast CSV",
            data=csv,
            file_name=f"forecast_{year}_{month}.csv",
            mime="text/csv"
        )
