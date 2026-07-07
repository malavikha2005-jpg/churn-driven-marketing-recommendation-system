import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

st.title("Customer Churn Marketing Analytics Dashboard")

st.write(
"""
This dashboard predicts customer churn probability and helps marketing teams
identify high-risk customers for targeted retention campaigns.
"""
)

model = joblib.load("../_models/churn_model.pkl")

uploaded_file = st.file_uploader("Upload Customer Feature Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "churn" in df.columns:
        X = df.drop(columns=["churn"])
    else:
        X = df.copy()

    df["churn_probability"] = model.predict_proba(X)[:,1]

    def risk_segment(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_segment"] = df["churn_probability"].apply(risk_segment)

    st.header("Customer Risk Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("High Risk", (df["risk_segment"]=="High Risk").sum())
    col3.metric("Medium Risk", (df["risk_segment"]=="Medium Risk").sum())
    col4.metric("Low Risk", (df["risk_segment"]=="Low Risk").sum())

    st.header("Risk Distribution")

    pie = px.pie(df, names="risk_segment",
                 title="Customer Risk Segmentation")

    st.plotly_chart(pie, use_container_width=True)

    st.header("Customer Value vs Churn Risk")

    if "total_revenue_log" in df.columns:

        scatter = px.scatter(
            df,
            x="total_revenue_log",
            y="churn_probability",
            color="risk_segment",
            title="Customer Value vs Churn Risk"
        )

        st.plotly_chart(scatter, use_container_width=True)

    st.header("Filter Customers")

    risk_filter = st.selectbox(
        "Select Risk Segment",
        ["All", "High Risk", "Medium Risk", "Low Risk"]
    )

    if risk_filter != "All":
        filtered_df = df[df["risk_segment"] == risk_filter]
    else:
        filtered_df = df

    st.dataframe(filtered_df)

    st.header("Search Customer")

    if "Customer_ID" in df.columns:

        customer_id = st.text_input("Enter Customer ID")

        if customer_id:
            result = df[df["Customer_ID"].astype(str) == customer_id]
            st.dataframe(result)

    st.header("High Risk Customers (Marketing Target List)")

    high_risk = df[df["risk_segment"] == "High Risk"]

    st.dataframe(high_risk)

    st.header("Marketing Campaign Recommendation")

    st.success(
        """
High Risk Customers → Send discount offers and loyalty rewards

Medium Risk Customers → Provide personalized promotions

Low Risk Customers → Maintain engagement with newsletters
"""
    )

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download Marketing Report",
        csv,
        "churn_marketing_report.csv",
        "text/csv"
    )