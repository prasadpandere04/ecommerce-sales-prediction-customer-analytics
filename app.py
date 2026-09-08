import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
orders = pd.read_csv(ROOT/"data/orders.csv", parse_dates=["order_date"])
customers = pd.read_csv(ROOT/"data/customers.csv")
products = pd.read_csv(ROOT/"data/products.csv")
segments = pd.read_csv(ROOT/"reports/customer_analytics.csv")

st.set_page_config(page_title="E-Commerce Intelligence", layout="wide")
st.title("🛒 E-Commerce Sales Prediction & Customer Analytics")

total_revenue = orders.revenue.sum()
total_profit = orders.profit.sum()
aov = orders.revenue.mean()
customers_count = orders.customer_id.nunique()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Total Profit", f"₹{total_profit:,.0f}")
c3.metric("Orders", f"{len(orders):,}")
c4.metric("Customers", f"{customers_count:,}")

tab1,tab2,tab3 = st.tabs(["Sales","Customers","Prediction"])

with tab1:
    monthly = orders.groupby(orders.order_date.dt.to_period("M"))["revenue"].sum()
    st.subheader("Monthly Revenue")
    st.line_chart(monthly)
    cat = orders.merge(products[["product_id","category"]],on="product_id").groupby("category").revenue.sum().sort_values(ascending=False)
    st.subheader("Revenue by Category")
    st.bar_chart(cat)

with tab2:
    st.subheader("Customer Segments")
    st.bar_chart(segments.segment.value_counts())
    st.dataframe(segments.sort_values("monetary",ascending=False).head(20), use_container_width=True)

with tab3:
    model = joblib.load(ROOT/"models/sales_forecast_model.joblib")
    latest = orders.set_index("order_date").resample("MS")["revenue"].sum()
    lag1,lag2 = latest.iloc[-1], latest.iloc[-2]
    roll = latest.iloc[-3:].mean()
    next_date = latest.index[-1] + pd.offsets.MonthBegin(1)
    X = pd.DataFrame([{"month_num":next_date.month,"year":next_date.year,"lag_1":lag1,"lag_2":lag2,"rolling_3":roll}])
    prediction = model.predict(X)[0]
    st.metric("Next Month Predicted Revenue", f"₹{prediction:,.0f}")
    st.info("Prediction uses monthly revenue lags and rolling average. Retrain the model after adding new data.")
