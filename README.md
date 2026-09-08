# E-Commerce Sales Prediction & Customer Analytics

Complete end-to-end Data Science + Data Analytics project.

## Stack
- MySQL: data storage and SQL analysis
- Python/Pandas: cleaning and EDA
- Scikit-learn: sales forecasting, customer segmentation, churn-risk model
- Power BI: dashboard layer
- Streamlit: optional interactive application

## Dataset
Synthetic Indian e-commerce data is included in `data/`:
- customers.csv
- products.csv
- orders.csv

## Run
1. Create a virtual environment.
2. Install packages:
   `pip install -r requirements.txt`
3. Run the ML/analytics pipeline:
   `python src/pipeline.py`
4. Start the optional app:
   `streamlit run app.py`

## SQL
Import `sql/schema.sql`, then load CSV data into the three tables. Use `sql/analysis_queries.sql` for analysis.

## Power BI
Import `data/orders.csv`, `data/customers.csv`, `data/products.csv`, and `reports/customer_analytics.csv`.
Recommended pages:
1. Executive Sales
2. Customer Analytics
3. Product Analytics
4. Sales Prediction

## Important
The included dataset is synthetic and intended for academic/demo use. Replace it with real business data for production use.
