import pandas as pd
from datetime import datetime

def get_rfm_segments(engine):
    """
    Calculates RFM scores and segments customers.
    Adapted for UCI Online Retail Dataset (Single Transactions Table)
    """
    query = """
    SELECT 
        customer_id,
        MAX(transaction_date) as last_date,
        COUNT(DISTINCT transaction_id) as frequency,
        SUM(amount) as monetary
    FROM transactions
    GROUP BY customer_id
    """
    rfm = pd.read_sql(query, engine)
    
    # Calculate Recency (days since last purchase)
    current_date = pd.to_datetime(rfm['last_date'].max())
    rfm['recency'] = (current_date - pd.to_datetime(rfm['last_date'])).dt.days
    
    # Drop possible outliers or negative monetary values (returns) if any slipped through
    rfm = rfm[rfm['monetary'] > 0]
    
    # Quintiles (using Quartiles 1-4)
    quantiles = rfm.quantile(q=[0.25, 0.5, 0.75], numeric_only=True)
    
    def RScore(x, p, d):
        if x <= d[p][0.25]: return 4
        elif x <= d[p][0.50]: return 3
        elif x <= d[p][0.75]: return 2
        else: return 1
        
    def FMScore(x, p, d):
        if x <= d[p][0.25]: return 1
        elif x <= d[p][0.50]: return 2
        elif x <= d[p][0.75]: return 3
        else: return 4
        
    rfm['R'] = rfm['recency'].apply(RScore, args=('recency', quantiles))
    rfm['F'] = rfm['frequency'].apply(FMScore, args=('frequency', quantiles))
    rfm['M'] = rfm['monetary'].apply(FMScore, args=('monetary', quantiles))
    
    rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
    
    def segment_customer(df):
        r = df['R']
        f = df['F']
        m = df['M']
        
        if r >= 4 and f >= 4 and m >= 4: return 'Champions'
        if r >= 3 and f >= 3: return 'Loyal'
        if r >= 3 and f <= 2: return 'Promising'
        if r <= 2 and f >= 3: return 'At Risk'
        if r <= 1 and f <= 1: return 'Lost'
        return 'Standard'
        
    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

def get_cohort_analysis(engine):
    """
    Calculates retention rates by cohort.
    """
    # Retrieve transaction data
    query = "SELECT customer_id, transaction_date FROM transactions"
    df = pd.read_sql(query, engine)
    
    # Create CohortMonth (month of first purchase)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['TransactionMonth'] = df['transaction_date'].dt.to_period('M')
    
    # Find first purchase for each customer
    df['CohortMonth'] = df.groupby('customer_id')['TransactionMonth'].transform('min')
                          
    # Calculate time offset
    def get_date_int(df, column):
        year = df[column].dt.year
        month = df[column].dt.month
        return year, month

    transaction_year, transaction_month = get_date_int(df, 'TransactionMonth')
    cohort_year, cohort_month = get_date_int(df, 'CohortMonth')
    
    years_diff = transaction_year - cohort_year
    months_diff = transaction_month - cohort_month
    
    df['CohortIndex'] = years_diff * 12 + months_diff + 1
    
    # Count unique customers in each cohort/index
    grouping = df.groupby(['CohortMonth', 'CohortIndex'])
    cohort_data = grouping['customer_id'].apply(pd.Series.nunique).reset_index()
    
    cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='customer_id')
    
    # Retention
    cohort_sizes = cohort_counts.iloc[:,0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    
    # Convert Period index to string for JSON serialization (Plotly/Streamlit)
    retention.index = retention.index.astype(str)
    
    return retention

def get_revenue_trends(engine):
    """
    Calculate month-over-month revenue trends for executive reporting.
    """
    query = """
    SELECT 
        strftime('%Y-%m', transaction_date) as month,
        SUM(amount) as revenue,
        COUNT(DISTINCT customer_id) as active_customers,
        COUNT(DISTINCT transaction_id) as orders
    FROM transactions
    GROUP BY month
    ORDER BY month
    """
    trends = pd.read_sql(query, engine)
    trends['month'] = pd.to_datetime(trends['month'])
    
    # Calculate growth rates
    trends['revenue_growth'] = trends['revenue'].pct_change() * 100
    trends['customer_growth'] = trends['active_customers'].pct_change() * 100
    
    return trends

def get_segment_performance(engine):
    """
    Detailed segment-level metrics for executive dashboards.
    """
    rfm = get_rfm_segments(engine)
    
    segment_stats = rfm.groupby('Segment').agg({
        'customer_id': 'count',
        'monetary': ['sum', 'mean'],
        'frequency': 'mean',
        'recency': 'mean'
    }).round(2)
    
    segment_stats.columns = ['Customer_Count', 'Total_Revenue', 'Avg_CLV', 'Avg_Orders', 'Avg_Recency']
    segment_stats = segment_stats.reset_index()
    
    # Calculate revenue share
    total_rev = segment_stats['Total_Revenue'].sum()
    segment_stats['Revenue_Share'] = (segment_stats['Total_Revenue'] / total_rev * 100).round(1)
    
    return segment_stats.sort_values('Total_Revenue', ascending=False)
