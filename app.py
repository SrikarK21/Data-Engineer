import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sqlalchemy import create_engine
from src.analysis import get_rfm_segments, get_cohort_analysis
from src.model import feature_engineering, load_data

# Config
st.set_page_config(page_title="Customer Analytics Platform", layout="wide")
DB_PATH = 'sqlite:///c:/projects/DE/New folder/customer_analytics_platform/data/customer_analytics.db'
MODEL_PATH = 'c:/projects/DE/New folder/customer_analytics_platform/data/churn_model.pkl'

@st.cache_resource
def get_db_engine():
    return create_engine(DB_PATH)

@st.cache_data
def load_rfm_data(_engine):
    return get_rfm_segments(_engine)

@st.cache_data
def load_cohort_data(_engine):
    return get_cohort_analysis(_engine)

@st.cache_resource
def load_churn_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def main():
    st.title("📊 Customer Analytics & Intelligence Platform")
    
    engine = get_db_engine()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", ["🎯 Executive Summary", "📊 Overview", "🔍 Segmentation", "🔮 Churn Predictions"])
    
    if page == "🎯 Executive Summary":
        show_executive_summary(engine)
    elif page == "📊 Overview":
        show_overview(engine)
    elif page == "🔍 Segmentation":
        show_segmentation(engine)
    elif page == "🔮 Churn Predictions":
        show_predictions(engine)

def show_executive_summary(engine):
    st.title("🎯 Executive Summary")
    st.markdown("### Your Business at a Glance")
    
    # Import additional analysis functions
    from src.analysis import get_revenue_trends, get_segment_performance
    
    # Load data
    rfm = load_rfm_data(engine)
    trends = get_revenue_trends(engine)
    segment_perf = get_segment_performance(engine)
    
    # === TOP-LINE METRICS ===
    st.markdown("---")
    st.subheader("💰 Financial Performance")
    
    total_revenue = rfm['monetary'].sum()
    total_customers = len(rfm)
    avg_clv = rfm['monetary'].mean()
    
    # Calculate trends (last vs previous period)
    latest_month_rev = trends.iloc[-1]['revenue']
    prev_month_rev = trends.iloc[-2]['revenue'] if len(trends) > 1 else latest_month_rev
    revenue_change = ((latest_month_rev - prev_month_rev) / prev_month_rev * 100) if prev_month_rev > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💵 Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{revenue_change:+.1f}% MoM",
            help="Total customer lifetime value across all segments"
        )
    
    with col2:
        st.metric(
            label="👥 Customer Base",
            value=f"{total_customers:,}",
            help="Total unique customers with purchases"
        )
    
    with col3:
        st.metric(
            label="📈 Avg Customer Value",
            value=f"${avg_clv:,.0f}",
            help="Average revenue per customer (CLV)"
        )
    
    with col4:
        # Revenue at Risk calculation
        at_risk_revenue = rfm[rfm['Segment'].isin(['At Risk', 'Lost'])]['monetary'].sum()
        risk_pct = (at_risk_revenue / total_revenue * 100)
        st.metric(
            label="⚠️ Revenue at Risk",
            value=f"${at_risk_revenue:,.0f}",
            delta=f"{risk_pct:.1f}% of total",
            delta_color="inverse",
            help="Revenue from customers likely to churn"
        )
    
    # === REVENUE TREND ===
    st.markdown("---")
    st.subheader("📊 Revenue Trend (Month-over-Month)")
    
    fig_trend = px.line(trends, x='month', y='revenue',
                        title="Monthly Revenue Performance",
                        labels={'month': 'Month', 'revenue': 'Revenue ($)'},
                        markers=True)
    fig_trend.update_traces(line_color='#00CC96', line_width=3)
    fig_trend.update_layout(hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # === SEGMENT PERFORMANCE TABLE ===
    st.markdown("---")
    st.subheader("🎯 Segment Performance Breakdown")
    
    # Format the dataframe for display
    display_perf = segment_perf.copy()
    display_perf.columns = ['Segment', 'Customers', 'Total Revenue', 'Avg CLV', 'Avg Orders', 'Avg Days Since Purchase', 'Revenue %']
    
    st.dataframe(
        display_perf.style.format({
            'Total Revenue': '${:,.0f}',
            'Avg CLV': '${:,.0f}',
            'Avg Orders': '{:.1f}',
            'Avg Days Since Purchase': '{:.0f}',
            'Revenue %': '{:.1f}%'
        }).background_gradient(subset=['Revenue %'], cmap='Greens'),
        use_container_width=True
    )
    
    # === SCENARIO MODELING ===
    st.markdown("---")
    st.subheader("💡 What-If Scenario: Churn Reduction Impact")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        churn_reduction = st.slider(
            "Reduce churn by:",
            min_value=0,
            max_value=50,
            value=10,
            step=5,
            format="%d%%",
            help="Model the financial impact of reducing customer churn"
        )
    
    with col2:
        # Calculate impact
        saved_revenue = at_risk_revenue * (churn_reduction / 100)
        new_total = total_revenue + saved_revenue
        
        st.success(f"""
        **💰 Projected Impact**:
        - **Saved Revenue**: ${saved_revenue:,.0f}
        - **New Total Revenue**: ${new_total:,.0f}
        - **ROI Opportunity**: {(saved_revenue/total_revenue*100):.1f}% revenue increase
        
        *Investing in retention campaigns targeting "At Risk" customers could unlock this value.*
        """)
    
    # === KEY RECOMMENDATIONS ===
    st.markdown("---")
    st.subheader("🚀 Strategic Recommendations")
    
    # Determine top priority based on data
    top_segment = segment_perf.iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **🏆 Protect Your Champions**
        
        {top_segment['Customer_Count']:.0f} customers generate 
        ${top_segment['Total_Revenue']:,.0f} ({top_segment['Revenue_Share']:.0f}%)
        
        → VIP loyalty program
        → Exclusive early access
        """)
    
    with col2:
        st.warning(f"""
        **⚠️ Win Back At-Risk**
        
        ${at_risk_revenue:,.0f} in jeopardy
        
        → Personalized win-back emails
        → Limited-time discount offers
        → Re-engagement campaigns
        """)
    
    with col3:
        retention_rate = trends['revenue'].iloc[-1] / trends['revenue'].iloc[0] if len(trends) > 1 else 1
        st.success(f"""
        **📈 Scale What Works**
        
        Revenue trend: {((retention_rate - 1) * 100):+.1f}%
        
        → Double down on top channels
        → Replicate Champion behaviors
        → Optimize customer journey
        """)

def show_overview(engine):
    st.header("📊 Executive Summary")
    
    st.markdown("""
    **What you're looking at**: A snapshot of your customer base health, showing who's driving revenue 
    and where opportunities lie.
    """)
    
    # KPIs
    rfm = load_rfm_data(engine)
    total_customers = len(rfm)
    total_rev = rfm['monetary'].sum()
    avg_customer_val = rfm['monetary'].mean()
    
    # Segment breakdown
    segment_counts = rfm['Segment'].value_counts()
    champions_pct = (segment_counts.get('Champions', 0) / total_customers * 100)
    at_risk_pct = (segment_counts.get('At Risk', 0) / total_customers * 100)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}", 
                help="Unique customers who made at least one purchase")
    col2.metric("Total Revenue", f"${total_rev:,.2f}",
                help="Sum of all customer spending in the dataset period")
    col3.metric("Avg Customer Value", f"${avg_customer_val:,.2f}",
                help="Average lifetime revenue per customer")
    
    # Key Insights
    st.subheader("🎯 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **Champions ({champions_pct:.1f}% of customers)**  
        These are your VIPs—they buy frequently, recently, and spend the most.  
        **Action**: Reward them with exclusive perks to maintain loyalty.
        """)
        
    with col2:
        if at_risk_pct > 10:
            st.warning(f"""
            **At Risk ({at_risk_pct:.1f}% of customers)**  
            These customers used to be active but haven't purchased recently.  
            **Action**: Launch a win-back campaign (discounts, personalized emails).
            """)
        else:
            st.info(f"""
            **At Risk ({at_risk_pct:.1f}% of customers)**  
            Small percentage of customers showing declining engagement.  
            **Action**: Monitor and engage proactively.
            """)
    
    # Charts
    st.subheader("💰 Revenue Distribution by Customer Segment")
    st.markdown("*This shows which customer groups contribute the most to your bottom line.*")
    
    segment_rev = rfm.groupby('Segment')['monetary'].sum().reset_index()
    segment_rev = segment_rev.sort_values('monetary', ascending=False)
    
    fig_pie = px.pie(segment_rev, values='monetary', names='Segment', 
                     hole=0.4, 
                     title="Where Your Revenue Comes From",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Business interpretation
    top_segment = segment_rev.iloc[0]
    st.info(f"""
    **💡 What this means**: The **{top_segment['Segment']}** segment generates 
    ${top_segment['monetary']:,.0f} ({top_segment['monetary']/total_rev*100:.1f}% of total revenue). 
    Focus your marketing budget here for maximum ROI.
    """)

def show_segmentation(engine):
    st.header("🔍 Advanced Customer Segmentation")
    
    st.markdown("""
    **What you're looking at**: Two powerful ways to understand your customers—by their behavior (RFM) 
    and by when they joined (Cohorts).
    """)
    
    tab1, tab2 = st.tabs(["RFM Analysis", "Cohort Analysis"])
    
    with tab1:
        st.markdown("""
        ### What is RFM?
        We score each customer on three dimensions:
        - **Recency**: How recently did they buy? (Lower = Better)
        - **Frequency**: How often do they buy? (Higher = Better)  
        - **Monetary**: How much do they spend? (Higher = Better)
        
        Based on these scores, customers are grouped into segments like "Champions" or "At Risk."
        """)
        
        rfm = load_rfm_data(engine)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Customer Behavior Map")
            st.caption("Each bubble is a customer. Size = Total Spending")
            fig = px.scatter(rfm, x='recency', y='frequency', size='monetary', color='Segment',
                             hover_data={'customer_id': True, 'monetary': ':$,.2f'},
                             title="Recency vs Frequency (Bubble Size = Monetary Value)",
                             labels={'recency': 'Days Since Last Purchase', 
                                    'frequency': 'Number of Purchases'},
                             color_discrete_map={
                                 'Champions': '#00CC96',
                                 'Loyal': '#636EFA',
                                 'Promising': '#FFA15A',
                                 'At Risk': '#EF553B',
                                 'Lost': '#AB63FA',
                                 'Standard': '#B6E880'
                             },
                             size_max=60)
            fig.update_xaxes(autorange="reversed")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Segment Distribution")
            counts = rfm['Segment'].value_counts()
            st.bar_chart(counts)
            
            st.info("""
            **How to use this**:
            - Focus retention efforts on "At Risk"
            - Upsell to "Loyal" customers
            - Learn from "Champions"
            """)
            
        with st.expander("📋 View Sample Customer Data"):
            st.dataframe(rfm.head(20))

    with tab2:
        st.markdown("""
        ### What is Cohort Analysis?
        This shows **retention over time**. Each row is a group of customers who made their first purchase 
        in the same month. The columns show what % of them came back in subsequent months.
        
        **Green = Good retention** | **Red = High churn**
        """)
        
        cohort = load_cohort_data(engine)
        
        fig = px.imshow(cohort, text_auto='.0%', aspect="auto", 
                        labels=dict(x="Months Since First Purchase", y="Cohort (First Purchase Month)", color="Retention %"),
                        title="Customer Retention Heatmap",
                        color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **💡 What this tells you**: If you see retention dropping sharply after Month 1, 
        it means you need a strong onboarding or "second purchase" incentive campaign.
        """)

def show_predictions(engine):
    st.header("🔮 Predictive Churn Analytics")
    
    st.markdown("""
    **What is Churn?** When a customer stops buying from you. Acquiring new customers costs 5-25x more 
    than retaining existing ones, so predicting who might leave is critical.
    
    **How it works**: Our AI model analyzes purchase patterns (recency, frequency, spending, tenure) 
    to predict which customers are likely to churn in the next 90 days.
    """)
    
    try:
        model = load_churn_model()
        transactions = load_data(engine)
        
        with st.spinner("🤖 AI is analyzing customer behavior patterns..."):
            features = feature_engineering(transactions)
            features_for_pred = features.drop(columns=['is_churn', 'customer_id'], errors='ignore')
            probs = model.predict_proba(features_for_pred)[:, 1]
            features['Churn_Probability'] = probs
            
        # Risk Segmentation
        high_risk = features[features['Churn_Probability'] > 0.6].sort_values('Churn_Probability', ascending=False)
        medium_risk = features[(features['Churn_Probability'] > 0.4) & (features['Churn_Probability'] <= 0.6)]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 High Risk (>60%)", len(high_risk), 
                   help="Customers very likely to churn—immediate action needed")
        col2.metric("🟡 Medium Risk (40-60%)", len(medium_risk),
                   help="Customers showing warning signs—proactive engagement recommended")
        col3.metric("📊 Overall Churn Risk", f"{probs.mean():.1%}",
                   help="Average churn probability across all customers")
        
        st.subheader("🚨 Priority Action List: High-Risk Customers")
        st.markdown("""
        **What to do**: Reach out to these customers NOW with personalized offers, loyalty rewards, 
        or a "We miss you" campaign.
        """)
        
        if len(high_risk) > 0:
            display_df = high_risk[['recency', 'frequency', 'monetary', 'tenure_days', 'Churn_Probability']].head(20)
            display_df.columns = ['Days Since Last Purchase', 'Total Purchases', 'Total Spent ($)', 
                                  'Customer Age (Days)', 'Churn Risk']
            
            st.dataframe(display_df.style.format({
                'Churn Risk': '{:.1%}', 
                'Total Spent ($)': '${:.2f}', 
                'Customer Age (Days)': '{:.0f}'
            }).background_gradient(subset=['Churn Risk'], cmap='Reds'))
            
            st.download_button(
                label="📥 Download High-Risk List (CSV)",
                data=high_risk.to_csv(index=False),
                file_name="high_risk_customers.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ Great news! No customers are at high risk of churning right now.")
        
        st.subheader("📊 What Drives Churn? (Model Insights)")
        st.markdown("*These factors have the biggest impact on whether a customer will churn.*")
        
        importances = pd.DataFrame({
            'Feature': features_for_pred.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Rename for clarity
        feature_names = {
            'recency': 'Days Since Last Purchase',
            'frequency': 'Purchase Frequency',
            'monetary': 'Total Spending',
            'tenure_days': 'Customer Tenure'
        }
        importances['Feature'] = importances['Feature'].map(lambda x: feature_names.get(x, x))
        
        fig = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                    title="Churn Prediction Drivers",
                    labels={'Importance': 'Impact on Prediction', 'Feature': 'Customer Attribute'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **💡 Interpretation**: If "Days Since Last Purchase" has the highest importance, 
        it means recency is your biggest churn indicator. Focus on re-engagement campaigns 
        for customers who haven't purchased in 30+ days.
        """)
        
    except Exception as e:
        st.error(f"Error loading model or generating predictions: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
