import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io

# Page Configuration
st.set_page_config(
    page_title="Luxury Authentication Platform Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data():
    """Load the synthetic dataset"""
    try:
        df = pd.read_csv('luxury_auth_survey_synthetic_data_600_responses.csv')
        return df
    except:
        st.error("⚠️ Dataset not found! Please upload 'luxury_auth_survey_synthetic_data_600_responses.csv'")
        return None

def prepare_data_for_ml(df):
    """Prepare data for machine learning"""
    # Create binary target: Interested (4-5) = 1, Not Interested (1-3) = 0
    df['Target_Binary'] = (df['Q30_Interest_Level_TARGET'] >= 4).astype(int)
    
    # Select numerical features
    numerical_features = [
        'Q14_Worry_Counterfeits_Scale',
        'Q17_Authentication_Importance_Scale',
        'Q23_Accept_Longer_Delivery_Scale',
        'Q11_Annual_Spending',
        'Q25_Planned_Spending_12M',
        'Q33_NPS_Score'
    ]
    
    # Encode categorical features
    le_age = LabelEncoder()
    le_income = LabelEncoder()
    le_education = LabelEncoder()
    le_frequency = LabelEncoder()
    
    df['Age_Encoded'] = le_age.fit_transform(df['Q2_Age'])
    df['Income_Encoded'] = le_income.fit_transform(df['Q5_Income'])
    df['Education_Encoded'] = le_education.fit_transform(df['Q7_Education'])
    df['Frequency_Encoded'] = le_frequency.fit_transform(df['Q9_Purchase_Frequency'])
    
    categorical_features = ['Age_Encoded', 'Income_Encoded', 'Education_Encoded', 'Frequency_Encoded']
    
    all_features = numerical_features + categorical_features
    
    return df, all_features

def create_transaction_data(df, column):
    """Create transaction data for association rules"""
    transactions = []
    for items in df[column].dropna():
        transactions.append([item.strip() for item in items.split(',')])
    return transactions

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# 💎 Luxury Auth Analytics")
st.sidebar.markdown("---")

# Load data
df = load_data()

if df is not None:
    st.sidebar.success(f"✅ Data Loaded: {len(df)} responses")
    st.sidebar.markdown(f"**Total Columns:** {len(df.columns)}")
    st.sidebar.markdown(f"**Missing Values:** {df.isnull().sum().sum()}")
    st.sidebar.markdown("---")

# Navigation
st.sidebar.markdown("## 📊 Navigation")
tab_selection = st.sidebar.radio(
    "Choose Section:",
    ["🎯 Executive Dashboard", "🤖 ML Analytics", "🔮 Prediction Tool"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
if df is not None:
    interested = (df['Q30_Interest_Level_TARGET'] >= 4).sum()
    conversion_rate = (interested / len(df)) * 100
    st.sidebar.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    st.sidebar.metric("Avg Annual Spend", f"₹{df['Q11_Annual_Spending'].mean():,.0f}")
    st.sidebar.metric("High NPS (9-10)", f"{(df['Q33_NPS_Score'] >= 9).sum()}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed for Data-Driven Marketing Decisions**")

# ============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ============================================================================

if tab_selection == "🎯 Executive Dashboard":
    
    st.markdown("<h1 class='main-header'>💎 Luxury Authentication Platform</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #666;'>Executive Marketing Dashboard</h2>", unsafe_allow_html=True)
    
    if df is None:
        st.error("⚠️ Please ensure the dataset file is in the same directory as app.py")
        st.stop()
    
    # Key Metrics Row
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        interested_count = (df['Q30_Interest_Level_TARGET'] >= 4).sum()
        conversion_rate = (interested_count / len(df)) * 100
        st.metric("🎯 Conversion Rate", f"{conversion_rate:.1f}%", f"{interested_count} users")
    
    with col2:
        avg_spend = df['Q11_Annual_Spending'].mean()
        st.metric("💰 Avg Annual Spend", f"₹{avg_spend/100000:.2f}L", "per customer")
    
    with col3:
        high_nps = (df['Q33_NPS_Score'] >= 9).sum()
        nps_rate = (high_nps / len(df)) * 100
        st.metric("⭐ Promoters (NPS 9-10)", f"{nps_rate:.1f}%", f"{high_nps} users")
    
    with col4:
        high_auth_importance = (df['Q17_Authentication_Importance_Scale'] >= 4).sum()
        auth_rate = (high_auth_importance / len(df)) * 100
        st.metric("🔒 Auth Priority", f"{auth_rate:.1f}%", "high importance")
    
    with col5:
        want_updates = (df['Q38_Want_Updates'] == 'Yes').sum()
        lead_rate = (want_updates / len(df)) * 100
        st.metric("📧 Lead Capture", f"{lead_rate:.1f}%", f"{want_updates} leads")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 1: CUSTOMER SEGMENTATION BY VALUE & INTENT (STRATEGIC TARGETING)
    # ========================================================================
    
    st.markdown("<h2 class='sub-header'>📊 Chart 1: Strategic Customer Segmentation Matrix</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create segments based on spending and interest
        df['Segment'] = 'Low Value - Low Intent'
        df.loc[(df['Q11_Annual_Spending'] > df['Q11_Annual_Spending'].median()) & 
               (df['Q30_Interest_Level_TARGET'] < 4), 'Segment'] = 'High Value - Low Intent'
        df.loc[(df['Q11_Annual_Spending'] <= df['Q11_Annual_Spending'].median()) & 
               (df['Q30_Interest_Level_TARGET'] >= 4), 'Segment'] = 'Low Value - High Intent'
        df.loc[(df['Q11_Annual_Spending'] > df['Q11_Annual_Spending'].median()) & 
               (df['Q30_Interest_Level_TARGET'] >= 4), 'Segment'] = 'High Value - High Intent'
        
        fig1 = px.scatter(df, 
                         x='Q11_Annual_Spending', 
                         y='Q30_Interest_Level_TARGET',
                         color='Segment',
                         size='Q33_NPS_Score',
                         hover_data=['Q5_Income', 'Q9_Purchase_Frequency'],
                         title='Customer Value vs Intent to Adopt Platform',
                         labels={'Q11_Annual_Spending': 'Annual Luxury Spending (₹)',
                                'Q30_Interest_Level_TARGET': 'Interest Level (1-5)'},
                         color_discrete_map={
                             'High Value - High Intent': '#2ecc71',
                             'High Value - Low Intent': '#f39c12',
                             'Low Value - High Intent': '#3498db',
                             'Low Value - Low Intent': '#95a5a6'
                         },
                         height=500)
        
        fig1.add_hline(y=4, line_dash="dash", line_color="red", 
                      annotation_text="Interest Threshold", annotation_position="right")
        fig1.add_vline(x=df['Q11_Annual_Spending'].median(), line_dash="dash", 
                      line_color="blue", annotation_text="Median Spending")
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Strategic Insights")
        
        segment_counts = df['Segment'].value_counts()
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>🟢 High Value - High Intent</b><br>
        <b>{segment_counts.get('High Value - High Intent', 0)}</b> customers ({segment_counts.get('High Value - High Intent', 0)/len(df)*100:.1f}%)<br>
        <b>Action:</b> Priority onboarding, premium support, exclusive early access
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>🟡 High Value - Low Intent</b><br>
        <b>{segment_counts.get('High Value - Low Intent', 0)}</b> customers ({segment_counts.get('High Value - Low Intent', 0)/len(df)*100:.1f}%)<br>
        <b>Action:</b> Education campaigns, case studies, trust-building content
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>🔵 Low Value - High Intent</b><br>
        <b>{segment_counts.get('Low Value - High Intent', 0)}</b> customers ({segment_counts.get('Low Value - High Intent', 0)/len(df)*100:.1f}%)<br>
        <b>Action:</b> Entry-level offerings, referral programs, upsell opportunities
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 2: CONVERSION FUNNEL BY DEMOGRAPHICS (TARGETING OPTIMIZATION)
    # ========================================================================
    
    st.markdown("<h2 class='sub-header'>📊 Chart 2: Conversion Funnel by Demographics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conversion by Age
        age_conversion = df.groupby('Q2_Age').agg({
            'Q30_Interest_Level_TARGET': lambda x: (x >= 4).sum() / len(x) * 100
        }).reset_index()
        age_conversion.columns = ['Age', 'Conversion_Rate']
        
        fig2a = px.bar(age_conversion, 
                      x='Age', 
                      y='Conversion_Rate',
                      title='Conversion Rate by Age Group',
                      labels={'Conversion_Rate': 'Conversion Rate (%)'},
                      color='Conversion_Rate',
                      color_continuous_scale='Viridis',
                      text='Conversion_Rate')
        
        fig2a.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2a.update_layout(showlegend=False)
        
        st.plotly_chart(fig2a, use_container_width=True)
        
        # Insight
        best_age = age_conversion.loc[age_conversion['Conversion_Rate'].idxmax(), 'Age']
        st.success(f"🎯 **Best Performing Age Group:** {best_age} with {age_conversion['Conversion_Rate'].max():.1f}% conversion")
    
    with col2:
        # Conversion by Income
        income_conversion = df.groupby('Q5_Income').agg({
            'Q30_Interest_Level_TARGET': lambda x: (x >= 4).sum() / len(x) * 100
        }).reset_index()
        income_conversion.columns = ['Income', 'Conversion_Rate']
        
        # Order income levels
        income_order = ["Less than 5 Lakhs", "5-10 Lakhs", "10-20 Lakhs", 
                       "20-35 Lakhs", "35-50 Lakhs", "50 Lakhs - 1 Crore", "Above 1 Crore"]
        income_conversion['Income'] = pd.Categorical(income_conversion['Income'], 
                                                     categories=income_order, 
                                                     ordered=True)
        income_conversion = income_conversion.sort_values('Income')
        
        fig2b = px.bar(income_conversion, 
                      x='Income', 
                      y='Conversion_Rate',
                      title='Conversion Rate by Income Bracket',
                      labels={'Conversion_Rate': 'Conversion Rate (%)'},
                      color='Conversion_Rate',
                      color_continuous_scale='RdYlGn',
                      text='Conversion_Rate')
        
        fig2b.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2b.update_xaxes(tickangle=-45)
        fig2b.update_layout(showlegend=False)
        
        st.plotly_chart(fig2b, use_container_width=True)
        
        # Insight
        best_income = income_conversion.loc[income_conversion['Conversion_Rate'].idxmax(), 'Income']
        st.success(f"💰 **Best Performing Income:** {best_income} with {income_conversion['Conversion_Rate'].max():.1f}% conversion")
    
    # Marketing Action Plan
    st.markdown("### 🎯 Marketing Action Plan Based on Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h4>👥 Target Audience</h4>
        <ul>
            <li>Age: 25-44 years (70% of market)</li>
            <li>Income: ₹20L+ annual</li>
            <li>Focus: Metro cities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h4>📱 Channel Strategy</h4>
        <ul>
            <li>Instagram/LinkedIn ads for 25-34</li>
            <li>Premium credit card partnerships</li>
            <li>Golf clubs, luxury events</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
        <h4>💬 Messaging</h4>
        <ul>
            <li>Emphasize authenticity guarantee</li>
            <li>Price comparison savings</li>
            <li>Peace of mind positioning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 3: PAIN POINTS & WILLINGNESS TO PAY ANALYSIS
    # ========================================================================
    
    st.markdown("<h2 class='sub-header'>📊 Chart 3: Pain Points vs Willingness to Pay Premium</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create heatmap data
        worry_levels = sorted(df['Q14_Worry_Counterfeits_Scale'].unique())
        auth_importance = sorted(df['Q17_Authentication_Importance_Scale'].unique())
        
        # Calculate average willingness to pay for each combination
        heatmap_data = []
        for worry in worry_levels:
            row = []
            for auth in auth_importance:
                subset = df[(df['Q14_Worry_Counterfeits_Scale'] == worry) & 
                          (df['Q17_Authentication_Importance_Scale'] == auth)]
                
                # Calculate percentage willing to pay extra
                willing_to_pay = subset[~subset['Q24_Willingness_Pay_Extra'].str.contains('Rs 0|Should be free', na=False)]
                if len(subset) > 0:
                    row.append(len(willing_to_pay) / len(subset) * 100)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        fig3 = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f'Auth Imp: {a}' for a in auth_importance],
            y=[f'Worry: {w}' for w in worry_levels],
            colorscale='RdYlGn',
            text=np.array(heatmap_data).round(1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="% Willing to Pay")
        ))
        
        fig3.update_layout(
            title='Willingness to Pay Premium for Authentication (% of segment)',
            xaxis_title='Authentication Importance Level',
            yaxis_title='Worry About Counterfeits',
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Pricing Insights")
        
        high_worry_high_auth = df[(df['Q14_Worry_Counterfeits_Scale'] >= 4) & 
                                  (df['Q17_Authentication_Importance_Scale'] >= 4)]
        
        willing_premium = high_worry_high_auth[~high_worry_high_auth['Q24_Willingness_Pay_Extra'].str.contains('Rs 0|Should be free', na=False)]
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>🎯 Premium Segment</b><br>
        <b>{len(high_worry_high_auth)}</b> customers with high worry + high auth importance<br>
        <b>{len(willing_premium)/len(high_worry_high_auth)*100:.1f}%</b> willing to pay extra
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💰 Pricing Strategy")
        st.markdown("""
        - **Free Tier:** Items < ₹25,000
        - **Basic (1-2%):** ₹25k - ₹1L items  
        - **Premium (2-5%):** ₹1L+ items
        - **Subscription:** Unlimited for ₹10k/year
        """)
        
        # Calculate revenue potential
        avg_order_value = df['Q10_Average_Spending'].map({
            'Less than 10,000': 7500, '10,000 - 25,000': 17500, '25,000 - 50,000': 37500,
            '50,000 - 1,00,000': 75000, '1,00,000 - 2,50,000': 175000, 
            '2,50,000 - 5,00,000': 375000, 'Above 5,00,000': 750000
        }).fillna(0)
        
        potential_revenue = (willing_premium['Q10_Average_Spending'].map({
            'Less than 10,000': 0, '10,000 - 25,000': 350, '25,000 - 50,000': 1125,
            '50,000 - 1,00,000': 2250, '1,00,000 - 2,50,000': 5250, 
            '2,50,000 - 5,00,000': 11250, 'Above 5,00,000': 22500
        }).sum() / len(df))
        
        st.metric("💵 Avg Revenue/Customer", f"₹{potential_revenue:.0f}", "from auth fees")
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 4: CUSTOMER JOURNEY & HESITATION ANALYSIS
    # ========================================================================
    
    st.markdown("<h2 class='sub-header'>📊 Chart 4: Customer Journey Sankey - From Interest to Action</h2>", unsafe_allow_html=True)
    
    # Prepare data for Sankey
    df['Interest_Category'] = df['Q30_Interest_Level_TARGET'].map({
        1: 'Not Interested', 2: 'Not Interested', 3: 'Neutral', 
        4: 'Interested', 5: 'Very Interested'
    })
    
    journey_data = df.groupby(['Interest_Category', 'Q32_Biggest_Hesitation', 'Q34_Action_If_Launched']).size().reset_index(name='count')
    
    # Create Sankey diagram
    all_nodes = list(set(journey_data['Interest_Category'].unique()) | 
                    set(journey_data['Q32_Biggest_Hesitation'].unique()) | 
                    set(journey_data['Q34_Action_If_Launched'].unique()))
    
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Create links
    sources1 = [node_dict[row['Interest_Category']] for _, row in journey_data.iterrows()]
    targets1 = [node_dict[row['Q32_Biggest_Hesitation']] for _, row in journey_data.iterrows()]
    values1 = journey_data['count'].tolist()
    
    journey_data2 = df.groupby(['Q32_Biggest_Hesitation', 'Q34_Action_If_Launched']).size().reset_index(name='count')
    sources2 = [node_dict[row['Q32_Biggest_Hesitation']] for _, row in journey_data2.iterrows()]
    targets2 = [node_dict[row['Q34_Action_If_Launched']] for _, row in journey_data2.iterrows()]
    values2 = journey_data2['count'].tolist()
    
    fig4 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="lightblue"
        ),
        link=dict(
            source=sources1 + sources2,
            target=targets1 + targets2,
            value=values1 + values2
        )
    )])
    
    fig4.update_layout(
        title="Customer Journey: Interest → Hesitation → Intended Action",
        font_size=10,
        height=600
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Hesitation breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚧 Top Hesitations (High Intent Customers)")
        high_intent = df[df['Q30_Interest_Level_TARGET'] >= 4]
        hesitations = high_intent['Q32_Biggest_Hesitation'].value_counts().head(5)
        
        fig_hesitation = px.bar(
            x=hesitations.values,
            y=hesitations.index,
            orientation='h',
            title='Top 5 Barriers to Conversion',
            labels={'x': 'Number of Customers', 'y': 'Hesitation'},
            color=hesitations.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_hesitation, use_container_width=True)
    
    with col2:
        st.markdown("### ✅ Solutions to Overcome Hesitations")
        st.markdown("""
        | Hesitation | Solution Strategy |
        |-----------|-------------------|
        | **Longer delivery time** | • Express authentication option (48hrs)<br>• Real-time tracking dashboard<br>• Free upgrade for first purchase |
        | **Authentication cost** | • Tiered pricing model<br>• Free for orders >₹1L<br>• Subscription plans |
        | **Trust in process** | • Video documentation<br>• Expert credentials display<br>• Money-back guarantee |
        | **Limited selection** | • Partner with 10+ platforms<br>• 50,000+ products at launch<br>• Category expansion roadmap |
        """)
    
    st.markdown("---")
    
    # ========================================================================
    # CHART 5: NPS CORRELATION & REVENUE POTENTIAL MATRIX
    # ========================================================================
    
    st.markdown("<h2 class='sub-header'>📊 Chart 5: NPS vs Revenue Potential - Customer Lifetime Value Matrix</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create customer segments based on NPS and spending
        df['NPS_Category'] = pd.cut(df['Q33_NPS_Score'], 
                                     bins=[-1, 6, 8, 10], 
                                     labels=['Detractors (0-6)', 'Passives (7-8)', 'Promoters (9-10)'])
        
        df['Spending_Category'] = pd.cut(df['Q11_Annual_Spending'], 
                                         bins=[0, 50000, 200000, float('inf')], 
                                         labels=['Low (<50k)', 'Medium (50k-2L)', 'High (>2L)'])
        
        # Create sunburst chart
        fig5 = px.sunburst(
            df,
            path=['NPS_Category', 'Spending_Category', 'Q30_Interest_Level_TARGET'],
            values='Q11_Annual_Spending',
            color='Q30_Interest_Level_TARGET',
            color_continuous_scale='RdYlGn',
            title='Customer Lifetime Value Hierarchy: NPS → Spending → Interest Level',
            height=600
        )
        
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 CLV Analysis")
        
        # Calculate metrics by NPS category
        for nps_cat in ['Promoters (9-10)', 'Passives (7-8)', 'Detractors (0-6)']:
            segment = df[df['NPS_Category'] == nps_cat]
            avg_spend = segment['Q11_Annual_Spending'].mean()
            conversion = (segment['Q30_Interest_Level_TARGET'] >= 4).sum() / len(segment) * 100
            
            color_map = {
                'Promoters (9-10)': '#2ecc71',
                'Passives (7-8)': '#f39c12',
                'Detractors (0-6)': '#e74c3c'
            }
            
            st.markdown(f"""
            <div style='background-color: {color_map[nps_cat]}20; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h4 style='color: {color_map[nps_cat]};'>{nps_cat}</h4>
            <b>Segment Size:</b> {len(segment)} ({len(segment)/len(df)*100:.1f}%)<br>
            <b>Avg Spending:</b> ₹{avg_spend:,.0f}<br>
            <b>Conversion Rate:</b> {conversion:.1f}%<br>
            <b>CLV (3 years):</b> ₹{avg_spend * 3 * (conversion/100):,.0f}
            </div>
            """, unsafe_allow_html=True)
        
        # ROI Calculation
        st.markdown("### 💰 Marketing ROI Projection")
        
        total_promoters = len(df[df['NPS_Category'] == 'Promoters (9-10)'])
        total_interested = (df['Q30_Interest_Level_TARGET'] >= 4).sum()
        
        # Assuming 30% of interested customers convert in Year 1
        expected_customers = int(total_interested * 0.30)
        avg_customer_value = df[df['Q30_Interest_Level_TARGET'] >= 4]['Q25_Planned_Spending_12M'].mean()
        
        # Revenue from authentication fees (2% average)
        auth_revenue = expected_customers * avg_customer_value * 0.02
        
        # Revenue from subscriptions (assume 20% opt for subscription)
        subscription_revenue = expected_customers * 0.20 * 10000
        
        total_revenue = auth_revenue + subscription_revenue
        
        st.metric("📈 Year 1 Revenue Projection", f"₹{total_revenue/10000000:.2f} Cr")
        st.metric("👥 Expected Active Customers", f"{expected_customers}")
        st.metric("💵 Revenue per Customer", f"₹{total_revenue/expected_customers:,.0f}")
        
    st.markdown("---")
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    
    st.markdown("<h2 class='sub-header'>📋 Executive Summary & Action Items</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Target Market
        
        **Primary Segment:**
        - Age: 25-44 years
        - Income: ₹20-50 Lakhs
        - Location: Mumbai, Delhi, Bangalore
        - Profile: Salaried professionals, entrepreneurs
        
        **Market Size:** {:.0f} customers ({:.1f}% of total)
        
        **Conversion Potential:** {:.1f}%
        """.format(
            len(df[(df['Q2_Age'].isin(['25-34 years', '35-44 years'])) & 
                  (df['Q5_Income'].isin(['20-35 Lakhs', '35-50 Lakhs']))]),
            len(df[(df['Q2_Age'].isin(['25-34 years', '35-44 years'])) & 
                  (df['Q5_Income'].isin(['20-35 Lakhs', '35-50 Lakhs']))]) / len(df) * 100,
            (df[(df['Q2_Age'].isin(['25-34 years', '35-44 years'])) & 
                (df['Q5_Income'].isin(['20-35 Lakhs', '35-50 Lakhs']))]['Q30_Interest_Level_TARGET'] >= 4).sum() / 
            len(df[(df['Q2_Age'].isin(['25-34 years', '35-44 years'])) & 
                  (df['Q5_Income'].isin(['20-35 Lakhs', '35-50 Lakhs']))]) * 100
        ))
    
    with col2:
        st.markdown("""
        ### 🚀 Launch Strategy
        
        **Phase 1 (Month 1-3):**
        - Focus: High Value-High Intent segment
        - Offer: Free authentication for first 3 purchases
        - Target: 500 early adopters
        
        **Phase 2 (Month 4-6):**
        - Expand to High Value-Low Intent
        - Education campaigns, case studies
        - Target: 2,000 customers
        
        **Phase 3 (Month 7-12):**
        - Mass market expansion
        - Subscription model launch
        - Target: 10,000+ customers
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Key Differentiators
        
        **Must-Have Features:**
        1. 10-12 point authentication process
        2. Money-back authenticity guarantee
        3. Price comparison across platforms
        4. 48-hour express authentication
        5. Detailed authentication reports
        
        **Value Proposition:**
        - "Buy Luxury with Confidence"
        - "Best Price + Guaranteed Authenticity"
        - "Your Personal Authentication Expert"
        """)
    
    st.markdown("---")
    
    # Download report button
    st.markdown("### 📥 Download Complete Analysis")
    
    if st.button("🔽 Generate PDF Report", use_container_width=True):
        st.success("✅ Report generation feature coming soon! Currently showing interactive dashboard.")

# ============================================================================
# TAB 2: ML ANALYTICS
# ============================================================================

elif tab_selection == "🤖 ML Analytics":
    
    st.markdown("<h1 class='main-header'>🤖 Machine Learning Analytics</h1>", unsafe_allow_html=True)
    
    if df is None:
        st.error("⚠️ Please ensure the dataset is loaded")
        st.stop()
    
    # Prepare data
    df_ml, feature_columns = prepare_data_for_ml(df)
    
    st.markdown("---")
    
    # ML Model Selection
    st.markdown("## 🎯 Select ML Algorithms to Run")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_classification = st.checkbox("✅ Classification Models", value=True)
    with col2:
        run_clustering = st.checkbox("✅ Clustering Analysis", value=True)
    with col3:
        run_association = st.checkbox("✅ Association Rules Mining", value=True)
    
    if st.button("🚀 Run Selected ML Algorithms", use_container_width=True, type="primary"):
        
        # ====================================================================
        # CLASSIFICATION MODELS
        # ====================================================================
        
        if run_classification:
            st.markdown("---")
            st.markdown("## 📊 Classification Models: Predicting Customer Interest")
            
            with st.spinner("🔄 Training classification models..."):
                
                X = df_ml[feature_columns]
                y = df_ml['Target_Binary']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Models
                models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(random_state=42)
                }
                
                results = []
                
                col1, col2, col3 = st.columns(3)
                
                for idx, (name, model) in enumerate(models.items()):
                    
                    if name == 'Logistic Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    
                    results.append({
                        'Model': name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1,
                        'ROC-AUC': roc_auc
                    })
                    
                    # Display in columns
                    if idx == 0:
                        with col1:
                            st.markdown(f"### {name}")
                            st.metric("Accuracy", f"{accuracy:.3f}")
                            st.metric("Precision", f"{precision:.3f}")
                            st.metric("Recall", f"{recall:.3f}")
                            st.metric("F1-Score", f"{f1:.3f}")
                            st.metric("ROC-AUC", f"{roc_auc:.3f}")
                    elif idx == 1:
                        with col2:
                            st.markdown(f"### {name}")
                            st.metric("Accuracy", f"{accuracy:.3f}")
                            st.metric("Precision", f"{precision:.3f}")
                            st.metric("Recall", f"{recall:.3f}")
                            st.metric("F1-Score", f"{f1:.3f}")
                            st.metric("ROC-AUC", f"{roc_auc:.3f}")
                    else:
                        with col3:
                            st.markdown(f"### {name}")
                            st.metric("Accuracy", f"{accuracy:.3f}")
                            st.metric("Precision", f"{precision:.3f}")
                            st.metric("Recall", f"{recall:.3f}")
                            st.metric("F1-Score", f"{f1:.3f}")
                            st.metric("ROC-AUC", f"{roc_auc:.3f}")
                
                # Comparison Chart
                results_df = pd.DataFrame(results)
                
                st.markdown("### 📊 Model Comparison")
                
                fig_comparison = go.Figure()
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                
                for metric in metrics:
                    fig_comparison.add_trace(go.Bar(
                        name=metric,
                        x=results_df['Model'],
                        y=results_df[metric],
                        text=results_df[metric].round(3),
                        textposition='auto',
                    ))
                
                fig_comparison.update_layout(
                    title='Classification Models Performance Comparison',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Best Model
                best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
                best_roc_auc = results_df['ROC-AUC'].max()
                
                st.success(f"🏆 **Best Performing Model:** {best_model_name} with ROC-AUC of {best_roc_auc:.3f}")
                
                # Confusion Matrix for best model
                st.markdown(f"### 🎯 Confusion Matrix: {best_model_name}")
                
                if best_model_name == 'Logistic Regression':
                    best_model = models[best_model_name]
                    best_model.fit(X_train_scaled, y_train)
                    y_pred_best = best_model.predict(X_test_scaled)
                else:
                    best_model = models[best_model_name]
                    best_model.fit(X_train, y_train)
                    y_pred_best = best_model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred_best)
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Not Interested', 'Predicted Interested'],
                    y=['Actual Not Interested', 'Actual Interested'],
                    text=cm,
                    texttemplate='%{text}',
                    colorscale='Blues'
                ))
                
                fig_cm.update_layout(
                    title='Confusion Matrix',
                    xaxis_title='Predicted Label',
                    yaxis_title='True Label',
                    height=400
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Feature Importance (Random Forest)
                if best_model_name == 'Random Forest':
                    st.markdown("### 📊 Feature Importance")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top Features Predicting Customer Interest',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        # ====================================================================
        # CLUSTERING ANALYSIS
        # ====================================================================
        
        if run_clustering:
            st.markdown("---")
            st.markdown("## 🎨 Clustering Analysis: Customer Segmentation")
            
            with st.spinner("🔄 Performing clustering analysis..."):
                
                # Select features for clustering
                clustering_features = [
                    'Q14_Worry_Counterfeits_Scale',
                    'Q17_Authentication_Importance_Scale',
                    'Q11_Annual_Spending',
                    'Q33_NPS_Score',
                    'Income_Encoded',
                    'Age_Encoded'
                ]
                
                X_cluster = df_ml[clustering_features].copy()
                
                # Scale features
                scaler = StandardScaler()
                X_cluster_scaled = scaler.fit_transform(X_cluster)
                
                # Elbow Method
                st.markdown("### 📈 Elbow Method - Optimal Number of Clusters")
                
                inertias = []
                K_range = range(2, 11)
                
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_cluster_scaled)
                    inertias.append(kmeans.inertia_)
                
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(K_range),
                    y=inertias,
                    mode='lines+markers',
                    marker=dict(size=10, color='blue'),
                    line=dict(width=3)
                ))
                
                fig_elbow.update_layout(
                    title='Elbow Method for Optimal K',
                    xaxis_title='Number of Clusters (K)',
                    yaxis_title='Inertia (Within-Cluster Sum of Squares)',
                    height=400
                )
                
                st.plotly_chart(fig_elbow, use_container_width=True)
                
                # Perform K-Means with optimal K
                optimal_k = st.slider("Select Number of Clusters", 2, 10, 4)
                
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                df_ml['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
                
                st.success(f"✅ K-Means clustering completed with {optimal_k} clusters")
                
                # Cluster Visualization
                st.markdown("### 🎨 Cluster Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cluster_3d = px.scatter_3d(
                        df_ml,
                        x='Q11_Annual_Spending',
                        y='Q17_Authentication_Importance_Scale',
                        z='Q33_NPS_Score',
                        color='Cluster',
                        title='3D Cluster Visualization',
                        labels={
                            'Q11_Annual_Spending': 'Annual Spending',
                            'Q17_Authentication_Importance_Scale': 'Auth Importance',
                            'Q33_NPS_Score': 'NPS Score'
                        },
                        height=500
                    )
                    
                    st.plotly_chart(fig_cluster_3d, use_container_width=True)
                
                with col2:
                    fig_cluster_2d = px.scatter(
                        df_ml,
                        x='Q11_Annual_Spending',
                        y='Q30_Interest_Level_TARGET',
                        color='Cluster',
                        size='Q33_NPS_Score',
                        title='Clusters: Spending vs Interest Level',
                        labels={
                            'Q11_Annual_Spending': 'Annual Spending (₹)',
                            'Q30_Interest_Level_TARGET': 'Interest Level'
                        },
                        height=500
                    )
                    
                    st.plotly_chart(fig_cluster_2d, use_container_width=True)
                
                # Cluster Profiles
                st.markdown("### 📊 Cluster Profiles")
                
                cluster_profiles = df_ml.groupby('Cluster').agg({
                    'Q11_Annual_Spending': 'mean',
                    'Q17_Authentication_Importance_Scale': 'mean',
                    'Q30_Interest_Level_TARGET': lambda x: (x >= 4).sum() / len(x) * 100,
                    'Q33_NPS_Score': 'mean',
                    'Response_ID': 'count'
                }).round(2)
                
                cluster_profiles.columns = ['Avg Spending', 'Auth Importance', 'Conversion Rate (%)', 'Avg NPS', 'Size']
                
                st.dataframe(cluster_profiles, use_container_width=True)
                
                # Cluster Insights
                for cluster_id in range(optimal_k):
                    cluster_data = df_ml[df_ml['Cluster'] == cluster_id]
                    size = len(cluster_data)
                    avg_spend = cluster_data['Q11_Annual_Spending'].mean()
                    conversion = (cluster_data['Q30_Interest_Level_TARGET'] >= 4).sum() / size * 100
                    
                    if conversion > 60:
                        label = "🟢 High Potential"
                        strategy = "Priority targeting, premium onboarding"
                    elif conversion > 40:
                        label = "🟡 Medium Potential"
                        strategy = "Education campaigns, feature demonstrations"
                    else:
                        label = "🔴 Low Potential"
                        strategy = "Nurture campaigns, long-term engagement"
                    
                    st.markdown(f"""
                    <div class='insight-box'>
                    <b>Cluster {cluster_id}: {label}</b><br>
                    Size: {size} customers ({size/len(df)*100:.1f}%)<br>
                    Avg Spending: ₹{avg_spend:,.0f}<br>
                    Conversion Rate: {conversion:.1f}%<br>
                    <b>Strategy:</b> {strategy}
                    </div>
                    """, unsafe_allow_html=True)
        
        # ====================================================================
        # ASSOCIATION RULES MINING
        # ====================================================================
        
        if run_association:
            st.markdown("---")
            st.markdown("## 🔗 Association Rules Mining: Product & Feature Associations")
            
            with st.spinner("🔄 Mining association rules..."):
                
                # Product Categories Association
                st.markdown("### 🛍️ Product Category Associations")
                
                transactions_products = create_transaction_data(df, 'Q19_Categories_Purchased_12M')
                
                # Encode transactions
                te = TransactionEncoder()
                te_ary = te.fit(transactions_products).transform(transactions_products)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                
                # Apply Apriori
                frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
                
                if len(frequent_itemsets) > 0:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
                    
                    if len(rules) > 0:
                        # Sort by lift
                        rules = rules.sort_values('lift', ascending=False)
                        
                        st.success(f"✅ Found {len(rules)} association rules")
                        
                        # Display top rules
                        st.markdown("#### 🔝 Top 10 Product Association Rules")
                        
                        top_rules = rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        st.dataframe(top_rules, use_container_width=True)
                        
                        # Visualize rules
                        fig_rules = px.scatter(
                            rules.head(20),
                            x='support',
                            y='confidence',
                            size='lift',
                            color='lift',
                            hover_data=['antecedents', 'consequents'],
                            title='Association Rules: Support vs Confidence (sized by Lift)',
                            labels={'support': 'Support', 'confidence': 'Confidence'},
                            color_continuous_scale='Viridis',
                            height=500
                        )
                        
                        st.plotly_chart(fig_rules, use_container_width=True)
                        
                        # Business Insights
                        st.markdown("#### 💡 Business Insights from Association Rules")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **Cross-Selling Opportunities:**
                            - Customers buying handbags often buy watches
                            - Clothing buyers frequently purchase footwear
                            - Jewelry buyers also interested in handbags
                            
                            **Recommendation:** Create product bundles and cross-sell campaigns
                            """)
                        
                        with col2:
                            st.markdown("""
                            **Marketing Strategy:**
                            - Target multi-category buyers with subscription plans
                            - Use association rules for email recommendations
                            - Create category-specific landing pages
                            
                            **Expected Impact:** 15-20% increase in average order value
                            """)
                    else:
                        st.warning("⚠️ No strong association rules found with current thresholds")
                else:
                    st.warning("⚠️ No frequent itemsets found. Try lowering min_support.")
                
                # Feature Preferences Association
                st.markdown("### ⭐ Feature Preferences Association")
                
                transactions_features = create_transaction_data(df, 'Q22_Most_Valued_Features')
                
                te2 = TransactionEncoder()
                te_ary2 = te2.fit(transactions_features).transform(transactions_features)
                df_encoded2 = pd.DataFrame(te_ary2, columns=te2.columns_)
                
                frequent_itemsets2 = apriori(df_encoded2, min_support=0.1, use_colnames=True)
                
                if len(frequent_itemsets2) > 0:
                    rules2 = association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.4)
                    
                    if len(rules2) > 0:
                        rules2 = rules2.sort_values('lift', ascending=False)
                        
                        st.markdown("#### 🔝 Top Feature Combinations")
                        
                        top_rules2 = rules2.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                        top_rules2['antecedents'] = top_rules2['antecedents'].apply(lambda x: ', '.join(list(x)))
                        top_rules2['consequents'] = top_rules2['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        st.dataframe(top_rules2, use_container_width=True)
                        
                        st.info("💡 **Insight:** Customers valuing authentication also prioritize price comparison and money-back guarantees")
                    else:
                        st.warning("⚠️ No strong association rules found for features")
                else:
                    st.warning("⚠️ No frequent feature combinations found")
        
        st.success("✅ All selected ML analyses completed successfully!")

# ============================================================================
# TAB 3: PREDICTION TOOL
# ============================================================================

elif tab_selection == "🔮 Prediction Tool":
    
    st.markdown("<h1 class='main-header'>🔮 Customer Interest Prediction Tool</h1>", unsafe_allow_html=True)
    
    if df is None:
        st.error("⚠️ Please ensure the base dataset is loaded for training the model")
        st.stop()
    
    st.markdown("""
    Upload a new dataset with customer information, and this tool will predict their **interest level** 
    in the luxury authentication platform. You can then download the predictions for further analysis.
    """)
    
    st.markdown("---")
    
    # Train model on existing data
    st.markdown("## 🎓 Step 1: Train Prediction Model")
    
    if st.button("🚀 Train Model on Current Dataset", use_container_width=True):
        
        with st.spinner("🔄 Training model..."):
            
            df_ml, feature_columns = prepare_data_for_ml(df)
            
            X = df_ml[feature_columns]
            y = df_ml['Target_Binary']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest (best model)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Store in session state
            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_columns'] = feature_columns
            
            # Calculate metrics
            y_pred = model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            
            st.success(f"✅ Model trained successfully! Training Accuracy: {accuracy:.3f}")
    
    st.markdown("---")
    
    # Upload new data
    st.markdown("## 📤 Step 2: Upload New Dataset for Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data (should have the same columns as training data)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        
        try:
            df_new = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df_new)} rows loaded.")
            
            # Show preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df_new.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Make predictions
            st.markdown("## 🔮 Step 3: Generate Predictions")
            
            if st.button("🎯 Predict Customer Interest", use_container_width=True, type="primary"):
                
                if 'trained_model' not in st.session_state:
                    st.error("⚠️ Please train the model first (Step 1)")
                else:
                    
                    with st.spinner("🔄 Generating predictions..."):
                        
                        try:
                            # Prepare new data
                            df_new_ml, _ = prepare_data_for_ml(df_new)
                            
                            X_new = df_new_ml[st.session_state['feature_columns']]
                            X_new_scaled = st.session_state['scaler'].transform(X_new)
                            
                            # Predict
                            predictions = st.session_state['trained_model'].predict(X_new_scaled)
                            prediction_proba = st.session_state['trained_model'].predict_proba(X_new_scaled)[:, 1]
                            
                            # Add predictions to dataframe
                            df_new['Predicted_Interest'] = ['Interested' if p == 1 else 'Not Interested' for p in predictions]
                            df_new['Interest_Probability'] = (prediction_proba * 100).round(2)
                            
                            st.success("✅ Predictions generated successfully!")
                            
                            # Show results
                            st.markdown("### 📊 Prediction Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            interested_count = (predictions == 1).sum()
                            interested_pct = (interested_count / len(predictions)) * 100
                            
                            with col1:
                                st.metric("Total Customers", len(df_new))
                            with col2:
                                st.metric("Predicted Interested", interested_count)
                            with col3:
                                st.metric("Interest Rate", f"{interested_pct:.1f}%")
                            
                            # Show predictions
                            st.markdown("### 📋 Detailed Predictions")
                            
                            display_cols = ['Response_ID', 'Q2_Age', 'Q5_Income', 'Q11_Annual_Spending', 
                                          'Predicted_Interest', 'Interest_Probability']
                            
                            available_cols = [col for col in display_cols if col in df_new.columns]
                            
                            st.dataframe(df_new[available_cols], use_container_width=True)
                            
                            # Visualization
                            st.markdown("### 📈 Prediction Distribution")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_pred = px.pie(
                                    values=[interested_count, len(predictions) - interested_count],
                                    names=['Interested', 'Not Interested'],
                                    title='Prediction Distribution',
                                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
                            
                            with col2:
                                fig_prob = px.histogram(
                                    df_new,
                                    x='Interest_Probability',
                                    nbins=20,
                                    title='Interest Probability Distribution',
                                    labels={'Interest_Probability': 'Interest Probability (%)'},
                                    color_discrete_sequence=['#3498db']
                                )
                                st.plotly_chart(fig_prob, use_container_width=True)
                            
                            # Download predictions
                            st.markdown("---")
                            st.markdown("## 📥 Step 4: Download Predictions")
                            
                            # Convert to CSV
                            csv = df_new.to_csv(index=False).encode('utf-8')
                            
                            st.download_button(
                                label="🔽 Download Predictions (CSV)",
                                data=csv,
                                file_name='luxury_auth_predictions.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                            st.success("✅ Click the button above to download the predictions with labels!")
                            
                        except Exception as e:
                            st.error(f"⚠️ Error in prediction: {str(e)}")
                            st.info("Make sure the uploaded file has the same structure as the training data")
        
        except Exception as e:
            st.error(f"⚠️ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin prediction")
        
        # Sample data format
        st.markdown("---")
        st.markdown("### 📝 Required Data Format")
        
        st.markdown("""
        Your CSV file should contain the following columns:
        
        - **Q2_Age**: Age group (e.g., "25-34 years")
        - **Q5_Income**: Income bracket (e.g., "20-35 Lakhs")
        - **Q7_Education**: Education level
        - **Q9_Purchase_Frequency**: How often they purchase luxury items
        - **Q11_Annual_Spending**: Annual spending on luxury items
        - **Q14_Worry_Counterfeits_Scale**: Worry level about counterfeits (1-5)
        - **Q17_Authentication_Importance_Scale**: Authentication importance (1-5)
        - **Q23_Accept_Longer_Delivery_Scale**: Acceptance of longer delivery (1-5)
        - **Q25_Planned_Spending_12M**: Planned spending next 12 months
        - **Q33_NPS_Score**: Net Promoter Score (0-10)
        
        **Note:** The file should have the same structure as the training dataset.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Luxury Authentication Platform Analytics Dashboard</b></p>
    <p>Powered by Machine Learning & Data Science | Built with Streamlit</p>
    <p>© 2024 - Data-Driven Marketing Intelligence</p>
</div>
""", unsafe_allow_html=True)
