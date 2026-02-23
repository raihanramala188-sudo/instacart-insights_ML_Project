# =============================================================================
# RETAIL INSIGHTS DASHBOARD — Streamlit App
# Run with: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Retail Insights Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2em; font-weight: bold; }
    .metric-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    h1 { color: #2d3748; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {}
    files = {
        'rules':      'results/association_rules.csv',
        'segments':   'results/customer_segments.csv',
        'revenue':    'results/revenue_simulation.csv',
        'promotion':  'results/promotion_efficiency.csv',
        'top_prod':   'results/top_products.csv',
        'algo_comp':  'results/algorithm_comparison.csv',
        'reorder':    'results/reorder_by_department.csv',
    }
    for key, path in files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = None
    return data

data = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=80)
    st.title("🛒 Retail Insights")
    st.markdown("**DSTI Data Analytics Project**")
    st.divider()
    
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "👥 Customer Segments",
         "🔗 Product Associations",
         "💰 Revenue Simulation",
         "📣 Promotion ROI"]
    )
    
    st.divider()
    st.caption("Instacart Online Grocery Dataset\n3M+ orders | 200K+ customers")

# ─── Helper: check data ───────────────────────────────────────────────────────
def data_missing(key):
    if data[key] is None:
        st.warning(f"⚠️ Run `pipeline.py` first to generate results. Missing: `results/{key}.csv`")
        return True
    return False

# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page == "🏠 Overview":
    st.title("🛒 Retail Insights Dashboard")
    st.markdown("*Data-Driven Cost Savings & Revenue Growth — DSTI Project*")
    st.divider()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    n_customers = len(data['segments']) if data['segments'] is not None else "—"
    n_rules     = len(data['rules'])    if data['rules']    is not None else "—"
    
    if data['revenue'] is not None:
        total_gain = data['revenue']['net_revenue_gain'].sum()
        gain_str = f"${total_gain:,.0f}"
    else:
        gain_str = "—"
    
    if data['promotion'] is not None:
        targeting_adv = data['promotion']['targeting_advantage'].sum()
        adv_str = f"${targeting_adv:,.0f}"
    else:
        adv_str = "—"

    with col1:
        st.metric("👥 Customers Analyzed", f"{n_customers:,}" if isinstance(n_customers, int) else n_customers)
    with col2:
        st.metric("🔗 Association Rules", f"{n_rules:,}" if isinstance(n_rules, int) else n_rules)
    with col3:
        st.metric("💰 Est. Revenue Gain", gain_str)
    with col4:
        st.metric("📈 Targeting Advantage", adv_str)

    st.divider()

    col_l, col_r = st.columns(2)
    
    # Top Products
    with col_l:
        st.subheader("🏆 Top 10 Most Ordered Products")
        if not data_missing('top_prod'):
            fig = px.bar(
                data['top_prod'].head(10).sort_values('order_count'),
                x='order_count', y='product_name',
                orientation='h',
                color='order_count',
                color_continuous_scale='Blues',
                labels={'order_count': 'Orders', 'product_name': ''}
            )
            fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Algorithm Comparison
    with col_r:
        st.subheader("⚡ Algorithm Performance Comparison")
        if not data_missing('algo_comp'):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=data['algo_comp']['Algorithm'],
                y=data['algo_comp']['Rules/Itemsets'],
                name='Rules Found', marker_color='#667eea'
            ))
            fig.add_trace(go.Scatter(
                x=data['algo_comp']['Algorithm'],
                y=data['algo_comp']['Time (s)'],
                name='Time (s)', mode='lines+markers',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=10)
            ), secondary_y=True)
            fig.update_layout(height=400, title_text="Rules Found vs. Execution Time")
            fig.update_yaxes(title_text="Rules/Itemsets", secondary_y=False)
            fig.update_yaxes(title_text="Time (seconds)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # Insight box
    st.markdown("""
    <div class="insight-box">
    <b>💡 Key Finding:</b> FP-Growth is significantly faster than Apriori on large datasets 
    while discovering the same rules. UP-Tree further improves relevance by identifying 
    <em>high-value</em> bundles, not just frequent ones.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTS
# =============================================================================
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.markdown("RFM Analysis + KMeans Clustering")
    st.divider()

    if data_missing('segments'):
        st.stop()

    seg = data['segments']

    # Segment overview
    seg_summary = seg.groupby('segment').agg(
        Customers=('user_id','count'),
        Avg_Frequency=('frequency','mean'),
        Avg_Days_Between=('avg_days_between_orders','mean'),
        Avg_Spend=('total_spend','mean'),
        Avg_Basket_Size=('avg_basket_size','mean'),
    ).reset_index().round(1)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Segment Breakdown")
        fig = px.pie(
            seg_summary, names='segment', values='Customers',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment Profiles")
        # Format for display
        display_df = seg_summary.rename(columns={
            'segment': 'Segment',
            'Customers': '# Customers',
            'Avg_Frequency': 'Avg Orders',
            'Avg_Days_Between': 'Days Between Orders',
            'Avg_Spend': 'Avg Total Spend ($)',
            'Avg_Basket_Size': 'Avg Basket Size'
        })
        st.dataframe(display_df.set_index('Segment'), use_container_width=True)

    st.divider()

    # Scatter: Frequency vs Spend
    st.subheader("Customer Map: Frequency vs. Total Spend")
    sample = seg.sample(min(3000, len(seg)), random_state=42)
    fig = px.scatter(
        sample,
        x='frequency', y='total_spend',
        color='segment',
        size='avg_basket_size',
        hover_data=['avg_basket_size', 'avg_days_between_orders'],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'frequency': 'Number of Orders', 'total_spend': 'Total Estimated Spend ($)'},
        opacity=0.6
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Basket variability
    st.subheader("Basket Size Variability by Segment")
    fig = px.box(
        seg, x='segment', y='avg_basket_size',
        color='segment',
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'avg_basket_size': 'Avg Basket Size', 'segment': ''},
        points="outliers"
    )
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>💡 Business Insight:</b> Premium & Loyal customers represent the highest 
    lifetime value. Targeting them with exclusive bundles and early access offers 
    can increase retention and basket size. Occasional Shoppers can be re-engaged 
    with first-purchase-back discount campaigns.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 3: PRODUCT ASSOCIATIONS
# =============================================================================
elif page == "🔗 Product Associations":
    st.title("🔗 Product Associations & Bundles")
    st.markdown("Frequent itemset mining — Apriori, FP-Growth, Eclat, UP-Tree")
    st.divider()

    if data_missing('rules'):
        st.stop()

    rules = data['rules'].copy()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Min Support", 0.01, 0.3, 0.02, 0.01)
    with col2:
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    with col3:
        min_lift = st.slider("Min Lift", 1.0, 10.0, 1.2, 0.1)

    filtered = rules[
        (rules['support'] >= min_support) &
        (rules['confidence'] >= min_confidence) &
        (rules['lift'] >= min_lift)
    ]

    st.markdown(f"**{len(filtered)} rules** match your filters")

    # Top rules table
    st.subheader("🏆 Top Association Rules (by Lift)")
    display_rules = filtered.sort_values('lift', ascending=False).head(20)[[
        'antecedents_str', 'consequents_str', 'support', 'confidence', 'lift'
    ]].rename(columns={
        'antecedents_str': 'If Customer Buys...',
        'consequents_str': '...They Also Buy',
        'support': 'Support',
        'confidence': 'Confidence',
        'lift': 'Lift'
    })
    display_rules[['Support','Confidence','Lift']] = display_rules[['Support','Confidence','Lift']].round(3)
    st.dataframe(display_rules, use_container_width=True, height=400)

    st.divider()

    # Scatter: Support vs Confidence coloured by Lift
    st.subheader("Rule Map: Support vs. Confidence")
    fig = px.scatter(
        filtered,
        x='support', y='confidence', color='lift',
        size='lift', hover_data=['antecedents_str', 'consequents_str'],
        color_continuous_scale='Viridis',
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
        opacity=0.75
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Utility rules
    if data['rules'] is not None:
        st.divider()
        st.subheader("💎 High-Utility Rules (UP-Tree — Value-Aware)")
        
        util_path = 'results/utility_rules.csv'
        if os.path.exists(util_path):
            util_rules = pd.read_csv(util_path)
            util_rules = util_rules.sort_values('lift', ascending=False).head(10)
            util_rules[['support','confidence','lift']] = util_rules[['support','confidence','lift']].round(3)
            st.dataframe(util_rules.rename(columns={
                'antecedents_str': 'If Customer Buys...',
                'consequents_str': '...They Also Buy'
            }), use_container_width=True)
            st.markdown("""
            <div class="insight-box">
            <b>💡 UP-Tree Advantage:</b> These rules are filtered to orders with above-average 
            basket value, meaning they represent bundles purchased by <em>high-spending</em> 
            customers — more relevant for revenue maximization than pure frequency alone.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Utility rules not yet generated. Run pipeline.py with sufficient data.")

# =============================================================================
# PAGE 4: REVENUE SIMULATION
# =============================================================================
elif page == "💰 Revenue Simulation":
    st.title("💰 Revenue Simulation")
    st.markdown("*If I adopt these insights, how much money do I save — and why?*")
    st.divider()

    if data_missing('revenue'):
        st.stop()

    rev = data['revenue'].copy()

    # Interactive sliders
    st.subheader("⚙️ Simulation Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_customers = st.number_input("Customer base size", 1000, 500000, 50000, 1000)
    with col2:
        avg_order_val = st.number_input("Avg order value ($)", 5.0, 200.0, 25.0, 1.0)
    with col3:
        discount_pct = st.slider("Bundle discount (%)", 5, 30, 10)

    # Recalculate
    rev_sim = rev.copy()
    rev_sim['est_new_orders'] = (rev_sim['support'] * n_customers * rev_sim['confidence']).astype(int)
    rev_sim['revenue_from_lift'] = rev_sim['est_new_orders'] * avg_order_val * (rev_sim['lift'] - 1)
    rev_sim['discount_cost'] = rev_sim['est_new_orders'] * avg_order_val * (discount_pct / 100)
    rev_sim['net_revenue_gain'] = rev_sim['revenue_from_lift'] - rev_sim['discount_cost']
    rev_sim['roi_pct'] = ((rev_sim['net_revenue_gain'] / (rev_sim['discount_cost'] + 0.01)) * 100).round(1)
    rev_sim = rev_sim.sort_values('net_revenue_gain', ascending=False)

    # KPI row
    total_gain = rev_sim['net_revenue_gain'].sum()
    total_cost = rev_sim['discount_cost'].sum()
    total_lift_rev = rev_sim['revenue_from_lift'].sum()
    avg_roi = rev_sim['roi_pct'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue Lift", f"${total_lift_rev:,.0f}")
    col2.metric("Total Discount Cost", f"${total_cost:,.0f}")
    col3.metric("Net Gain", f"${total_gain:,.0f}")
    col4.metric("Avg ROI", f"{avg_roi:.0f}%")

    st.divider()

    # Top bundles bar chart
    st.subheader("🏆 Top 15 Bundles by Net Revenue Gain")
    top15 = rev_sim.head(15).copy()
    top15['bundle'] = top15['antecedents_str'] + " → " + top15['consequents_str']
    top15['bundle'] = top15['bundle'].str[:60] + "..."
    
    fig = px.bar(
        top15.sort_values('net_revenue_gain'),
        x='net_revenue_gain', y='bundle', orientation='h',
        color='roi_pct', color_continuous_scale='RdYlGn',
        labels={'net_revenue_gain': 'Net Revenue Gain ($)', 'bundle': '', 'roi_pct': 'ROI (%)'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Gain vs Cost scatter
    st.subheader("Revenue Lift vs. Discount Cost")
    fig = px.scatter(
        rev_sim.head(100),
        x='discount_cost', y='revenue_from_lift',
        size='est_new_orders', color='roi_pct',
        hover_data=['antecedents_str', 'consequents_str'],
        color_continuous_scale='RdYlGn',
        labels={'discount_cost': 'Discount Cost ($)', 'revenue_from_lift': 'Revenue Lift ($)'}
    )
    # Add break-even line
    max_val = max(rev_sim['discount_cost'].max(), rev_sim['revenue_from_lift'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', line=dict(dash='dash', color='red'),
        name='Break-even'
    ))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    with st.expander("📋 Full Results Table"):
        st.dataframe(rev_sim[['antecedents_str','consequents_str','est_new_orders',
                               'revenue_from_lift','discount_cost','net_revenue_gain','roi_pct'
                              ]].rename(columns={
            'antecedents_str': 'If Buys', 'consequents_str': 'Also Buys',
            'est_new_orders': 'Est. Orders', 'revenue_from_lift': 'Revenue Lift ($)',
            'discount_cost': 'Discount Cost ($)', 'net_revenue_gain': 'Net Gain ($)', 'roi_pct': 'ROI (%)'
        }).round(2), use_container_width=True)

# =============================================================================
# PAGE 5: PROMOTION ROI
# =============================================================================
elif page == "📣 Promotion ROI":
    st.title("📣 Promotion Efficiency Analysis")
    st.markdown("Targeted data-driven discounts vs. blanket untargeted discounts")
    st.divider()

    if data_missing('promotion'):
        st.stop()

    promo = data['promotion'].copy()

    st.subheader("⚙️ Promotion Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        blanket_disc = st.slider("Blanket discount (%)", 5, 30, 15)
    with col2:
        targeted_disc = st.slider("Targeted discount (%)", 5, 20, 10)
    with col3:
        targeted_lift = st.slider("Expected lift from targeting", 1.05, 2.0, 1.25, 0.05)

    # Recalculate
    promo['blanket_discount_cost'] = promo['customers'] * promo['avg_spend'] * (blanket_disc / 100)
    promo['targeted_discount_cost'] = promo['customers'] * promo['avg_spend'] * (targeted_disc / 100)
    promo['targeted_revenue_lift'] = promo['customers'] * promo['avg_spend'] * (targeted_lift - 1)
    promo['targeted_net_gain'] = promo['targeted_revenue_lift'] - promo['targeted_discount_cost']
    promo['blanket_net_gain'] = -promo['blanket_discount_cost']
    promo['targeting_advantage'] = promo['targeted_net_gain'] - promo['blanket_net_gain']

    total_targeted = promo['targeted_net_gain'].sum()
    total_blanket = promo['blanket_net_gain'].sum()
    advantage = promo['targeting_advantage'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Blanket Discount P&L", f"${total_blanket:,.0f}", delta_color="off")
    col2.metric("Targeted Discount P&L", f"${total_targeted:,.0f}", delta=f"+${total_targeted-total_blanket:,.0f} vs blanket")
    col3.metric("Targeting Advantage", f"${advantage:,.0f}")

    st.divider()

    # Grouped bar by segment
    st.subheader("P&L by Customer Segment: Targeted vs. Blanket")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Blanket Discount P&L',
        x=promo['segment'], y=promo['blanket_net_gain'],
        marker_color='#e74c3c'
    ))
    fig.add_trace(go.Bar(
        name='Targeted Discount P&L',
        x=promo['segment'], y=promo['targeted_net_gain'],
        marker_color='#2ecc71'
    ))
    fig.update_layout(barmode='group', height=420,
                      yaxis_title='Net P&L ($)', xaxis_title='Customer Segment')
    st.plotly_chart(fig, use_container_width=True)

    # Cost vs lift breakdown
    st.subheader("Cost vs Revenue Lift by Segment")
    fig = px.bar(
        promo.melt(id_vars='segment', value_vars=['targeted_discount_cost','targeted_revenue_lift'],
                   var_name='Component', value_name='Amount ($)'),
        x='segment', y='Amount ($)', color='Component',
        barmode='group',
        color_discrete_map={
            'targeted_discount_cost': '#e74c3c',
            'targeted_revenue_lift': '#2ecc71'
        },
        labels={'segment': 'Customer Segment'}
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>💡 Key Takeaway for Shop Owners:</b> Blanket discounts erode margin without guaranteed 
    incremental revenue. Targeted promotions based on association rules and customer segments 
    deliver discounts only to customers most likely to respond — maximizing ROI and reducing 
    unnecessary spend. Even a modest 25% uplift from targeting can convert a loss-making 
    promotion into a profitable growth lever.
    </div>
    """, unsafe_allow_html=True)

    # Full table
    with st.expander("📋 Full Segment Table"):
        display_promo = promo.round(0)[['segment','customers','avg_spend',
                                        'blanket_discount_cost','targeted_discount_cost',
                                        'targeted_revenue_lift','targeted_net_gain','targeting_advantage']]
        display_promo.columns = ['Segment','Customers','Avg Spend ($)',
                                  'Blanket Cost ($)','Targeted Cost ($)',
                                  'Revenue Lift ($)','Targeted Net ($)','Advantage ($)']
        st.dataframe(display_promo, use_container_width=True)
