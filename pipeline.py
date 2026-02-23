# =============================================================================
# RETAIL INSIGHTS — COMPLETE ML PIPELINE
# Instacart Online Grocery Basket Analysis
# =============================================================================
# Run each section in a Jupyter Notebook cell by cell, or as a script.
# pip install -r requirements.txt first!
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Create output dirs
os.makedirs('data/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("RETAIL INSIGHTS PIPELINE STARTING")
print("=" * 60)

# =============================================================================
# SECTION 1: DATA LOADING & MERGING
# =============================================================================
print("\n[1/7] Loading and merging data...")

# --- Load raw files ---
orders           = pd.read_csv('data/raw/orders.csv')
order_products_p = pd.read_csv('data/raw/order_products__prior.csv')
order_products_t = pd.read_csv('data/raw/order_products__train.csv')
products         = pd.read_csv('data/raw/products.csv')
aisles           = pd.read_csv('data/raw/aisles.csv')
departments      = pd.read_csv('data/raw/departments.csv')

# --- Combine prior + train order-product data ---
order_products = pd.concat([order_products_p, order_products_t], ignore_index=True)

# --- Enrich products with aisle & department names ---
products = (products
            .merge(aisles, on='aisle_id')
            .merge(departments, on='department_id'))

# --- Create master DataFrame ---
df = (order_products
      .merge(orders, on='order_id')
      .merge(products, on='product_id'))

print(f"  ✓ Total records: {len(df):,}")
print(f"  ✓ Unique customers: {df['user_id'].nunique():,}")
print(f"  ✓ Unique products: {df['product_id'].nunique():,}")
print(f"  ✓ Unique orders: {df['order_id'].nunique():,}")

df.to_csv('data/processed/master.csv', index=False)
print("  ✓ Saved: data/processed/master.csv")

# =============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[2/7] Exploratory Data Analysis...")

# --- Top 20 products ---
top_products = (df.groupby('product_name')['order_id']
                  .count()
                  .sort_values(ascending=False)
                  .head(20)
                  .reset_index()
                  .rename(columns={'order_id': 'order_count'}))
top_products.to_csv('results/top_products.csv', index=False)
print(f"  ✓ Top product: {top_products.iloc[0]['product_name']} ({top_products.iloc[0]['order_count']:,} orders)")

# --- Orders by hour ---
hour_dist = orders['order_hour_of_day'].value_counts().sort_index()

# --- Orders by day of week ---
dow_map = {0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'}
orders['day_name'] = orders['order_dow'].map(dow_map)
dow_dist = orders['day_name'].value_counts()

# --- Reorder rate by department ---
reorder_dept = (df.groupby('department')['reordered']
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index()
                  .rename(columns={'reordered': 'reorder_rate'}))
reorder_dept['reorder_rate'] = (reorder_dept['reorder_rate'] * 100).round(1)
reorder_dept.to_csv('results/reorder_by_department.csv', index=False)

# --- Plot: top products ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_products['product_name'][:10][::-1], top_products['order_count'][:10][::-1], color='steelblue')
ax.set_xlabel('Number of Orders')
ax.set_title('Top 10 Most Ordered Products')
plt.tight_layout()
plt.savefig('plots/top_products.png', dpi=150)
plt.close()

# --- Plot: orders by hour ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(hour_dist.index, hour_dist.values, color='coral')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Orders')
ax.set_title('Order Volume by Hour of Day')
plt.tight_layout()
plt.savefig('plots/orders_by_hour.png', dpi=150)
plt.close()

print("  ✓ EDA complete. Plots saved to plots/")

# =============================================================================
# SECTION 3: ASSOCIATION RULE MINING (Apriori, FP-Growth, Eclat)
# =============================================================================
print("\n[3/7] Association Rule Mining...")

# --- Build basket: use top N customers for speed ---
# Using top 5000 users keeps it manageable; increase for better rules
TOP_USERS = 5000
MIN_SUPPORT = 0.01   # 1% of baskets
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.2

top_users = df['user_id'].value_counts().head(TOP_USERS).index
basket_df = df[df['user_id'].isin(top_users)]

# One order = one "transaction" → list of product names
transactions = (basket_df.groupby('order_id')['product_name']
                .apply(list)
                .reset_index()['product_name']
                .tolist())

# --- Encode transactions ---
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
basket_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(f"  Transaction matrix: {basket_encoded.shape[0]:,} orders × {basket_encoded.shape[1]:,} products")

# ---- 3A: APRIORI ----
print("  Running Apriori...")
import time
t0 = time.time()
frequent_items_apriori = apriori(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)
rules_apriori = association_rules(frequent_items_apriori, metric='lift', min_threshold=MIN_LIFT)
rules_apriori = rules_apriori[rules_apriori['confidence'] >= MIN_CONFIDENCE]
t_apriori = time.time() - t0
print(f"    Apriori: {len(rules_apriori)} rules in {t_apriori:.1f}s")

# ---- 3B: FP-GROWTH ----
print("  Running FP-Growth...")
t0 = time.time()
frequent_items_fp = fpgrowth(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)
rules_fp = association_rules(frequent_items_fp, metric='lift', min_threshold=MIN_LIFT)
rules_fp = rules_fp[rules_fp['confidence'] >= MIN_CONFIDENCE]
t_fp = time.time() - t0
print(f"    FP-Growth: {len(rules_fp)} rules in {t_fp:.1f}s")

# ---- 3C: ECLAT (manual implementation via frequent itemsets) ----
print("  Running Eclat (vertical data format)...")

def eclat(transactions, min_support_count):
    """Simple Eclat using vertical (tidlist) representation."""
    from collections import defaultdict
    # Build tidlists
    tidlists = defaultdict(set)
    for tid, items in enumerate(transactions):
        for item in items:
            tidlists[frozenset([item])].add(tid)
    
    n = len(transactions)
    frequent = {}
    
    # Single items
    candidates = {k: v for k, v in tidlists.items() if len(v) >= min_support_count}
    frequent.update(candidates)
    
    # Pairs (k=2)
    items = list(candidates.keys())
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            new_key = items[i] | items[j]
            tidlist = candidates[items[i]] & candidates[items[j]]
            if len(tidlist) >= min_support_count:
                frequent[new_key] = tidlist
    
    # Convert to support df
    results = []
    for itemset, tidlist in frequent.items():
        results.append({'itemsets': itemset, 'support': len(tidlist)/n})
    return pd.DataFrame(results)

min_sup_count = int(MIN_SUPPORT * len(transactions))
t0 = time.time()
eclat_frequent = eclat(transactions, min_sup_count)
t_eclat = time.time() - t0
print(f"    Eclat: {len(eclat_frequent)} frequent itemsets in {t_eclat:.1f}s")

# --- Algorithm comparison ---
algo_comparison = pd.DataFrame({
    'Algorithm': ['Apriori', 'FP-Growth', 'Eclat'],
    'Rules/Itemsets': [len(rules_apriori), len(rules_fp), len(eclat_frequent)],
    'Time (s)': [round(t_apriori,2), round(t_fp,2), round(t_eclat,2)]
})
algo_comparison.to_csv('results/algorithm_comparison.csv', index=False)
print("\n  Algorithm Comparison:")
print(algo_comparison.to_string(index=False))

# --- Save best rules (FP-Growth) ---
rules_fp['antecedents_str'] = rules_fp['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules_fp['consequents_str'] = rules_fp['consequents'].apply(lambda x: ', '.join(sorted(x)))
top_rules = rules_fp.sort_values('lift', ascending=False).head(200)
top_rules[['antecedents_str','consequents_str','support','confidence','lift']].to_csv(
    'results/association_rules.csv', index=False)
print(f"\n  ✓ Saved top {len(top_rules)} rules to results/association_rules.csv")
print(f"  Top rule: {top_rules.iloc[0]['antecedents_str']} → {top_rules.iloc[0]['consequents_str']} (lift={top_rules.iloc[0]['lift']:.2f})")

# =============================================================================
# SECTION 4: UP-TREE UTILITY MINING (Value-Aware)
# =============================================================================
print("\n[4/7] UP-Tree Utility Mining...")

# --- Assign estimated prices per department (fallback if no real prices) ---
# (Data engineers can replace this with real prices from Open Food Facts)
dept_avg_price = {
    'produce': 1.5, 'dairy eggs': 3.0, 'meat seafood': 7.0,
    'beverages': 2.5, 'snacks': 2.8, 'frozen': 3.5,
    'pantry': 2.0, 'bakery': 2.5, 'deli': 4.5,
    'canned goods': 1.8, 'dry goods pasta': 1.5, 'breakfast': 3.0,
    'personal care': 5.0, 'household': 4.0, 'babies': 6.0,
    'international': 2.5, 'other': 2.0, 'missing': 2.0,
    'alcohol': 8.0, 'bulk': 1.0, 'pets': 5.0,
}

products['est_price'] = products['department'].str.lower().map(dept_avg_price).fillna(2.0)
df = df.merge(products[['product_id','est_price']], on='product_id', how='left')

# --- Calculate utility per order ---
order_utility = (df.groupby('order_id')['est_price']
                   .sum()
                   .reset_index()
                   .rename(columns={'est_price': 'order_value'}))

# --- High-utility rules: filter association rules by basket value ---
# Join order_utility to basket orders, find high-value co-purchases
high_value_orders = order_utility[order_utility['order_value'] > order_utility['order_value'].quantile(0.75)]
high_value_basket = basket_df[basket_df['order_id'].isin(high_value_orders['order_id'])]

hv_transactions = (high_value_basket.groupby('order_id')['product_name']
                   .apply(list)
                   .tolist())

if len(hv_transactions) > 100:
    te2 = TransactionEncoder()
    te2_array = te2.fit_transform(hv_transactions)
    hv_encoded = pd.DataFrame(te2_array, columns=te2.columns_)
    
    freq_hv = fpgrowth(hv_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    if len(freq_hv) > 0:
        rules_hv = association_rules(freq_hv, metric='lift', min_threshold=MIN_LIFT)
        rules_hv['antecedents_str'] = rules_hv['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules_hv['consequents_str'] = rules_hv['consequents'].apply(lambda x: ', '.join(sorted(x)))
        rules_hv[['antecedents_str','consequents_str','support','confidence','lift']].to_csv(
            'results/utility_rules.csv', index=False)
        print(f"  ✓ {len(rules_hv)} high-utility rules saved to results/utility_rules.csv")
    else:
        print("  ⚠ Not enough high-value transactions for utility rules — try lowering min_support")
else:
    print("  ⚠ Not enough high-value transactions — skipping utility rules")

# =============================================================================
# SECTION 5: CUSTOMER SEGMENTATION (RFM + KMeans)
# =============================================================================
print("\n[5/7] Customer Segmentation (RFM + KMeans)...")

# --- RFM Calculation ---
# Recency: days since last order (lower = more recent)
# Frequency: number of orders
# Monetary: estimated total spend

user_orders = orders[orders['eval_set'].isin(['prior','train'])].copy()

# Frequency
freq = user_orders.groupby('user_id')['order_id'].count().reset_index()
freq.columns = ['user_id', 'frequency']

# Recency (use days_since_prior_order as proxy)
recency = user_orders.groupby('user_id')['days_since_prior_order'].mean().reset_index()
recency.columns = ['user_id', 'avg_days_between_orders']

# Monetary (estimated)
user_spend = (df.merge(order_utility, on='order_id')
               .groupby('user_id')['order_value']
               .sum()
               .reset_index()
               .rename(columns={'order_value': 'total_spend'}))

# Basket size
basket_size = (df.groupby(['user_id','order_id'])['product_id']
                .count()
                .reset_index()
                .rename(columns={'product_id': 'basket_size'}))
avg_basket = basket_size.groupby('user_id')['basket_size'].agg(['mean','std']).reset_index()
avg_basket.columns = ['user_id', 'avg_basket_size', 'basket_size_std']
avg_basket['basket_size_std'] = avg_basket['basket_size_std'].fillna(0)

# Merge RFM
rfm = freq.merge(recency, on='user_id').merge(user_spend, on='user_id').merge(avg_basket, on='user_id')
rfm = rfm.dropna()

print(f"  RFM built for {len(rfm):,} customers")

# --- KMeans Clustering ---
features = ['frequency', 'avg_days_between_orders', 'total_spend', 'avg_basket_size']
X = rfm[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using inertia
inertia = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Use k=4 (good balance of interpretability)
K = 4
km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm['cluster'] = km_final.fit_predict(X_scaled)

# --- Label clusters based on spend & frequency ---
cluster_summary = rfm.groupby('cluster')[features].mean()
cluster_summary['total_spend_rank'] = cluster_summary['total_spend'].rank()
cluster_summary['freq_rank'] = cluster_summary['frequency'].rank()

def label_cluster(row):
    spend = row['total_spend_rank']
    freq = row['freq_rank']
    if spend >= 3 and freq >= 3:
        return 'Premium & Loyal'
    elif spend >= 3:
        return 'High Spender (Irregular)'
    elif freq >= 3:
        return 'Frequent Budget Shopper'
    else:
        return 'Occasional Shopper'

cluster_summary['label'] = cluster_summary.apply(label_cluster, axis=1)
rfm['segment'] = rfm['cluster'].map(cluster_summary['label'])

rfm.to_csv('results/customer_segments.csv', index=False)
print(f"  ✓ Segments saved to results/customer_segments.csv")
print("\n  Segment distribution:")
print(rfm['segment'].value_counts().to_string())

# --- Plot: elbow ---
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(k_range, inertia, 'bx-')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal k')
plt.tight_layout()
plt.savefig('plots/elbow.png', dpi=150)
plt.close()

# --- Plot: segment distribution ---
seg_counts = rfm['segment'].value_counts()
fig, ax = plt.subplots(figsize=(7,4))
seg_counts.plot(kind='bar', ax=ax, color=['#2ecc71','#3498db','#e74c3c','#f39c12'])
ax.set_title('Customer Segments')
ax.set_ylabel('Number of Customers')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plots/segments.png', dpi=150)
plt.close()

# =============================================================================
# SECTION 6: REVENUE SIMULATION
# =============================================================================
print("\n[6/7] Revenue Simulation...")

# --- Bundle Revenue Simulation ---
# For each top rule: estimate revenue lift if we promote the bundle
if os.path.exists('results/association_rules.csv'):
    rules = pd.read_csv('results/association_rules.csv')
    
    # Estimate: if we target N customers with a bundle at 10% discount,
    # and confidence% will buy the consequent → net revenue change
    N_CUSTOMERS = rfm['user_id'].nunique()
    AVG_ORDER_VALUE = order_utility['order_value'].mean()
    DISCOUNT = 0.10  # 10% discount on bundle

    rules['est_new_orders'] = (rules['support'] * N_CUSTOMERS * rules['confidence']).astype(int)
    rules['revenue_from_lift'] = rules['est_new_orders'] * AVG_ORDER_VALUE * (rules['lift'] - 1)
    rules['discount_cost'] = rules['est_new_orders'] * AVG_ORDER_VALUE * DISCOUNT
    rules['net_revenue_gain'] = rules['revenue_from_lift'] - rules['discount_cost']
    rules['roi_pct'] = ((rules['net_revenue_gain'] / (rules['discount_cost'] + 0.01)) * 100).round(1)
    
    rules.sort_values('net_revenue_gain', ascending=False).to_csv(
        'results/revenue_simulation.csv', index=False)
    
    top_bundle = rules.iloc[0]
    print(f"  Best bundle: '{top_bundle['antecedents_str']}' → '{top_bundle['consequents_str']}'")
    print(f"  Estimated net revenue gain: ${top_bundle['net_revenue_gain']:,.0f}")
    print(f"  ROI: {top_bundle['roi_pct']}%")
    print("  ✓ Saved: results/revenue_simulation.csv")

# =============================================================================
# SECTION 7: PROMOTION EFFICIENCY ANALYSIS
# =============================================================================
print("\n[7/7] Promotion Efficiency Analysis...")

# Compare: targeted discount (based on rules) vs random/untargeted
segment_spend = rfm.groupby('segment').agg(
    customers=('user_id','count'),
    avg_spend=('total_spend','mean'),
    avg_frequency=('frequency','mean'),
    avg_basket=('avg_basket_size','mean')
).reset_index()

# Simulate: 15% blanket discount vs 10% targeted discount
BLANKET_DISCOUNT = 0.15
TARGETED_DISCOUNT = 0.10
TARGETED_LIFT = 1.25  # assumption: targeted promos drive 25% more spend

segment_spend['blanket_discount_cost'] = segment_spend['customers'] * segment_spend['avg_spend'] * BLANKET_DISCOUNT
segment_spend['targeted_discount_cost'] = segment_spend['customers'] * segment_spend['avg_spend'] * TARGETED_DISCOUNT
segment_spend['targeted_revenue_lift'] = segment_spend['customers'] * segment_spend['avg_spend'] * (TARGETED_LIFT - 1)
segment_spend['targeted_net_gain'] = segment_spend['targeted_revenue_lift'] - segment_spend['targeted_discount_cost']
segment_spend['blanket_net_gain'] = -segment_spend['blanket_discount_cost']  # no lift assumed for blanket
segment_spend['targeting_advantage'] = segment_spend['targeted_net_gain'] - segment_spend['blanket_net_gain']

segment_spend.to_csv('results/promotion_efficiency.csv', index=False)
total_advantage = segment_spend['targeting_advantage'].sum()
print(f"  Targeted vs Blanket discount advantage: ${total_advantage:,.0f}")
print("  ✓ Saved: results/promotion_efficiency.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("\nFiles generated:")
for f in ['data/processed/master.csv',
          'results/top_products.csv',
          'results/association_rules.csv',
          'results/utility_rules.csv',
          'results/customer_segments.csv',
          'results/revenue_simulation.csv',
          'results/promotion_efficiency.csv',
          'results/algorithm_comparison.csv']:
    status = "✓" if os.path.exists(f) else "✗"
    print(f"  {status} {f}")

print("\nNext step: run `streamlit run app.py` to launch the dashboard!")
