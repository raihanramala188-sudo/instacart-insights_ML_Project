# =============================================================================
# INSTACART ML PIPELINE — OPTIMIZED
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings, os, time
warnings.filterwarnings('ignore')

# --- Create dirs ---
os.makedirs('data/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# --- CONFIG (tune these) ---
TOP_USERS = 5000
TOP_PRODUCTS = 500
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.2
MIN_ORDERS_PER_USER = 3  # Filter inactive users
MIN_ITEMS_PER_ORDER = 1   # Filter empty orders
SKIP_PLOTS = False         # Set True for faster debugging iterations

print("="*60)
print("RETAIL INSIGHTS PIPELINE STARTING")
print("="*60)

# =============================================================================
# 1. DATA LOADING, CLEANING & MERGING (optimized)
# =============================================================================
print("\n[1/7] Loading and cleaning data...")
t1 = time.time()

# Load only needed columns
orders = pd.read_csv('data/raw/orders.csv', usecols=['order_id','user_id','eval_set','order_number','order_dow','order_hour_of_day','days_since_prior_order'])
order_products_p = pd.read_csv('data/raw/order_products__prior.csv', usecols=['order_id','product_id','reordered'])
order_products_t = pd.read_csv('data/raw/order_products__train.csv', usecols=['order_id','product_id','reordered'])
products = pd.read_csv('data/raw/products.csv', usecols=['product_id','product_name','aisle_id','department_id'])
aisles = pd.read_csv('data/raw/aisles.csv')
departments = pd.read_csv('data/raw/departments.csv')

print(f"  Raw records: orders={len(orders):,}, order_products={len(order_products_p)+len(order_products_t):,}")

# --- DATA CLEANING ---
# Remove nulls
orders = orders.dropna(subset=['user_id','order_id'])
products = products.dropna(subset=['product_id','product_name'])

# Remove duplicates
order_products_p = order_products_p.drop_duplicates(subset=['order_id','product_id'])
order_products_t = order_products_t.drop_duplicates(subset=['order_id','product_id'])

# Combine order products
order_products = pd.concat([order_products_p, order_products_t], ignore_index=True)
del order_products_p, order_products_t  # Free memory
order_products = order_products.drop_duplicates(subset=['order_id','product_id'])

# Merge products with metadata (aisles + departments)
products = products.merge(aisles, on='aisle_id', how='left').merge(departments, on='department_id', how='left')
products['aisle'] = products['aisle'].fillna('Unknown')
products['department'] = products['department'].fillna('Unknown')

# --- EARLY FILTERING (before main merge) ---
# Filter to active users only (3+ orders)
active_users = orders.groupby('user_id').size()
active_users = active_users[active_users >= MIN_ORDERS_PER_USER].index
orders = orders[orders['user_id'].isin(active_users)]
print(f"  Filtered users: {len(active_users):,} active (3+ orders)")

# Filter to eval_set='train' (faster training)
orders = orders[orders['eval_set'] == 'train']

# Keep only order_products with valid orders
order_products = order_products[order_products['order_id'].isin(orders['order_id'])]

# Remove items with null product_id
order_products = order_products.dropna(subset=['product_id'])

# Final merge (much smaller now!)
df = order_products.merge(orders[['order_id','user_id','order_dow','order_hour_of_day','days_since_prior_order']], on='order_id', how='inner')
df = df.merge(products[['product_id','product_name','aisle','department']], on='product_id', how='inner')

# Remove orders with no items
df = df.groupby('order_id').filter(lambda x: len(x) >= MIN_ITEMS_PER_ORDER)

print(f"  After cleaning: {len(df):,} records ({time.time()-t1:.1f}s)")
df.to_csv('data/processed/master.csv', index=False)
print(f"  ✓ Cleaned master dataset saved")

# --- DATA QUALITY REPORT ---
null_counts = df.isnull().sum()
duplicates_check = df.duplicated(subset=['order_id','product_id']).sum()
print(f"  Nulls: {null_counts.sum():,} | Duplicates: {duplicates_check:,}")
if null_counts.sum() > 0:
    print(f"  Warning: {list(null_counts[null_counts>0].to_dict().items())}")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[2/7] Exploratory Data Analysis...")
t2 = time.time()

if not SKIP_PLOTS:
    top_products = df['product_name'].value_counts().head(20).reset_index()
    top_products.columns = ['product_name','order_count']
    top_products.to_csv('results/top_products.csv', index=False)

    # Plot top 10 products
    plt.figure(figsize=(10,6))
    sns.barplot(x='order_count', y='product_name', data=top_products.head(10), color='steelblue')
    plt.xlabel('Number of Orders'); plt.ylabel('Product'); plt.title('Top 10 Products')
    plt.tight_layout(); plt.savefig('plots/top_products.png'); plt.close()

    # Orders by hour
    hour_dist = orders['order_hour_of_day'].value_counts().sort_index()
    plt.figure(figsize=(10,4)); plt.bar(hour_dist.index, hour_dist.values, color='coral')
    plt.xlabel('Hour'); plt.ylabel('Orders'); plt.title('Orders by Hour'); plt.tight_layout()
    plt.savefig('plots/orders_by_hour.png'); plt.close()
    print(f"  ✓ EDA complete ({time.time()-t2:.1f}s)")
else:
    print(f"  ⊘ Skipping plots (SKIP_PLOTS=True) ({time.time()-t2:.1f}s)")

# =============================================================================
# 3. ASSOCIATION RULE MINING (FP-Growth, top products, sparse)
# =============================================================================
print("\n[3/7] Association Rule Mining...")
t3 = time.time()

# Sample top users & products early
top_users = df['user_id'].value_counts().head(TOP_USERS).index
top_products_list = df['product_name'].value_counts().head(TOP_PRODUCTS).index
basket_df = df[(df['user_id'].isin(top_users)) & (df['product_name'].isin(top_products_list))].copy()

print(f"  Basket data: {len(basket_df):,} rows, {basket_df['order_id'].nunique():,} orders")

# FP-Growth (faster than Apriori)
transactions = basket_df.groupby('order_id')['product_name'].apply(list).values
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions, sparse=True)
basket_encoded = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)

freq_items = fpgrowth(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)

if len(freq_items) > 0:
    rules_fp = association_rules(freq_items, metric='lift', min_threshold=MIN_LIFT)
    rules_fp = rules_fp[rules_fp['confidence'] >= MIN_CONFIDENCE]
    # Keep only top rules
    rules_fp = rules_fp.nlargest(200, 'lift')
    
    rules_fp['antecedents_str'] = rules_fp['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules_fp['consequents_str'] = rules_fp['consequents'].apply(lambda x: ', '.join(sorted(x)))
    rules_fp.to_csv('results/association_rules.csv', index=False)
    print(f"  ✓ FP-Growth rules saved ({len(rules_fp)} rules in {time.time()-t3:.1f}s)")
else:
    print(f"  ⚠ No frequent itemsets found (try lower MIN_SUPPORT)")

# =============================================================================
# 4. UTILITY MINING (high-value orders)
# =============================================================================
print("\n[4/7] High-Utility Mining...")
t4 = time.time()

# Generate estimates ONCE at the start (not per product)
np.random.seed(42)  # For reproducibility
dept_prices = {d: np.random.uniform(1.5, 8.0) for d in products['department'].unique()}
products['est_price'] = products['department'].map(dept_prices).fillna(2.0)

# Merge prices once
df = df.merge(products[['product_id','est_price']], on='product_id', how='left')
df['est_price'] = df['est_price'].fillna(2.0)

# Calculate order values
order_utility = df.groupby('order_id')['est_price'].sum().reset_index().rename(columns={'est_price':'order_value'})
high_value_orders = order_utility[order_utility['order_value'] > order_utility['order_value'].quantile(0.75)]

hv_basket = basket_df[basket_df['order_id'].isin(high_value_orders['order_id'])]
hv_transactions = hv_basket.groupby('order_id')['product_name'].apply(list).tolist()

if len(hv_transactions) > 100:
    te2 = TransactionEncoder()
    hv_encoded = pd.DataFrame.sparse.from_spmatrix(te2.fit(hv_transactions).transform(hv_transactions, sparse=True),
                                                   columns=te2.columns_)
    freq_hv = fpgrowth(hv_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    if len(freq_hv) > 0:
        rules_hv = association_rules(freq_hv, metric='lift', min_threshold=MIN_LIFT)
        rules_hv[['antecedents','consequents','support','confidence','lift']].to_csv('results/utility_rules.csv', index=False)
        print(f"  ✓ High-utility rules saved ({len(rules_hv)} rules in {time.time()-t4:.1f}s)")
    else:
        print("  ⚠ Not enough high-value rules")
else:
    print("  ⚠ Not enough high-value transactions")

# =============================================================================
# 5. CUSTOMER SEGMENTATION (RFM + KMeans)
# =============================================================================
print("\n[5/7] Customer Segmentation...")
freq = orders.groupby('user_id')['order_id'].count().reset_index().rename(columns={'order_id':'frequency'})
recency = orders.groupby('user_id')['days_since_prior_order'].mean().reset_index().rename(columns={'days_since_prior_order':'avg_days_between_orders'})
monetary = df.groupby('user_id')['est_price'].sum().reset_index().rename(columns={'est_price':'total_spend'})
basket = df.groupby(['user_id','order_id']).size().groupby('user_id').agg(['mean','std']).reset_index()
basket.columns = ['user_id','avg_basket_size','basket_size_std']; basket['basket_size_std'] = basket['basket_size_std'].fillna(0)
rfm = freq.merge(recency,on='user_id').merge(monetary,on='user_id').merge(basket,on='user_id')

features = ['frequency','avg_days_between_orders','total_spend','avg_basket_size']
X_scaled = StandardScaler().fit_transform(rfm[features])
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
rfm['segment'] = km.labels_
rfm.to_csv('results/customer_segments.csv', index=False)
print("  ✓ Customer segmentation complete")

# =============================================================================
# 6. REVENUE SIMULATION
# =============================================================================
print("\n[6/7] Revenue Simulation...")
if os.path.exists('results/association_rules.csv'):
    rules = pd.read_csv('results/association_rules.csv')
    N_CUSTOMERS = rfm['user_id'].nunique()
    AVG_ORDER = df.groupby('order_id')['est_price'].sum().mean()
    DISCOUNT = 0.1
    rules['est_new_orders'] = (rules['support']*N_CUSTOMERS*rules['confidence']).astype(int)
    rules['revenue_from_lift'] = rules['est_new_orders']*AVG_ORDER*(rules['lift']-1)
    rules['discount_cost'] = rules['est_new_orders']*AVG_ORDER*DISCOUNT
    rules['net_revenue_gain'] = rules['revenue_from_lift']-rules['discount_cost']
    rules.sort_values('net_revenue_gain', ascending=False).to_csv('results/revenue_simulation.csv', index=False)
    print(f"  ✓ Revenue simulation done")

# =============================================================================
# 7. PROMOTION EFFICIENCY ANALYSIS
# =============================================================================
print("\n[7/7] Promotion Efficiency...")
seg_spend = rfm.groupby('segment')['user_id'].count().reset_index().rename(columns={'user_id':'customers'})
seg_spend['avg_spend'] = rfm.groupby('segment')['total_spend'].mean().values
BLANKET = 0.15; TARGETED = 0.10; LIFT = 1.25
seg_spend['blanket_cost'] = seg_spend['customers']*seg_spend['avg_spend']*BLANKET
seg_spend['targeted_cost'] = seg_spend['customers']*seg_spend['avg_spend']*TARGETED
seg_spend['targeted_gain'] = seg_spend['customers']*seg_spend['avg_spend']*(LIFT-1) - seg_spend['targeted_cost']
seg_spend['advantage'] = seg_spend['targeted_gain'] - (-seg_spend['blanket_cost'])
seg_spend.to_csv('results/promotion_efficiency.csv', index=False)
print("  ✓ Promotion efficiency complete")

print("\nPipeline complete!")
print("="*60)
print(f"Total time: {time.time()-t1:.1f}s")
print("="*60)