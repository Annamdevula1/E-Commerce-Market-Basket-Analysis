import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv("Online Retail.csv")

# Preprocessing
df = df.dropna()
df = df[~df["InvoiceNo"].astype(str).str.contains("C")]
df = df[df["Quantity"] > 0]
df = df[["InvoiceNo", "Description", "Quantity"]]

# Create basket
basket = (
    df.groupby(["InvoiceNo", "Description"])["Quantity"]
      .sum()
      .unstack()
      .fillna(0)
)

basket = basket.apply(lambda col: col.map(lambda x: 1 if x > 0 else 0))

# Train Apriori model
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(frequent_itemsets.head())
print(rules.head())
