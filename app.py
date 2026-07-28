# ============================
# IMPORT LIBRARIES
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)
st.image(
    "E-COMMERCE banner.jpeg",
    use_container_width=
)

# ============================
# CUSTOM CSS
# ============================
theme = st.sidebar.radio(
    "🌙 Theme",
    ["Light", "Dark"]
)

if theme == "Dark":
    bg = "#0E1117"
    text = "white"
    card = "#262730"
else:
    bg = "#F4F8FB"
    text = "black"
    card = "white"
    
st.markdown("""
<style>

.main{
    background-color:#F4F8FB;
}

h1,h2,h3{
    color:#0F4C81;
}

.stButton>button{
    background:linear-gradient(90deg,#0F4C81,#00B4D8);
    color:white;
    font-weight:bold;
    border-radius:10px;
    height:50px;
    width:100%;
    border:none;
}

.stDownloadButton>button{
    background:#198754;
    color:white;
    font-weight:bold;
    border-radius:10px;
}

.metric-box{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.15);
    text-align:center;
}

.upload-box{
    background:#E3F2FD;
    padding:20px;
    border-radius:15px;
    border:2px dashed #2196F3;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================

st.markdown("""
# 🛒 Market Basket Analysis Dashboard

### Discover Product Associations using the Apriori Algorithm

Analyze customer purchasing behaviour and identify products
that are frequently bought together.
""")

st.info("""
📌 **Project Name:** Market Basket Analysis

📊 **Algorithm:** Apriori

📚 **Library:** mlxtend

🎯 **Objective:** Find frequent itemsets and generate product recommendations.
""")

# ============================
# SIDEBAR
# ============================

st.sidebar.image(
    "https://img.icons8.com/color/480/shopping-cart.png",
    width=140
)

st.sidebar.title("⚙ Analysis Settings")

support = st.sidebar.slider(
    "Minimum Support",
    0.01,
    0.20,
    0.02,
    0.01
)

confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.10,
    1.00,
    0.50,
    0.05
)

lift = st.sidebar.slider(
    "Minimum Lift",
    1.00,
    10.00,
    1.20,
    0.10
)

st.sidebar.markdown("---")

st.sidebar.success("Ready for Analysis")

# ============================
# DATASET UPLOAD
# ============================

st.markdown("## 📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload Online Retail CSV File",
    type=["csv"]
)

if uploaded_file is None:

    st.warning("Please upload the Online Retail dataset to continue.")

    st.stop()

# ============================
# LOAD DATASET
# ============================

df = pd.read_csv(uploaded_file)

st.success("Dataset Uploaded Successfully")

# ============================
# DASHBOARD METRICS
# ============================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Unique Products",
        df["Description"].nunique()
    )

with col4:
    st.metric(
        "Unique Transactions",
        df["InvoiceNo"].nunique()
    )

st.markdown("---")

# ============================
# DATA PREVIEW
# ============================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.markdown("---")
# ============================
# DATA CLEANING
# ============================

st.subheader("🧹 Data Cleaning")

# Remove missing values
df = df.dropna()

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.contains("C")]

# Remove negative quantities
df = df[df["Quantity"] > 0]

# Keep only required columns
df = df[["InvoiceNo", "Description", "Quantity"]]

st.success("✅ Data cleaning completed successfully.")

# ============================
# CREATE BASKET MATRIX
# ============================

st.subheader("🛒 Creating Basket Matrix")

basket = (
    df.groupby(["InvoiceNo", "Description"])["Quantity"]
      .sum()
      .unstack()
      .fillna(0)
)

# Convert quantities to binary values
basket = basket.apply(lambda x: x > 0)

st.success("✅ Basket matrix created.")

st.write("Basket Matrix Shape:", basket.shape)

# ============================
# FREQUENT ITEMSETS
# ============================

st.subheader("📦 Frequent Itemset Mining")

with st.spinner("Running Apriori Algorithm..."):

    frequent_itemsets = apriori(
        basket,
        min_support=support,
        use_colnames=True,
        low_memory=True
    )

st.success("✅ Frequent Itemsets Generated")

st.metric(
    "Frequent Itemsets",
    len(frequent_itemsets)
)
frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
    lambda x: ", ".join(list(x))
)
st.dataframe(
    frequent_itemsets.head(20),
    use_container_width=True
)

# ============================
# ASSOCIATION RULES
# ============================

st.subheader("🔗 Association Rules")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=confidence
)

rules = rules[rules["lift"] >= lift]

st.metric(
    "Association Rules",
    len(rules)
)

# Convert frozensets to readable text
rules["antecedents"] = rules["antecedents"].apply(
    lambda x: ", ".join(list(x))
)

rules["consequents"] = rules["consequents"].apply(
    lambda x: ", ".join(list(x))
)

st.dataframe(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ],
    use_container_width=True
)
st.subheader("🔍 Search Product Recommendations")

product = st.selectbox(
    "Choose Product",
    sorted(df["Description"].unique())
)

result = rules[
    rules["antecedents"].str.contains(
        product,
        case=False,
        na=False
    )
]

if len(result):

    st.success("Recommended Products")

    st.dataframe(
        result[
            ["antecedents",
             "consequents",
             "confidence",
             "lift"]
        ]
    )

else:

    st.warning("No recommendation found.")

# ============================
# TOP ASSOCIATIONS
# ============================

st.subheader("🏆 Top Product Associations")

top_rules = rules.sort_values(
    by="lift",
    ascending=False
).head(10)

st.dataframe(
    top_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ],
    use_container_width=True
)

st.markdown("---")
# ============================
# INTERACTIVE VISUALIZATIONS
# ============================

st.markdown("## 📊 Interactive Dashboard")

tab1, tab2, tab3 = st.tabs([
    "📈 Frequent Itemsets",
    "🎯 Association Rules",
    "💡 Product Recommendations"
])

# ----------------------------------
# TAB 1
# ----------------------------------
with tab1:

    st.subheader("Top Frequent Itemsets")

    top_items = frequent_itemsets.sort_values(
        by="support",
        ascending=False
    ).head(15)

    top_items["Itemset"] = top_items["itemsets"].astype(str)

    fig = px.bar(
        top_items,
        x="support",
        y="Itemset",
        orientation="h",
        title="Top 15 Frequent Itemsets"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ----------------------------------
# TAB 2
# ----------------------------------
with tab2:

    st.subheader("Support vs Confidence")

    fig = px.scatter(
        rules,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_data=[
            "antecedents",
            "consequents"
        ],
        title="Association Rules"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ----------------------------------
# TAB 3
# ----------------------------------
with tab3:

    st.subheader("Recommended Product Pairs")

    recommendations = rules.sort_values(
        by="lift",
        ascending=False
    )

    for _, row in recommendations.head(10).iterrows():

        st.success(
            f"""
🛒 **Buy:** {row['antecedents']}

➡️ **Recommend:** {row['consequents']}

📈 Confidence: {row['confidence']:.2f}

⭐ Lift: {row['lift']:.2f}
"""
        )

# ============================
# DOWNLOAD SECTION
# ============================

st.markdown("---")
st.subheader("📥 Download Results")

col1, col2 = st.columns(2)

with col1:

    frequent_csv = frequent_itemsets.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Frequent Itemsets",
        data=frequent_csv,
        file_name="frequent_itemsets.csv",
        mime="text/csv"
    )

with col2:

    rules_csv = rules.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Association Rules",
        data=rules_csv,
        file_name="association_rules.csv",
        mime="text/csv"
    )

# ============================
# PROJECT SUMMARY
# ============================

st.markdown("---")

st.subheader("📌 Analysis Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.info(f"📦 Frequent Itemsets\n\n**{len(frequent_itemsets)}**")

with c2:
    st.info(f"🔗 Association Rules\n\n**{len(rules)}**")

with c3:
    st.info(f"🛍 Products\n\n**{basket.shape[1]}**")

# ============================
# FOOTER
# ============================

st.markdown("---")

st.markdown("""
<div style='text-align:center;
padding:20px;
background:linear-gradient(90deg,#1E3C72,#2A5298);
border-radius:12px;
color:white;'>

<h2>🛒 Market Basket Analysis Dashboard</h2>

<p>Developed using <b>Python</b>, <b>Streamlit</b>, <b>Pandas</b>, <b>Plotly</b> and <b>MLxtend</b></p>

<p>📊 Apriori Algorithm | Association Rule Mining | Product Recommendation System</p>

<p>© 2026 All Rights Reserved</p>

</div>
""", unsafe_allow_html=True)   
