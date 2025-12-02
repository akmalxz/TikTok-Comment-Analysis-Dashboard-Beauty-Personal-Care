import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import seaborn as sns
import plotly.express as px
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TikTok Comment Analysis – Beauty & Personal Care",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. CUSTOM CSS (Hard-coded Light Theme)
# ==========================================
CUSTOM_CSS = """
<style>
/* Force light background + dark text */
html, body, [class*="css"]  { background-color:#ffffff !important; color:#111111 !important; }

/* Headings */
h1,h2,h3,h4 { color:#111111; }

/* Card */
.card {
  border:1px solid #e6e6e6;
  border-radius:14px;
  padding:16px 18px;
  background:#f9fafb;
  box-shadow:0 1px 2px rgba(0,0,0,0.04);
}

/* KPI */
.kpi-label { font-size:14px; color:#5f6368; font-weight:600; }
.kpi-value { font-size:28px; font-weight:800; color:#111111; }

/* Divider */
.hr { border:0; border-top:1px solid #e6e6e6; margin:24px 0; }

/* Tables */
table.styled-table {
  width:100%; border-collapse:collapse; font-family:"Segoe UI", sans-serif;
  font-size:16px; background:#ffffff; color:#111111;
}
.styled-table thead th {
  background:#f2f4f7; color:#111111; text-align:left; font-weight:700;
  border-bottom:1px solid #e6e6e6; padding:10px 12px;
}
.styled-table td {
  border-bottom:1px solid #e6e6e6; padding:10px 12px; color:#111111;
}
.styled-table tbody tr:hover { background:#fbfbfc; }

/* Tab Styling */
.stTabs [role="tablist"] button p,
.stTabs [role="tablist"] button span,
.stTabs [role="tablist"] button > div[data-testid="stMarkdownContainer"] p {
  font-size: 22px !important;
  font-weight: 800 !important;
  color: white !important;
  margin: 0 !important;
  line-height: 1.2 !important;
}

/* Word Table Styling */
.word-card { border-radius:10px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,0.06); margin-top:6px; }
.word-table { width:100%; border-collapse:collapse; font-size:16px; }
.word-table th { text-align:left; padding:8px 12px; font-size:17px; }
.word-table td { padding:6px 12px; font-weight:500; }
.word-positive { background:#e8f5e9; color:#2E7D32; }
.word-neutral  { background:#fff3e0; color:#E67514; }
.word-negative { background:#ffebee; color:#C62828; }
.word-table tr:nth-child(even) td { filter:brightness(0.98); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING (Cached)
# ==========================================
@st.cache_data
def load_all_data():
    """
    Loads all CSVs and performs basic cleaning.
    Cached to improve performance on interactions.
    """
    # Load raw
    s_df = pd.read_csv("sentiment.csv")
    t_df = pd.read_csv("topic_info_gpt_final.csv")
    r_df = pd.read_csv("final_revenue_dataset.csv")
    m_df = pd.read_csv("final_cleaned_tiktok_dataset.csv")
    tok_df = pd.read_csv("distinctive_tokens.csv")

    # Clean Topic DF
    t_df = t_df[t_df['Topic'] != -1]

    # Helper to clean numeric columns if they have strings/NaNs
    def clean_numeric(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    
    # Ensure numeric types for critical columns
    r_cols = ["Revenue", "Followers", "RevenuePerFollower", "RevenuePerVideo", "RevenuePerLive"]
    r_df = clean_numeric(r_df, r_cols)

    m_cols = ["playCount", "likesCount", "commentCount", "shareCount", 
              "savesCount", "duration", "hashtagCount", "engagement_rate"]
    m_df = clean_numeric(m_df, m_cols)

    return s_df, t_df, r_df, m_df, tok_df

# Load Data
sentiment_df, topic_df, df_revenue, df_metadata, df_tokens = load_all_data()

# ==========================================
# 4. CONSTANTS & PALETTES
# ==========================================
GREEN = "#2E7D32"
GREY  = "#9E9E9E"
RED   = "#C62828"
BLUE  = "#1F77B4"
TEAL  = "#2AA198"
SENT_COLORS = [GREEN, "#E67514", RED] # Pos, Neu, Neg colors (Hex)

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================
st.title("TikTok Comment Analysis Dashboard – Beauty & Personal Care")

tab_overview, tab_metadata, tab_sentiment, tab_topics = st.tabs(
    ["Revenue Analytics", "Video Analytics", "Sentiment Analysis", "Topic Modeling"]
)

# ------------------------------------------
# TAB 1: Revenue Analytics
# ------------------------------------------
with tab_overview:
    st.subheader("Revenue Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="card">
          <div class="kpi-label">👥 Total Influencers</div>
          <div class="kpi-value">{len(df_revenue):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        rev_mean = df_revenue['Revenue'].mean()
        st.markdown(f"""
        <div class="card">
          <div class="kpi-label">💰 Avg Revenue (MYR)</div>
          <div class="kpi-value">{rev_mean:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        fol_mean = df_revenue['Followers'].mean()
        st.markdown(f"""
        <div class="card">
          <div class="kpi-label">📣 Avg Followers</div>
          <div class="kpi-value">{fol_mean:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    # Top 10 Table
    df_rev = df_revenue.copy()
    df_top10 = (
        df_rev[["Nickname","Revenue","RevenuePerFollower","RevenuePerVideo","RevenuePerLive"]]
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    # Anonymize/Label
    df_top10["Nickname"] = [f"Influencer{i+1:02d}" for i in range(len(df_top10))]

    st.markdown("<h3 style='font-size:24px; font-weight:700;'>Top 10 Influencers by Revenue</h3>", unsafe_allow_html=True)
    st.markdown(df_top10.to_html(classes="styled-table", index=False, border=0), unsafe_allow_html=True)

    # Correlation Heatmap
    st.markdown("<h3 style='font-size:24px; font-weight:700; margin-top:16px;'>Revenue Variable Correlations</h3>", unsafe_allow_html=True)
    
    corr_rev = df_rev.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr_rev, annot=True, cmap=sns.light_palette(BLUE, as_cmap=True), fmt=".2f",
                ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("") ; ax.set_xlabel("") ; ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig) # Cleanup

# ------------------------------------------
# TAB 2: Video Analytics
# ------------------------------------------
with tab_metadata:
    st.subheader("Video Metadata Summary")

    meta_cols = [
        "playCount","likesCount","commentCount","shareCount",
        "savesCount","duration","hashtagCount","engagement_rate"
    ]
    
    # 1. Summary Stats Table
    meta_summary = df_metadata[meta_cols].describe().T.round(2)
    st.markdown("<h3 style='font-size:24px; font-weight:700;'>Summary Statistics</h3>", unsafe_allow_html=True)
    st.markdown(meta_summary.to_html(classes="styled-table", border=0), unsafe_allow_html=True)

    # 2. Correlation Matrix
    st.markdown("<h3 style='font-size:24px; font-weight:700; margin-top:16px;'>Correlation Matrix</h3>", unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(8,5))
    sns.heatmap(df_metadata[meta_cols].corr(), annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                ax=ax_corr, cbar_kws={"shrink": .8})
    ax_corr.set_title("") ; ax_corr.set_xlabel("") ; ax_corr.set_ylabel("")
    st.pyplot(fig_corr)
    plt.close(fig_corr) # Cleanup

    # 3. Hashtag vs Engagement (Plotly)
    st.markdown("<h2 style='margin:0;'>Average Engagement Rate by Hashtag Count</h2>", unsafe_allow_html=True)
    
    df_meta = df_metadata.copy()
    # Apply logic to normalize percentage if needed
    if df_meta["engagement_rate"].dropna().max() > 1.5:
        df_meta["engagement_rate"] = df_meta["engagement_rate"] / 100.0

    agg = (
        df_meta.dropna(subset=["hashtagCount","engagement_rate"])
            .groupby("hashtagCount", as_index=False)["engagement_rate"].mean()
            .sort_values("hashtagCount")
    )

    max_y = float(agg["engagement_rate"].max()) if not agg.empty else 0.02

    fig = px.bar(
        agg, x="hashtagCount", y="engagement_rate",
        color="engagement_rate",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(
        marker_line_color="rgba(0,0,0,0.25)",
        marker_line_width=0.8,
        opacity=1,
        hovertemplate="Hashtags: %{x}<br>Avg ER: %{y:.2%}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=40, r=60, t=10, b=40),
        font=dict(size=16, color="black"),
        xaxis=dict(title="Hashtag Count", showgrid=False, linecolor="black"),
        yaxis=dict(title="Average Engagement Rate", tickformat=".1%", showline=True, linecolor="black"),
        coloraxis_colorbar=dict(title="Engagement Rate", tickformat=".1%")
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 3: Sentiment Analysis
# ------------------------------------------
with tab_sentiment:
    st.subheader("Sentiment Analysis Insights")

    labels = ["Positive", "Neutral", "Negative"]
    vc = sentiment_df["gpt_label"].value_counts()
    counts = [int(vc.get(l, 0)) for l in labels]
    
    # --- Visuals ---
    left, right = st.columns(2, gap="large")

    # Pie Chart
    with left:
        fig_pie, ax_pie = plt.subplots(figsize=(5.5, 3.0), dpi=160)
        wedges, texts, autotexts = ax_pie.pie(
            counts, labels=labels, colors=SENT_COLORS,
            startangle=140, autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            textprops={"fontsize": 7, "color": "black"},
            wedgeprops={"edgecolor": "black", "linewidth": 0.6},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontsize(7)
            t.set_fontweight("bold")
        ax_pie.set_title("Sentiment Distribution", fontsize=14, color="black", pad=20)
        fig_pie.patch.set_facecolor("white")
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    # Bar Chart
    with right:
        fig_bar, ax_bar = plt.subplots(figsize=(8.5, 5.4), dpi=160)
        bars = ax_bar.bar(labels, counts, color=SENT_COLORS, edgecolor="black", linewidth=0.6)
        
        # Annotate bars
        for b in bars:
            h = b.get_height()
            ax_bar.annotate(f"{h:,}", (b.get_x() + b.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=12, color="black", xytext=(0, 6), textcoords="offset points")

        ax_bar.set_title("Number of Comments by Sentiment", fontsize=14, color="black", pad=20)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.yaxis.grid(True, linestyle="--", linewidth=1, alpha=0.35, color="#c8c8c8")
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # --- Key Terms Tables ---
    # Configuration for filtering
    TOP_N_POSITIVE = 16
    TOP_N_NEUTRAL  = 12
    BRAND_STOP = {"medicube", "numbuzin"}
    MISC_STOP  = {"kod", "masak", "sushi", "gading", "keli", "merajuk"}
    PROPER_NAME_STOP = {"mek", "sue", "rozy", "dino", "azad", "yaya", "pak"}
    NEGATIVE_KEEP = {
        "loudly_crying_face","crying_face","pleading_face","mending_heart","anxious_face_with_sweat",
        "menangis","pusing","sakit","bahaya","palsu","terkejut","cirit","pemasaran"
    }

    def pick_tokens(df, label):
        sub = df[df["label"] == label].copy()
        # Prioritize manually selected tokens if column exists
        if "selected" in sub.columns:
            keep = (sub["selected"].astype(str).str.lower().isin({"1","true","yes"}))
            chosen = sub.loc[keep].sort_values("z", ascending=False)["token"].tolist()
            if chosen: return chosen
            
        sub = sub.sort_values("z", ascending=False)
        if label == "Positive":
            def ok(t): return str(t).lower() not in BRAND_STOP|MISC_STOP|PROPER_NAME_STOP
            return [t for t in sub["token"] if ok(t)][:TOP_N_POSITIVE]
        if label == "Neutral":
            def ok(t): return str(t).lower() not in BRAND_STOP|MISC_STOP
            return [t for t in sub["token"] if ok(t)][:TOP_N_NEUTRAL]
        if label == "Negative":
            return [t for t in sub["token"] if t in NEGATIVE_KEEP]
        return []

    pos_terms = pick_tokens(df_tokens, "Positive")
    neu_terms = pick_tokens(df_tokens, "Neutral")
    neg_terms = pick_tokens(df_tokens, "Negative")

    def render_vertical_table(title, tokens, sentiment_class):
        html = f"<div class='word-card'><table class='word-table {sentiment_class}'>"
        html += f"<tr><th>{title}</th></tr>"
        for t in tokens:
            html += f"<tr><td>{t}</td></tr>"
        html += "</table></div>"
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("### Key Terms by Sentiment")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1: render_vertical_table("Positive", pos_terms, "word-positive")
    with c2: render_vertical_table("Neutral",  neu_terms, "word-neutral")
    with c3: render_vertical_table("Negative", neg_terms, "word-negative")

# ------------------------------------------
# TAB 4: Topic Modeling
# ------------------------------------------
with tab_topics:
    st.subheader("Topic Modeling Insights")

    topic_df_sorted = topic_df.sort_values(by="Count", ascending=False).reset_index(drop=True)
    
    topic_labels = [
        f"T{int(r.Topic)}" if pd.isna(getattr(r, "Name", None)) or not str(r.Name).strip()
        else f"T{int(r.Topic)} – {str(r.Name)}"
        for r in topic_df_sorted.itertuples()
    ]

    PALETTE = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
               "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"]
    pie_colors = [PALETTE[i % len(PALETTE)] for i in range(len(topic_df_sorted))]

    col_pie, col_bar = st.columns(2, gap="large")

    # Topic Pie
    with col_pie:
        fig_tpie, ax_tpie = plt.subplots(figsize=(5.5, 3.0), dpi=160)
        wedges, texts, autotexts = ax_tpie.pie(
            topic_df_sorted["Count"].tolist(),
            labels=[f"T{int(t)}" for t in topic_df_sorted["Topic"]],
            colors=pie_colors, startangle=140,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            textprops={"fontsize": 10, "color": "black"},
            wedgeprops={"edgecolor": "black", "linewidth": 0.6},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontsize(7)
            t.set_fontweight("bold")
        ax_tpie.set_title("Share of Comments by Topic", fontsize=14, color="black")
        fig_tpie.patch.set_facecolor("white")
        st.pyplot(fig_tpie)
        plt.close(fig_tpie)

    # Topic Bar
    with col_bar:
        fig_tbar, ax_tbar = plt.subplots(figsize=(8.5, 5.3), dpi=160)
        ax_tbar.bar(topic_labels, topic_df_sorted["Count"], color="#1F77B4", edgecolor="black", linewidth=0.6)

        for x, h in zip(range(len(topic_df_sorted)), topic_df_sorted["Count"]):
            ax_tbar.annotate(f"{int(h):,}", (x, h),
                             ha="center", va="bottom", fontsize=10, color="black",
                             xytext=(0, 6), textcoords="offset points")

        ax_tbar.set_title("Number of Comments per Topic", fontsize=14, color="black")
        ax_tbar.spines["top"].set_visible(False)
        ax_tbar.spines["right"].set_visible(False)
        ax_tbar.yaxis.grid(True, linestyle="--", linewidth=1, alpha=0.35, color="#c8c8c8")
        st.pyplot(fig_tbar)
        plt.close(fig_tbar)

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    # Topic Drilldown (Cards + WordClouds)
    for i in range(0, len(topic_df_sorted), 2):
        c1, c2 = st.columns(2, gap="large")
        # Handle case where odd number of topics exists
        rows = topic_df_sorted.iloc[i:i+2].itertuples()
        
        for col, row in zip([c1, c2], rows):
            with col:
                st.markdown(f"**Topic {row.Topic}: {row.Name}** \n*{row.Count} comments*")

                # Parse lists safely
                try:
                    keywords = ast.literal_eval(row.Representation)
                except Exception:
                    keywords = str(row.Representation).split(",")
                
                try:
                    comments = ast.literal_eval(row.Representative_Docs)
                except Exception:
                    comments = [str(row.Representative_Docs)]

                # Generate WordCloud
                text_string = " ".join(map(str, comments))
                wc = WordCloud(width=600, height=280, background_color="white",
                               max_words=100, colormap="viridis").generate(text_string)
                
                f_wc, a_wc = plt.subplots(figsize=(8.5, 4), dpi=150)
                a_wc.imshow(wc, interpolation="bilinear")
                a_wc.axis("off")
                st.pyplot(f_wc)
                plt.close(f_wc) # Important cleanup

                st.markdown("**Top Keywords**")
                st.markdown(", ".join(map(str, keywords[:10])))

                st.markdown("**Sample Comments**")
                for c in comments[:5]:
                    st.markdown(f"- {c}")
