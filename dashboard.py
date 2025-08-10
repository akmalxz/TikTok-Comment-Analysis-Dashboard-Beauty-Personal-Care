import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import seaborn as sns
import plotly.express as px
from collections import Counter

# Load data
sentiment_df = pd.read_csv("sentiment.csv")
topic_df = pd.read_csv("topic_info_gpt_final.csv")
topic_df = topic_df[topic_df['Topic'] != -1]  # Remove noise topic
df_revenue = pd.read_csv("final_revenue_dataset.csv")
df_metadata = pd.read_csv("final_cleaned_tiktok_dataset.csv")

# --- Dashboard Title ---
st.set_page_config(layout="wide")
st.title("TikTok Comment Analysis Dashboard – Beauty & Personal Care")

# --- Header Metrics ---

col1, col2, col3 = st.columns(3)
#col1.metric("Total Comments", f"{len(sentiment_df):,}")
sent_counts = sentiment_df['gpt_label'].value_counts()
#col2.metric("Positive %", f"{(sent_counts.get('Positive', 0) / len(sentiment_df)) * 100:.1f}%")
#col3.metric("Topics Identified", f"{len(topic_df)}")

colors = {
        'Positive': '#347433',
        'Neutral': '#D7D7D7',
        'Negative': '#DC2525'
    }

# --- Section 1.1 Revenue Summary ---
st.header("1. Preliminary Analytics")
st.subheader("1.1 Revenue Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        ">
            <div style="font-size:16px; font-weight:600; color:white;">👥 Total Influencers</div>
            <div style="font-size:25px; font-weight:700; color:white;">{:,}</div>
        </div>
    """.format(len(df_revenue)), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        ">
            <div style="font-size:16px; font-weight:600; color:white;">💰 Avg Revenue (MYR)</div>
            <div style="font-size:25px; font-weight:700; color:white;">{:,.0f}</div>
        </div>
    """.format(df_revenue['Revenue'].mean()), unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        ">
            <div style="font-size:16px; font-weight:600; color:white;">📣 Avg Followers</div>
            <div style="font-size:25px; font-weight:700; color:white;">{:,.0f}</div>
        </div>
    """.format(df_revenue['Followers'].mean()), unsafe_allow_html=True)



# Extract and format the top 10
df_revenue = df_revenue.copy()  # avoid modifying original if needed elsewhere
df_revenue['Nickname'] = ['Influencer{:02d}'.format(i+1) for i in range(len(df_revenue))]

df_top10 = df_revenue[['Nickname', 'Revenue', 'RevenuePerFollower', 'RevenuePerVideo', 'RevenuePerLive']] \
    .sort_values(by='Revenue', ascending=False).head(10)

# Convert to HTML with custom class
html_top10 = df_top10.to_html(classes='styled-table', index=False, border=0)

# Apply CSS styling
st.markdown("""
    <style>
    .styled-table {
        font-size: 19px;
        font-family: "Segoe UI", sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .styled-table thead {
        background-color: #black;
        font-weight: bold;
    }
    .styled-table td, .styled-table th {
        padding: 8px 12px;
        text-align: right;
        border-bottom: 1px solid #ddd;
    }
    .styled-table th {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Display the table
st.markdown(html_top10, unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("**Revenue Variable Correlations**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_revenue.corr(numeric_only=True), annot=True, cmap='Blues', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Section 1.2 Metadata Summary ---
st.header("1.2 Metadata Summary (TikTok Videos)")

# --- Summary Stats Table ---
st.subheader("Summary Statistics")

metadata_summary = df_metadata[[
    'playCount', 'likesCount', 'commentCount', 'shareCount',
    'savesCount', 'duration', 'hashtagCount', 'engagement_rate'
]].describe().T.round(2)

# Format the DataFrame into HTML with styling
styled_html = metadata_summary.to_html(classes='styled-table', border=0)

# Inject CSS
st.markdown("""
    <style>
    .styled-table {
        font-size: 19px;
        font-family: "Segoe UI", sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .styled-table thead {
        background-color: #black;
        font-weight: bold;
    }
    .styled-table td, .styled-table th {
        padding: 8px 12px;
        text-align: right;
        border-bottom: 1px solid #ddd;
    }
    .styled-table th {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(styled_html, unsafe_allow_html=True)


# --- Correlation Heatmap ---
st.subheader("Correlation Matrix")

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_metadata[[
        'playCount', 'likesCount', 'commentCount', 'shareCount',
        'savesCount', 'duration', 'hashtagCount', 'engagement_rate'
    ]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# --- Hashtag Count vs Engagement Rate ---
st.subheader("Hashtag Count vs Engagement Rate")

hashtag_engagement = df_metadata.groupby("hashtagCount")["engagement_rate"].mean().reset_index()

fig_hashtag = px.bar(
    hashtag_engagement,
    x="hashtagCount",
    y="engagement_rate",
    title="Average Engagement Rate by Hashtag Count",
    labels={"hashtagCount": "Hashtag Count", "engagement_rate": "Avg Engagement Rate"},
    color="engagement_rate",
    color_continuous_scale="viridis"
)

# Update layout for better styling
fig_hashtag.update_layout(
    title_font_size=18,
    title_x=0.05,
    font=dict(size=14),
    xaxis=dict(
        title="Hashtag Count",
        tickfont=dict(size=12),
        #titlefont=dict(size=14),
        tickmode="linear"
    ),
    yaxis=dict(
        title="Average Engagement Rate",
        tickfont=dict(size=12),
        #titlefont=dict(size=14),
        tickformat=".2%",  # optional: percentage format
        gridcolor='rgba(200,200,200,0.2)'
    ),
    plot_bgcolor="white",
    coloraxis_colorbar=dict(
        title="Engagement Rate",
        tickformat=".2%"
    )
)

# Optional: Custom hover formatting
fig_hashtag.update_traces(
    hovertemplate='Hashtags: %{x}<br>Avg Engagement Rate: %{y:.2%}<extra></extra>',
    marker_line_color='rgba(58, 58, 58, 0.3)',
    marker_line_width=0.8
)

st.plotly_chart(fig_hashtag, use_container_width=True)

# --- Section 2: Sentiment Overview ---
st.header("2. Sentiment Overview")

# Context box for metadata
with st.expander("📘 About This Section", expanded=True):
    st.markdown("""
    <ul style="font-size:17px; line-height:1.6;">
        <li><strong>Data Size:</strong> 2,016 unique comments (after deduplication)</li>
        <li><strong>Sentiment Model:</strong> GPT-4o</li>
        <li><strong>Label Types:</strong> Positive, Neutral, Negative</li>
        <li><strong>Comment Source:</strong> TikTok (Beauty & Personal Care, Jan–Aug 2024)</li>
    </ul>
    """, unsafe_allow_html=True)

# Clarification caption below
st.markdown(
    '<p style="font-size:17px; color:lightgray;">'
    'Note: This dashboard analyzes a sample of <strong>2,016 unique comments</strong> after merging identical entries by text and video ID. '
    'The original dataset includes over <strong>34,597 raw comments</strong> collected from top TikTok influencers.'
    '</p>',
    unsafe_allow_html=True
)

# Sentiment selection for highlight
st.markdown('<div style="font-size:22px; font-weight:600; margin-bottom:6px;">Highlight Sentiment</div>', unsafe_allow_html=True)

# Selectbox with no label
sentiment_options = ['None (show all equally)', 'Positive', 'Neutral', 'Negative']
selected_sentiment = st.selectbox(label="", options=sentiment_options, index=0)


# Sentiment counts and labels
labels = ['Positive', 'Neutral', 'Negative']
counts = [sent_counts.get(label, 0) for label in labels]

# Define colors and highlight logic
base_colors = {
    'Positive': '#347433',
    'Neutral': '#E67514',
    'Negative': '#DC2525'
}
dim_color = '#E0E0E0'  # Light gray for de-emphasis

# Apply highlight color logic
highlighted_colors = [
    base_colors[label] if selected_sentiment in ('None (show all equally)', label)
    else dim_color
    for label in labels
]

# Layout
col1, col2 = st.columns([1, 2])

# --- Pie Chart ---
with col1:
    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots(figsize=(4, 4))

    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=labels,
        colors=highlighted_colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 8},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)

    ax1.set_title("Sentiment Distribution", fontsize=10)
    ax1.axis("equal")
    fig1.tight_layout()
    st.pyplot(fig1)

# --- Bar Chart ---
with col2:
    st.subheader("Sentiment Count")
    fig2, ax2 = plt.subplots(figsize=(8, 4.1))
    bars = ax2.bar(labels, counts, color=highlighted_colors)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    ax2.set_ylabel("Comment Count", fontsize=11)
    ax2.set_title("Number of Comments by Sentiment", fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    fig2.tight_layout()
    st.pyplot(fig2)

st.markdown("<hr style='border:1px solid #555; margin:30px 0;'>", unsafe_allow_html=True)

# --- Section 2: Topic Modeling Insights ---
st.header("3. Topic Modeling Insights")

# Sort topics
topic_df_sorted = topic_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

# First row: Chart (left) and empty (right)
col_chart, col_empty = st.columns([2, 1])
with col_chart:
    st.subheader("📊 Topic Distribution")

    fig_dist, ax_dist = plt.subplots(figsize=(7, 4), dpi=500)
    sns.barplot(
        x='Topic',
        y='Count',
        data=topic_df_sorted,
        palette="Blues_d",
        ax=ax_dist
    )

    # Value labels on bars
    for p in ax_dist.patches:
        height = int(p.get_height())
        ax_dist.annotate(f'{height}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom',
                         fontsize=9, color='black',
                         xytext=(0, 5),
                         textcoords='offset points')

    # Styling
    ax_dist.set_ylabel("Comment Count", fontsize=10)
    ax_dist.set_xlabel("Topic", fontsize=10)
    ax_dist.set_title("Number of Comments per Topic", fontsize=11, weight='bold')
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.grid(axis='y', linestyle='--', alpha=0.5)
    fig_dist.tight_layout()

    st.pyplot(fig_dist)

st.markdown("<hr style='border:1px solid #555; margin:30px 0;'>", unsafe_allow_html=True)

# Next: Loop through topics two at a time, side by side
for i in range(0, len(topic_df_sorted), 2):
    col1, col2 = st.columns(2)

    for col, row in zip([col1, col2], topic_df_sorted.iloc[i:i+2].itertuples()):
        with col:
            st.markdown(f"### Topic {row.Topic}: {row.Name} ({row.Count} comments)")

            # --- Load Keywords ---
            try:
                keywords = ast.literal_eval(row.Representation)
            except:
                keywords = row.Representation.split(",")

            # --- Load Sample Comments (once only) ---
            try:
                comments = ast.literal_eval(row.Representative_Docs)
            except:
                comments = [row.Representative_Docs]

            # --- Word Cloud from Sample Comments ---
            text_string = " ".join(comments)
            wordcloud = WordCloud(
                width=500,
                height=250,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text_string)

            fig_wc, ax_wc = plt.subplots(figsize=(10, 5), dpi=450)
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            # --- Top Keywords ---
            st.markdown(
                f"<p style='font-size:19px; color:white;'><strong>Top Keywords</strong></p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='font-size:21px; color:#aaa;'>{', '.join(keywords[:10])}</p>",
                unsafe_allow_html=True
            )

            # --- Sample Comments ---
            st.markdown("**Sample Comments:**")
            for comment in comments[:5]:
                st.markdown(f"- {comment}", unsafe_allow_html=True)
