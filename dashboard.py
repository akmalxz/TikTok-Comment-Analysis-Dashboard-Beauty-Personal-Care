import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import seaborn as sns

# Load data
sentiment_df = pd.read_csv("sentiment_cleaned.csv")
topic_df = pd.read_csv("topic_info_gpt_final.csv")
topic_df = topic_df[topic_df['Topic'] != -1]  # Remove noise topic

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
        'Mixed': '#91C8E4',
        'Negative': '#DC2525'
    }
# --- Section 1: Sentiment Overview ---
st.header("1. Sentiment Overview")

# Context box for metadata
with st.expander("📘 About This Section", expanded=True):
    st.markdown("""
    <ul style="font-size:17px; line-height:1.6;">
        <li><strong>Data Size:</strong> 1,108 unique comments (after deduplication)</li>
        <li><strong>Sentiment Model:</strong> GPT-4o</li>
        <li><strong>Label Types:</strong> Positive, Neutral, Mixed, Negative</li>
        <li><strong>Comment Source:</strong> TikTok (Beauty & Personal Care, Jan–Aug 2024)</li>
    </ul>
    """, unsafe_allow_html=True)

# Clarification caption below
st.markdown(
    '<p style="font-size:17px; color:lightgray;">'
    'Note: This dashboard analyzes a sample of <strong>1,108 unique comments</strong> after merging identical entries by text and video ID. '
    'The original dataset includes over <strong>34,597 raw comments</strong> collected from top TikTok influencers.'
    '</p>',
    unsafe_allow_html=True
)

# Sentiment selection for highlight
st.markdown('<div style="font-size:22px; font-weight:600; margin-bottom:6px;">Highlight Sentiment</div>', unsafe_allow_html=True)

# Selectbox with no label
sentiment_options = ['None (show all equally)', 'Positive', 'Neutral', 'Mixed', 'Negative']
selected_sentiment = st.selectbox(label="", options=sentiment_options, index=0)


# Sentiment counts and labels
labels = ['Positive', 'Neutral', 'Mixed', 'Negative']
counts = [sent_counts.get(label, 0) for label in labels]

# Define colors and highlight logic
base_colors = {
    'Positive': '#347433',
    'Neutral': '#E67514',
    'Mixed': "#91C8E4",
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
st.header("2. Topic Modeling Insights")

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
