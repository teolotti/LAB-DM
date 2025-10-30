import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os

st.set_page_config(page_title="PatientLens", layout="wide")

# File path to the cleaned data
INPUT_CSV = 'data/PMC-Patients.csv'
PUBMED_CSV = 'data/pubmed_data.csv'
CLEAN_OUT = 'outputs/PMC_clean.parquet'
TRANS_OUT = 'outputs/transactions.parquet'
PATTERN_OUT = 'outputs/patterns.parquet'
TS_OUT = 'outputs/timeseries.parquet'
SNIPPET_OUT = 'outputs/snippets.parquet'
DEMOG_OUT = 'outputs/demographics.parquet'

# Data caching
@st.cache_data
def load_data():
    clean_df = pd.read_parquet(CLEAN_OUT)
    pattern_df = pd.read_parquet(PATTERN_OUT)
    timeseries_df = pd.read_parquet(TS_OUT)
    snippets_df = pd.read_parquet(SNIPPET_OUT)
    demog_df = pd.read_parquet(DEMOG_OUT)
    return clean_df, pattern_df, timeseries_df, snippets_df, demog_df

clean_df, pattern_df, timeseries_df, snippets_df, demog_df = load_data()

st.title("🩺 PatientLens — Visual Text Mining of clinical cases")

# Sidebar
st.sidebar.header("⚙️ Controlli globali")
if st.sidebar.button("Ricarica dati"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.info("You can search and select patterns, view time series, and explore patient demographics.")

# Layout
left, right = st.columns([3, 7])

# Left column
with left:
    st.header("📋 Patterns List")

    if 'support_count' not in pattern_df.columns:
        pattern_df['support_count'] = (pattern_df['support'] * len(clean_df)).astype(int)
    
    # Filter and search
    min_len = st.slider("Minimum itemset length", 1, 5, 1)
    min_freq = st.number_input("Minimum support count", min_value=1, value=10, step=1)
    order_by = st.selectbox("Order by", options=['support_count', 'len'], index=0)
    ascending = st.checkbox("Ascending order", value=False)

    # Dynamic filtering
    filt = pattern_df[
        (pattern_df['len'] >= min_len) &
        (pattern_df['support_count'] >= min_freq)
    ]

    filt = filt.sort_values(by=order_by, ascending=ascending).reset_index(drop=True)


    st.write(f"Total patients: {len(clean_df)}")


    # Display patterns
    options = [f"{r['pattern_label']} - {r['support_count']}" for _, r in filt.iterrows()]
    selected = st.multiselect("Select patterns to visualize", options, default=options[:1], max_selections=5)
    selected_patterns = [s.split(" - ")[0] for s in selected]

    st.dataframe(
        filt[['pattern_label', 'support_count', 'support', 'len']]
        .rename(columns={'pattern_label': 'Pattern', 'support_count': '#Patients', 'support': 'Support', 'len': '#Terms'}),
        use_container_width=True,
        height=300
    )
# Right column
with right:
    st.header("📈 Time Series Visualization")

    if not selected_patterns:
        st.info("Select patterns from the left panel to visualize their time series.")
    else:
        # Time aggregation
        st.subheader("Time Aggregation Options")
        freq_choice = st.radio(
            "Time Frequency:",
            ['Month', 'Trimester', 'Semester', 'Year'],
            horizontal=True
        )

        freq_map = {
            'Month': 'M',
            'Trimester': '3M',
            'Semester': '6M',
            'Year': 'Y'
        }

        freq = freq_map[freq_choice]

        # Select period
        if 'pub_date' in clean_df.columns:
            min_date = pd.to_datetime(clean_df['pub_date']).min()
            max_date = pd.to_datetime(clean_df['pub_date']).max()
        else:
            min_date = pd.to_datetime("2000-01-01")
            max_date = pd.to_datetime("2025-12-31")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", min_date.date())
        with col2:
            end_date = st.date_input("To", max_date.date())

        # More filtering

        st.subheader("Demographic Splits")

        split_age = st.checkbox("Split by Age Group", value=False)
        split_gender = st.checkbox("Split by Gender", value=False)

        # Prepare data for plotting

        age_order = ["<1", "0-17", "18-39", "40-59", "60-79", "80+", "unknown"]
        color_map = {"M": "#1f77b4", "F": "#e377c2"} 


        fig = px.line()
        for pat in selected_patterns:
            sub = timeseries_df[timeseries_df['pattern_label'] == pat].copy()
            if sub.empty:
                continue
            sub['pub_date'] = pd.to_datetime(sub['pub_date'])
            sub = sub[(sub['pub_date'] >= pd.to_datetime(start_date)) & (sub['pub_date'] <= pd.to_datetime(end_date))]

            sub = sub.groupby(pd.Grouper(key='pub_date', freq=freq)).sum().reset_index()

            fig.add_scatter(x=sub['pub_date'], y=sub['count'], mode='lines+markers', name=f"{pat} ({freq_choice})")

            # Age split
            if split_age and not split_gender:
                sub_demog = demog_df[demog_df['pattern_label'] == pat]
                fig_age = px.bar()
                for age_bin in sub_demog['age_bin'].unique():
                    val = sub_demog[sub_demog['age_bin'] == age_bin]['count'].sum()
                    fig_age.add_bar(x=[age_bin], y=[val], name=age_bin)
                fig_age.update_layout(title=f"Age Distribution for {pat} ({freq_choice})",
                                          xaxis_title="Age Group",
                                          yaxis_title="Number of Patients")
                fig_age.update_xaxes(categoryorder='array', categoryarray=age_order)

            # Gender split
            if split_gender and not split_age:
                sub_demog = demog_df[demog_df['pattern_label'] == pat]
                # Use graph_objects to control bar width and gaps (narrower bins)
                fig_gender = go.Figure()
                for gender in sub_demog['gender'].unique():
                    val = sub_demog[sub_demog['gender'] == gender]['count'].sum()
                    fig_gender.add_trace(go.Bar(
                        x=[gender],
                        y=[val],
                        name="Male" if gender == 'M' else "Female",
                        marker=dict(color=color_map.get(gender, '#7f7f7f')),
                        width=0.4  # shrink bin width (smaller -> narrower bars)
                    ))
                fig_gender.update_layout(
                    title=f"Gender Distribution for {pat} ({freq_choice})",
                    xaxis_title="Gender",
                    yaxis_title="Number of Patients",
                    bargap=0.25,       # gap between bars of adjacent location coordinates
                    bargroupgap=0.1   # gap between bars of the same location coordinate
                )

            # Both splits
            if split_age and split_gender:
                sub_demog = demog_df[demog_df['pattern_label'] == pat]
                male_data = sub_demog[sub_demog['gender'] == 'M'].groupby('age_bin')['count'].sum().reset_index()
                female_data = sub_demog[sub_demog['gender'] == 'F'].groupby('age_bin')['count'].sum().reset_index()

                # Creiamo il grafico verticale affiancato
                fig_age_gender = go.Figure()
                fig_age_gender.add_trace(go.Bar(
                    x=male_data['age_bin'],
                    y=male_data['count'],
                    name='Male',
                    marker_color=color_map['M']
                ))
                fig_age_gender.add_trace(go.Bar(
                    x=female_data['age_bin'],
                    y=female_data['count'],
                    name='Female',
                    marker_color=color_map['F']
                ))

                # Massimo valore per asse y
                max_val = max(female_data['count'].max() if not female_data.empty else 0,
                            male_data['count'].max() if not male_data.empty else 0)

                fig_age_gender.update_layout(
                    title=f"Age and Gender Distribution for {pat} ({freq_choice})",
                    xaxis=dict(title="Age Group", categoryorder='array', categoryarray=age_order),
                    yaxis=dict(title="Number of Patients", range=[0, max_val*1.1]),
                    barmode='group',  # Affianca le barre per età
                    bargap=0.2,
                    showlegend=True,
                    legend=dict(x=0.8, y=1.05, orientation='h')
                )

        fig.update_layout(
            title=f"Time Series of Selected Patterns ({freq_choice})",
            xaxis_title="Publication Date",
            yaxis_title="Number of Patients",
            legend_title="Patterns",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        if split_age and not split_gender:
            st.plotly_chart(fig_age, use_container_width=True)
        elif split_gender and not split_age:
            st.plotly_chart(fig_gender, use_container_width=True)
        elif split_age and split_gender:
            st.plotly_chart(fig_age_gender, use_container_width=True)

    st.markdown("---")
    st.subheader("🧾 Text Snippets")
    for pat in selected_patterns:
        st.markdown(f"**Pattern:** {pat}")
        snippets_sub = snippets_df[snippets_df['pattern_label'] == pat]
        if not snippets_sub.empty:
            snippets = snippets_sub.iloc[0]['snippets']
            for snip in snippets:
                highlighted = re.sub(r'(' + '|'.join(re.escape(tok) for tok in pat.split(' || ')) + r')', r"**\1**", snip, flags=re.IGNORECASE)
                st.markdown(f"- {highlighted}")
        else:
            st.write("No snippets available.")

