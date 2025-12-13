import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os

st.set_page_config(page_title="PatientLens", layout="wide")

INPUT_CSV = 'data/PMC-Patients.csv'
PUBMED_CSV = 'data/pubmed_data.csv'
CLEAN_OUT = 'outputs/PMC_clean.parquet'
TRANS_OUT = 'outputs/transactions.parquet'
PATTERN_OUT = 'outputs/patterns.parquet'
TS_OUT = 'outputs/timeseries.parquet'
SNIPPET_OUT = 'outputs/snippets.parquet'
DEMOG_OUT = 'outputs/demographics.parquet'

@st.cache_data
def load_data():
    clean_df = pd.read_parquet(CLEAN_OUT)
    pattern_df = pd.read_parquet(PATTERN_OUT)
    timeseries_df = pd.read_parquet(TS_OUT)
    snippets_df = pd.read_parquet(SNIPPET_OUT)
    demog_df = pd.read_parquet(DEMOG_OUT)
    return clean_df, pattern_df, timeseries_df, snippets_df, demog_df

clean_df, pattern_df, timeseries_df, snippets_df, demog_df = load_data()

st.title("PatientLens — Visual Text Mining of clinical cases")

st.sidebar.header("Global Controls")
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    clean_df, pattern_df, timeseries_df, snippets_df, demog_df = load_data()

st.sidebar.info("You can search and select patterns, view time series, and explore patient demographics.")

left, right = st.columns([3, 7])

with left:
    st.header("Patterns List")

    if 'support_count' not in pattern_df.columns:
        pattern_df['support_count'] = (pattern_df['support'] * len(clean_df)).astype(int)
    
    min_len = st.slider("Minimum itemset length", 1, 5, 1)
    min_freq = st.number_input("Minimum support count", min_value=1, value=10, step=1)
    order_by = st.selectbox("Order by", options=['support_count', 'len'], index=0)
    ascending = st.checkbox("Ascending order", value=False)

    filt = pattern_df[
        (pattern_df['len'] >= min_len) &
        (pattern_df['support_count'] >= min_freq)
    ]

    filt = filt.sort_values(by=order_by, ascending=ascending).reset_index(drop=True)


    st.write(f"Total patients: {len(clean_df)}")


    options = [f"{r['pattern_label']} - {r['support_count']}" for _, r in filt.iterrows()]
    selected = st.multiselect("Select patterns to visualize", options, default=options[:1], max_selections=5)
    selected_patterns = [s.split(" - ")[0] for s in selected]

    st.dataframe(
        filt[['pattern_label', 'support_count', 'support', 'len']]
        .rename(columns={'pattern_label': 'Pattern', 'support_count': '#Patients', 'support': 'Support', 'len': '#Terms'}),
        use_container_width=True,
        height=300
    )
with right:
    st.header("Time Series Visualization")

    if not selected_patterns:
        st.info("Select patterns from the left panel to visualize their time series.")
    else:
        st.subheader("Time Aggregation Options")
        freq_choice = st.radio(
            "Time Frequency:",
            ['Month', 'Trimester', 'Semester', 'Year'],
            horizontal=True
        )

        freq_map = {
            'Month': 'M',
            'Trimester': 'Q',
            'Semester': 'BQ',
            'Year': 'Y'
        }

        freq_display_map = {
            'Month': 'MS',  # Month Start
            'Trimester': 'QS',  # Quarter Start
            'Semester': 'BQS',  # 2 Quarter Start
            'Year': 'YS'  # Year Start
        }

        freq = freq_map[freq_choice]
        display_freq = freq_display_map[freq_choice]

        if not timeseries_df.empty:
            timeseries_df['pub_date'] = pd.to_datetime(timeseries_df['pub_date'])
            min_date = timeseries_df['pub_date'].min()
            max_date = timeseries_df['pub_date'].max()
        elif 'pub_date' in clean_df.columns:
            clean_df['pub_date'] = pd.to_datetime(clean_df['pub_date'])
            min_date = clean_df['pub_date'].min()
            max_date = clean_df['pub_date'].max()
        else:
            min_date = pd.to_datetime("2020-01-01")
            max_date = pd.to_datetime("2021-12-31")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", min_date.date())
        with col2:
            end_date = st.date_input("To", max_date.date())


        st.subheader("Demographic Splits")

        split_age = st.checkbox("Split by Age Group", value=False)
        split_gender = st.checkbox("Split by Gender", value=False)

        st.subheader("Normalization")
        
        normalization_unit = st.selectbox("Normalization unit", options=['Percentage', 'Raw count'], index=0)

        age_order = ["<1", "0-17", "18-39", "40-59", "60-79", "80+", "unknown"]
        color_map = {"M": "#1f77b4", "F": "#e377c2"}

        date_filtered_clean = clean_df[
            (clean_df['pub_date'] >= pd.to_datetime(start_date)) &
            (clean_df['pub_date'] <= pd.to_datetime(end_date))
        ].copy()
        
        if not date_filtered_clean.empty:
            date_filtered_clean = date_filtered_clean.set_index('pub_date')
            total_per_period = date_filtered_clean.resample(display_freq).size()
            total_per_period = total_per_period.reset_index(name='total_articles')
            total_per_period['period_label'] = total_per_period['pub_date'].dt.strftime('%Y-%m')
            
            if freq_choice == 'Trimester':
                total_per_period['period_label'] = total_per_period['pub_date'].dt.to_period('Q').astype(str)
            elif freq_choice == 'Semester':
                total_per_period['period_label'] = total_per_period['pub_date'].dt.to_period('2Q').astype(str)
            elif freq_choice == 'Year':
                total_per_period['period_label'] = total_per_period['pub_date'].dt.year.astype(str)
        else:
            total_per_period = pd.DataFrame(columns=['pub_date', 'total_articles', 'period_label'])

        fig = px.line()
        all_merged_data = []
        for pat in selected_patterns:
            sub = timeseries_df[timeseries_df['pattern_label'] == pat].copy()
            if sub.empty:
                continue
            sub['pub_date'] = pd.to_datetime(sub['pub_date'])
            sub = sub[(sub['pub_date'] >= pd.to_datetime(start_date)) & (sub['pub_date'] <= pd.to_datetime(end_date))]

            sub = sub.set_index('pub_date')
            sub_resampled = sub['count'].resample(display_freq).sum().reset_index()
            sub_resampled['period_label'] = sub_resampled['pub_date'].dt.strftime('%Y-%m')

            if freq_choice == 'Trimester':
                sub_resampled['period_label'] = sub_resampled['pub_date'].dt.to_period('Q').astype(str)
            elif freq_choice == 'Semester':
                sub_resampled['period_label'] = sub_resampled['pub_date'].dt.to_period('2Q').astype(str)
            elif freq_choice == 'Year':
                sub_resampled['period_label'] = sub_resampled['pub_date'].dt.year.astype(str)

            sub_resampled['display_date'] = sub_resampled['pub_date']
            if not total_per_period.empty:
                merged = pd.merge(total_per_period[['period_label', 'total_articles']], sub_resampled[['period_label', 'count', 'display_date']], on='period_label', how='outer')
                merged['count'] = merged['count'].fillna(0).astype(int)
                merged['total_articles'] = merged['total_articles'].fillna(0).astype(int)
            else:
                merged = sub_resampled.copy()
                merged['total_articles'] = 0

            st.caption(f"Pattern '{pat}': {len(merged)} periods, total count: {merged['count'].sum()}")
            

            if normalization_unit == 'Percentage':
                merged['rate'] = merged.apply(
                    lambda r: (r['count'] / r['total_articles'] * 100) if r['total_articles'] > 0 else 0.0,
                    axis=1
                )
                y_col = 'rate'
                y_label = 'Rate of Patients (%)'
            else:
                y_col = 'count'
                y_label = 'Number of Patients'

            fig.add_scatter(x=merged['display_date'], y=merged[y_col], mode='lines+markers', name=f"{pat} ({freq_choice})")

        fig.update_layout(
            title=f"Time Series of Selected Patterns ({freq_choice})",
            xaxis_title="Publication Date",
            yaxis_title=y_label,
            legend_title="Patterns",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        if split_age or split_gender:
            st.subheader("Demographic Distributions")
            
            for pat in selected_patterns:
                sub_demog = demog_df[demog_df['pattern_label'] == pat]
                
                if sub_demog.empty:
                    continue
                                
                if split_age and not split_gender:
                    fig_age = px.bar()
                    for age_bin in sub_demog['age_bin'].unique():
                        val = sub_demog[sub_demog['age_bin'] == age_bin]['count'].sum()
                        fig_age.add_bar(x=[age_bin], y=[val], name=age_bin)
                    fig_age.update_layout(
                        title=f"Age Distribution for {pat}",
                        xaxis_title="Age Group",
                        yaxis_title="Number of Patients"
                    )
                    fig_age.update_xaxes(categoryorder='array', categoryarray=age_order)
                    st.plotly_chart(fig_age, use_container_width=True)

                elif split_gender and not split_age:
                    fig_gender = go.Figure()
                    for gender in sub_demog['gender'].unique():
                        val = sub_demog[sub_demog['gender'] == gender]['count'].sum()
                        fig_gender.add_trace(go.Bar(
                            x=[gender],
                            y=[val],
                            name="Male" if gender == 'M' else "Female",
                            marker=dict(color=color_map.get(gender, '#7f7f7f')),
                            width=0.4
                        ))
                    fig_gender.update_layout(
                        title=f"Gender Distribution for {pat}",
                        xaxis_title="Gender",
                        yaxis_title="Number of Patients",
                        bargap=0.25,
                        bargroupgap=0.1
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)

                elif split_age and split_gender:
                    male_data = sub_demog[sub_demog['gender'] == 'M'].groupby('age_bin')['count'].sum().reset_index()
                    female_data = sub_demog[sub_demog['gender'] == 'F'].groupby('age_bin')['count'].sum().reset_index()

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

                    max_val = max(
                        female_data['count'].max() if not female_data.empty else 0,
                        male_data['count'].max() if not male_data.empty else 0
                    )

                    fig_age_gender.update_layout(
                        title=f"Age and Gender Distribution for {pat}",
                        xaxis=dict(title="Age Group", categoryorder='array', categoryarray=age_order),
                        yaxis=dict(title="Number of Patients", range=[0, max_val*1.1]),
                        barmode='group',
                        bargap=0.2,
                        showlegend=True,
                        legend=dict(x=0.8, y=1.05, orientation='h')
                    )
                    st.plotly_chart(fig_age_gender, use_container_width=True)

    st.markdown("---")
    st.subheader("Text Snippets")
    for pat in selected_patterns:
        st.markdown(f"**Pattern:** {pat}")
        snippets_sub = snippets_df[snippets_df['pattern_label'] == pat]
        if not snippets_sub.empty:
            snippets = snippets_sub.iloc[0]['snippets']
            titles = snippets_sub.iloc[0]['titles']
            for snip, tit  in zip(snippets, titles):
                highlighted = re.sub(r'(' + '|'.join(re.escape(tok) for tok in pat.split(' || ')) + r')', r"**\1**", snip, flags=re.IGNORECASE)
                st.markdown(f"- ### {tit}")
                st.markdown(f"{highlighted}")
        else:
            st.write("No snippets available.")

