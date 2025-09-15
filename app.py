import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import os

st.set_page_config(page_title="Marketing Intelligence Dashboard", layout="wide")

# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def read_csv_flexible(source):
    
    if source is None:
        return None
    try:
        if hasattr(source, 'read'):
            # file-like (Streamlit uploader)
            source.seek(0)
            return pd.read_csv(source, parse_dates=['date'], dayfirst=True, engine='python')
        else:
            return pd.read_csv(source, parse_dates=['date'], dayfirst=True, engine='python')
    except Exception:
        # fallback without parse_dates so we can coerce later
        if hasattr(source, 'read'):
            source.seek(0)
            df = pd.read_csv(source, engine='python')
        else:
            df = pd.read_csv(source, engine='python')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df


def normalize_business(df):
    if df is None:
        return None
    df = df.copy()
    # common rename map based on sample
    rename_map = {
        '# of orders': 'orders',
        '# of new orders': 'new_orders',
        'new customers': 'new_customers',
        'total revenue': 'total_revenue',
        'gross profit': 'gross_profit',
        'COGS': 'cogs'
    }
    # strip columns and lowercase
    df.columns = [c.strip() for c in df.columns]
    # apply explicit renames
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # normalize to snake_case simple
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    # coerce numeric
    for col in ['orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def normalize_marketing(df, channel_name):
    if df is None:
        return None
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # unify impression vs impressions
    if 'impression' in df.columns and 'impressions' not in df.columns:
        df.rename(columns={'impression': 'impressions'}, inplace=True)
    # unify attributed revenue naming
    if 'attributed_revenue' not in df.columns and 'attribution_revenue' in df.columns:
        df.rename(columns={'attribution_revenue': 'attributed_revenue'}, inplace=True)
    # coerce types
    for col in ['impressions', 'clicks', 'spend', 'attributed_revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['channel'] = channel_name
    # ensure tactic/state/campaign are strings
    for c in ['tactic', 'state', 'campaign']:
        if c in df.columns:
            df[c] = df[c].astype(str)
        else:
            df[c] = ''
    return df


@st.cache_data(show_spinner=False)
def prepare_all(business_src, facebook_src, google_src, tiktok_src):
    business = read_csv_flexible(business_src)
    fb = read_csv_flexible(facebook_src)
    gg = read_csv_flexible(google_src)
    tt = read_csv_flexible(tiktok_src)

    business = normalize_business(business)
    fb = normalize_marketing(fb, 'Facebook')
    gg = normalize_marketing(gg, 'Google')
    tt = normalize_marketing(tt, 'TikTok')

    # Combine marketing
    marketing_dfs = [df for df in [fb, gg, tt] if df is not None and not df.empty]
    if marketing_dfs:
        marketing = pd.concat(marketing_dfs, ignore_index=True)
    else:
        marketing = pd.DataFrame()

    # Derived columns for marketing
    if marketing is not None and not marketing.empty:
        # CTR, CPC
        marketing['ctr'] = (marketing['clicks'] / marketing['impressions']).replace([np.inf, -np.inf], np.nan)
        marketing['cpc'] = (marketing['spend'] / marketing['clicks']).replace([np.inf, -np.inf], np.nan)
    
    # aggregate marketing per date
    if marketing is not None and 'date' in marketing.columns:
        marketing_daily = marketing.groupby(['date', 'channel']).agg(
            impressions_total=('impressions', 'sum'),
            clicks_total=('clicks', 'sum'),
            spend_total=('spend', 'sum'),
            attributed_revenue_total=('attributed_revenue', 'sum')
        ).reset_index()
    else:
        marketing_daily = pd.DataFrame()

    # daily totals across channels
    if not marketing_daily.empty:
        marketing_totals_by_date = marketing_daily.groupby('date').agg(
            impressions=('impressions_total', 'sum'),
            clicks=('clicks_total', 'sum'),
            spend=('spend_total', 'sum'),
            attributed_revenue=('attributed_revenue_total', 'sum')
        ).reset_index()
    else:
        marketing_totals_by_date = pd.DataFrame()

    # Merge with business
    if business is not None and not business.empty and not marketing_totals_by_date.empty:
        merged = pd.merge(business, marketing_totals_by_date, on='date', how='left')
    elif business is not None and not business.empty:
        merged = business.copy()
        for c in ['impressions', 'clicks', 'spend', 'attributed_revenue']:
            merged[c] = 0
    else:
        merged = pd.DataFrame()

    # Fill na for merged
    if not merged.empty:
        for c in ['impressions', 'clicks', 'spend', 'attributed_revenue']:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0)
        # AOV (average order value)
        merged['aov'] = merged.apply(lambda r: (r['total_revenue'] / r['orders']) if (r.get('orders', 0) and r['orders'] > 0) else np.nan, axis=1)
    
    # Estimate channel-level attributed orders and CPA using AOV per date
    if not marketing_daily.empty and 'aov' in merged.columns:
        marketing_daily = marketing_daily.merge(merged[['date', 'aov']], on='date', how='left')
        marketing_daily['estimated_attributed_orders'] = marketing_daily.apply(
            lambda r: (r['attributed_revenue_total'] / r['aov']) if (pd.notnull(r['aov']) and r['aov'] > 0) else np.nan, axis=1)
        marketing_daily['estimated_cpa'] = (marketing_daily['spend_total'] / marketing_daily['estimated_attributed_orders']).replace([np.inf, -np.inf], np.nan)
        marketing_daily['roas'] = (marketing_daily['attributed_revenue_total'] / marketing_daily['spend_total']).replace([np.inf, -np.inf], np.nan)
    
    return {
        'business': business,
        'marketing_raw': marketing,
        'marketing_daily': marketing_daily,
        'merged_daily': merged
    }


# ---------------------- UI ----------------------
st.title("📊 Marketing Intelligence — Interactive Dashboard")
st.markdown("Builds connections between marketing inputs and business outcomes. Upload your CSVs or keep defaults if present in the working dir.")

with st.sidebar.expander('Upload data (or leave empty to use files from current working directory)'):
    business_file = st.file_uploader('business.csv', type=['csv'], key='business')
    facebook_file = st.file_uploader('facebook.csv', type=['csv'], key='facebook')
    google_file = st.file_uploader('google.csv', type=['csv'], key='google')
    tiktok_file = st.file_uploader('tiktok.csv', type=['csv'], key='tiktok')

    st.markdown('---')
    st.write('Filters & settings')
    smoothing = st.slider('Moving average window (days, 0 = none)', 0, 14, 3)
    min_roas = st.number_input('Show campaigns with ROAS >=', min_value=0.0, value=0.0, step=0.1)

data = prepare_all(
    business_file if business_file else "business.csv" if os.path.exists("business.csv") else None,
    facebook_file if facebook_file else "facebook.csv" if os.path.exists("facebook.csv") else None,
    google_file   if google_file   else "google.csv"   if os.path.exists("google.csv")   else None,
    tiktok_file   if tiktok_file   else "tiktok.csv"   if os.path.exists("tiktok.csv")   else None,
)

business = data['business']
marketing_raw = data['marketing_raw']
marketing_daily = data['marketing_daily']
merged_daily = data['merged_daily']

if business is None or business.empty:
    st.warning('Business data is missing or empty. Upload business.csv or place it in the working directory.')
    st.stop()

# default date range
min_date = business['date'].min()
max_date = business['date'].max()

col1, col2 = st.columns([3, 1])
with col2:
    st.write('Date range')
    start_date, end_date = st.date_input('Select date range', [min_date, max_date])
    if isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date).date()

# filters for channels
channels_available = []
if marketing_raw is not None and not marketing_raw.empty:
    channels_available = sorted(marketing_raw['channel'].unique())
selected_channels = st.multiselect('Channels', options=channels_available, default=channels_available)

# apply filters
mask = (business['date'].dt.date >= start_date) & (business['date'].dt.date <= end_date)
business_sel = business.loc[mask].copy()
merged_sel = merged_daily.loc[mask].copy()

if marketing_daily is not None and not marketing_daily.empty:
    md_mask = (marketing_daily['date'].dt.date >= start_date) & (marketing_daily['date'].dt.date <= end_date) & (marketing_daily['channel'].isin(selected_channels))
    marketing_daily_sel = marketing_daily.loc[md_mask].copy()
else:
    marketing_daily_sel = pd.DataFrame()

if marketing_raw is not None and not marketing_raw.empty:
    mraw_mask = (marketing_raw['date'].dt.date >= start_date) & (marketing_raw['date'].dt.date <= end_date) & (marketing_raw['channel'].isin(selected_channels))
    marketing_raw_sel = marketing_raw.loc[mraw_mask].copy()
else:
    marketing_raw_sel = pd.DataFrame()

# KPIs
with st.container():
    k1, k2, k3, k4, k5 = st.columns(5)
    total_revenue = business_sel['total_revenue'].sum()
    total_orders = business_sel['orders'].sum()
    total_spend = marketing_raw_sel['spend'].sum()
    total_attr_revenue = marketing_raw_sel['attributed_revenue'].sum()
    gross_profit = business_sel['gross_profit'].sum()

    k1.metric('Total revenue (selected)', f"₹{total_revenue:,.0f}", delta=None)
    k2.metric('Total orders (selected)', f"{int(total_orders):,}")
    k3.metric('Total marketing spend', f"₹{total_spend:,.0f}")
    roas_display = (total_attr_revenue / total_spend) if total_spend>0 else np.nan
    k4.metric('Aggregate ROAS (attributed revenue ÷ spend)', f"{(roas_display if pd.notnull(roas_display) else 0):.2f}")
    k5.metric('Gross profit', f"₹{gross_profit:,.0f}")

# Time series: Revenue vs Spend vs Attributed Revenue
st.markdown('### Time series: Revenue vs Spend vs Attributed Revenue')
if not merged_sel.empty:
    ts = merged_sel[['date', 'total_revenue', 'spend', 'attributed_revenue']].copy()
    ts = ts.sort_values('date')
    if smoothing > 0:
        ts['total_revenue_ma'] = ts['total_revenue'].rolling(smoothing, min_periods=1).mean()
        ts['spend_ma'] = ts['spend'].rolling(smoothing, min_periods=1).mean()
        ts['attributed_revenue_ma'] = ts['attributed_revenue'].rolling(smoothing, min_periods=1).mean()
    else:
        ts['total_revenue_ma'] = ts['total_revenue']
        ts['spend_ma'] = ts['spend']
        ts['attributed_revenue_ma'] = ts['attributed_revenue']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ts['date'], y=ts['spend_ma'], name='Marketing Spend (MA)'), secondary_y=False)
    fig.add_trace(go.Line(x=ts['date'], y=ts['total_revenue_ma'], name='Total Revenue (MA)'), secondary_y=True)
    fig.add_trace(go.Line(x=ts['date'], y=ts['attributed_revenue_ma'], name='Attributed Revenue (MA)'), secondary_y=True)
    fig.update_layout(height=420, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_yaxes(title_text='Spend (₹)', secondary_y=False)
    fig.update_yaxes(title_text='Revenue (₹)', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No merged daily data to show here. Check uploads and date filters.')

# Channel performance
st.markdown('### Channel performance (aggregated over selected period)')
if not marketing_daily_sel.empty:
    channel_agg = marketing_daily_sel.groupby('channel').agg(
        spend=('spend_total', 'sum'),
        attributed_revenue=('attributed_revenue_total', 'sum'),
        impressions=('impressions_total', 'sum'),
        clicks=('clicks_total', 'sum')
    ).reset_index()
    channel_agg['roas'] = (channel_agg['attributed_revenue'] / channel_agg['spend']).replace([np.inf, -np.inf], np.nan)
    channel_agg['ctr'] = (channel_agg['clicks'] / channel_agg['impressions']).replace([np.inf, -np.inf], np.nan)
    # bar chart for spend vs attributed revenue
    fig2 = px.bar(channel_agg.melt(id_vars='channel', value_vars=['spend', 'attributed_revenue']), x='channel', y='value', color='variable', barmode='group', labels={'value':'₹ Amount'})
    st.plotly_chart(fig2, use_container_width=True, height=380)
    st.dataframe(channel_agg.sort_values('spend', ascending=False).reset_index(drop=True))
else:
    st.info('No marketing channel data for selected filters.')

# Campaign-level table and insights
st.markdown('### Campaign-level breakdown (sorted by spend)')
if not marketing_raw_sel.empty:
    camp = marketing_raw_sel.groupby(['campaign', 'channel', 'tactic', 'state']).agg(
        impressions=('impressions', 'sum'),
        clicks=('clicks', 'sum'),
        spend=('spend', 'sum'),
        attributed_revenue=('attributed_revenue', 'sum')
    ).reset_index()
    # Estimate ROAS and CPC
    camp['roas'] = (camp['attributed_revenue'] / camp['spend']).replace([np.inf, -np.inf], np.nan)
    camp['cpc'] = (camp['spend'] / camp['clicks']).replace([np.inf, -np.inf], np.nan)
    camp_display = camp.sort_values('spend', ascending=False).reset_index(drop=True)
    st.dataframe(camp_display)
    # filter by ROAS
    if min_roas > 0:
        st.markdown(f"Showing campaigns with ROAS >= {min_roas}")
        st.dataframe(camp_display[camp_display['roas'] >= min_roas])

    # Top campaign chart by ROAS
    top_by_roas = camp_display.dropna(subset=['roas']).sort_values('roas', ascending=False).head(10)
    if not top_by_roas.empty:
        fig3 = px.bar(top_by_roas, x='campaign', y='roas', hover_data=['spend', 'attributed_revenue'], title='Top campaigns by ROAS (top 10)')
        st.plotly_chart(fig3, use_container_width=True)

    # Download filtered campaign data
    csv = camp_display.to_csv(index=False).encode('utf-8')
    st.download_button('Download campaign table (CSV)', csv, file_name='campaigns_filtered.csv', mime='text/csv')
else:
    st.info('No campaign-level marketing rows to show. Please check data uploads and filters.')

# Attribution reconciliation
st.markdown('### Attribution check: Marketing-attributed revenue vs reported business revenue')
if not merged_sel.empty:
    total_attr = merged_sel['attributed_revenue'].sum()
    total_business_rev = merged_sel['total_revenue'].sum()
    share_attr = (total_attr / total_business_rev) if total_business_rev > 0 else np.nan
    st.metric('Total marketing-attributed revenue', f"₹{total_attr:,.0f}")
    st.metric('Total reported business revenue', f"₹{total_business_rev:,.0f}")
    st.metric('Share attributed', f"{(share_attr*100 if pd.notnull(share_attr) else 0):.2f}%")
    if share_attr < 0.5:
        st.info('Less than 50% of reported revenue is attributed to marketing channels — consider investigating alternative attribution windows or offline sources.')
else:
    st.info('No merged data for attribution check.')

# Data quality and diagnostics
st.markdown('### Data quality & diagnostics')
if marketing_raw is not None:
    miss_marketing = marketing_raw.isna().sum()
    st.write('Missing values in marketing raw (full dataset):')
    st.write(miss_marketing[miss_marketing>0])
if business is not None:
    miss_business = business.isna().sum()
    st.write('Missing values in business data:')
    st.write(miss_business[miss_business>0])

# Actionable recommendations (simple heuristics)
st.markdown('### Quick automated insights (heuristic)')
insights = []
if not marketing_raw_sel.empty:
    channel_agg_tot = marketing_raw_sel.groupby('channel').agg(spend=('spend','sum'), attributed_revenue=('attributed_revenue','sum')).reset_index()
    channel_agg_tot['roas'] = (channel_agg_tot['attributed_revenue'] / channel_agg_tot['spend']).replace([np.inf, -np.inf], np.nan)
    low_roas = channel_agg_tot[channel_agg_tot['roas'] < 1]
    if not low_roas.empty:
        for _, r in low_roas.iterrows():
            insights.append(f"{r['channel']}: ROAS={r['roas']:.2f} — consider pausing/optimizing low-performing tactics.")
    high_spend_low_return = channel_agg_tot[(channel_agg_tot['spend'] > channel_agg_tot['spend'].median()) & (channel_agg_tot['roas'] < 1.2)]
    if not high_spend_low_return.empty:
        insights.append('One or more channels have above-median spend but low ROAS — investigate targeting and creative.')

if not insights:
    st.write('No automated insights from heuristics. Review charts and campaign table for deeper analysis.')
else:
    for i in insights:
        st.write('- ' + i)

# Footer: export processed data
st.markdown('---')
st.write('Export processed datasets for reporting or further analysis')
if not marketing_daily.empty:
    md_csv = marketing_daily.to_csv(index=False).encode('utf-8')
    st.download_button('Download marketing_daily.csv', md_csv, 'marketing_daily.csv', mime='text/csv')
if not merged_daily.empty:
    m_csv = merged_daily.to_csv(index=False).encode('utf-8')
    st.download_button('Download merged_daily.csv', m_csv, 'merged_daily.csv', mime='text/csv')

st.markdown('## Implementation notes')
st.markdown("""
- This app estimates channel-level CPA by dividing channel spend by estimated attributed orders (estimated using date-level AOV).
- Attribution reconciliation compares marketing-attributed revenue (sum across channels) to business-reported revenue.
- For production: move data into a database or pre-aggregated Parquet/BigQuery table and use caching for speed.
- To improve attribution: add conversion-level events (orders with UTM/campaign tags) or use a multi-touch attribution approach.
""")


