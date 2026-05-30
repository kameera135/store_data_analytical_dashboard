# =============================================================================
# ASSESSMENT 4: RETAIL SALES ANALYTICAL DASHBOARD
# ICT702 Business Analytics and Visualisation
# Tool: Python Dash + Plotly
# Run: python dashboard.py  →  open http://127.0.0.1:8050 in your browser
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    """Load and merge all three retail dataset files."""
    sales    = pd.read_csv("sales data-set.csv")
    stores   = pd.read_csv("stores data-set.csv")
    features = pd.read_csv("Features data set.csv")

    # Parse dates
    sales["Date"]    = pd.to_datetime(sales["Date"],    dayfirst=True)
    features["Date"] = pd.to_datetime(features["Date"], dayfirst=True)

    # Fill missing markdowns with 0 (no promotion active)
    md_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
    features[md_cols] = features[md_cols].fillna(0)

    # Fill CPI and Unemployment with store-level median
    for col in ["CPI", "Unemployment"]:
        features[col] = features.groupby("Store")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Remove negative sales (returns exceeding sales)
    sales = sales[sales["Weekly_Sales"] >= 0].copy()

    # Merge all datasets
    df = sales.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store","Date"], how="left", suffixes=("","_feat"))
    if "IsHoliday_feat" in df.columns:
        df.drop(columns=["IsHoliday_feat"], inplace=True)

    # Feature engineering
    df["Year"]          = df["Date"].dt.year
    df["Month"]         = df["Date"].dt.month
    df["Week"]          = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]       = df["Date"].dt.quarter
    df["TotalMarkDown"] = df[md_cols].sum(axis=1)
    df["MonthYear"]     = df["Date"].dt.to_period("M").astype(str)

    # Holiday labelling
    holiday_map = {
        "Super Bowl":   [6, 7],
        "Labor Day":    [36],
        "Thanksgiving": [47],
        "Christmas":    [51, 52],
    }
    def label_holiday(week):
        for name, weeks in holiday_map.items():
            if week in weeks:
                return name
        return "Non-Holiday"
    df["HolidayEvent"] = df["Week"].apply(label_holiday)

    # Encode store type
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"])

    return df

print("Loading data...")
df = load_data()
print(f"Data loaded: {df.shape[0]:,} rows")

# =============================================================================
# 2. TRAIN PREDICTIVE MODEL (once at startup)
# =============================================================================

print("Training predictive model...")
feature_cols = [
    "Store","Dept","Week","Month","Year","Quarter",
    "Type_encoded","Size","Temperature","Fuel_Price",
    "CPI","Unemployment","TotalMarkDown","IsHoliday"
]

df_model = df[feature_cols + ["Weekly_Sales"]].dropna().copy()
df_model["IsHoliday"] = df_model["IsHoliday"].astype(int)

train_mask = df_model["Year"] < 2012
df_train   = resample(df_model[train_mask],
                      n_samples=int(len(df_model[train_mask]) * 0.3),
                      random_state=42)

rf = RandomForestRegressor(
    n_estimators=50, max_depth=12,
    min_samples_split=10, random_state=42, n_jobs=-1
)
rf.fit(df_train[feature_cols], df_train["Weekly_Sales"])

# Feature importance for the insights tab
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

print("Model ready.")

# =============================================================================
# 3. COLOUR PALETTE & THEME
# =============================================================================

COLORS = {
    "primary":    "#2C3E7A",   # deep navy
    "accent":     "#E8572A",   # warm orange
    "success":    "#27AE60",   # green
    "warning":    "#F39C12",   # amber
    "light_bg":   "#F4F6FB",
    "card_bg":    "#FFFFFF",
    "text":       "#2C3E50",
    "muted":      "#7F8C8D",
    "grid":       "#ECF0F1",
    "christmas":  "#E53935",
    "thanks":     "#FB8C00",
    "superbowl":  "#1E88E5",
    "laborday":   "#43A047",
    "nonholiday": "#9E9E9E",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Segoe UI, Arial", color=COLORS["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["light_bg"],
        colorway=[COLORS["primary"], COLORS["accent"], COLORS["success"],
                  COLORS["warning"], "#8E44AD", "#16A085"],
        xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

# =============================================================================
# 4. HELPER: KPI CARD
# =============================================================================

def kpi_card(title, value, subtitle="", color=COLORS["primary"], icon="📊"):
    return dbc.Card([
        dbc.CardBody([
            html.Div(icon, style={"fontSize":"28px","marginBottom":"4px"}),
            html.H4(value, style={
                "color": color, "fontWeight":"700",
                "fontSize":"1.6rem", "margin":"0"
            }),
            html.P(title, style={
                "color": COLORS["text"], "fontWeight":"600",
                "fontSize":"0.85rem", "margin":"2px 0 0 0"
            }),
            html.Small(subtitle, style={"color": COLORS["muted"], "fontSize":"0.75rem"})
        ])
    ], style={
        "background": COLORS["card_bg"],
        "borderRadius": "12px",
        "boxShadow": "0 2px 12px rgba(44,62,122,0.08)",
        "border": f"2px solid {color}",
        "textAlign": "center",
        "padding": "4px"
    })

# =============================================================================
# 5. PRE-COMPUTE SUMMARY STATS
# =============================================================================

total_revenue    = df["Weekly_Sales"].sum()
avg_weekly       = df.groupby("Date")["Weekly_Sales"].sum().mean()
best_store       = df.groupby("Store")["Weekly_Sales"].sum().idxmax()
best_store_sales = df.groupby("Store")["Weekly_Sales"].sum().max()
holiday_uplift   = (
    df[df["IsHoliday"]==True]["Weekly_Sales"].mean() /
    df[df["IsHoliday"]==False]["Weekly_Sales"].mean() - 1
) * 100

store_options = [{"label": f"Store {s}", "value": s} for s in sorted(df["Store"].unique())]
store_options_all = [{"label": "All Stores", "value": 0}] + store_options

year_options = [{"label": str(y), "value": y} for y in sorted(df["Year"].unique())]
year_options_all = [{"label": "All Years", "value": 0}] + year_options

# =============================================================================
# 6. APP LAYOUT
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Retail Analytics Dashboard"
)

app.layout = dbc.Container([

    # ── HEADER ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("🛒 Retail Sales Analytics Dashboard",
                        style={"color": "#FFFFFF", "fontWeight":"700",
                               "margin":"0", "fontSize":"1.7rem"}),
                html.P("ICT702 | 45 Stores | 2010–2012 | Powered by Python Dash",
                       style={"color":"rgba(255,255,255,0.75)",
                              "margin":"2px 0 0 0", "fontSize":"0.85rem"})
            ], style={
                "background": f"linear-gradient(135deg, {COLORS['primary']}, #4A5FA8)",
                "padding": "18px 28px",
                "borderRadius": "14px",
                "marginBottom": "18px"
            })
        ])
    ]),

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(kpi_card("Total Revenue",
                         f"${total_revenue/1e9:.2f}B",
                         "All stores 2010–2012",
                         COLORS["primary"], "💰"), width=3),
        dbc.Col(kpi_card("Avg Weekly Sales",
                         f"${avg_weekly/1e6:.2f}M",
                         "Across all 45 stores",
                         COLORS["accent"], "📈"), width=3),
        dbc.Col(kpi_card("Top Store",
                         f"Store {best_store}",
                         f"${best_store_sales/1e6:.1f}M total",
                         COLORS["success"], "🏆"), width=3),
        dbc.Col(kpi_card("Holiday Uplift",
                         f"+{holiday_uplift:.1f}%",
                         "vs non-holiday weeks",
                         COLORS["warning"], "🎄"), width=3),
    ], className="mb-3"),

    # ── FILTERS ───────────────────────────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Store", style={"fontWeight":"600","fontSize":"0.85rem"}),
                    dcc.Dropdown(
                        id="filter-store",
                        options=store_options_all,
                        value=0,
                        clearable=False,
                        style={"fontSize":"0.85rem"}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Year", style={"fontWeight":"600","fontSize":"0.85rem"}),
                    dcc.Dropdown(
                        id="filter-year",
                        options=year_options_all,
                        value=0,
                        clearable=False,
                        style={"fontSize":"0.85rem"}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Store Type", style={"fontWeight":"600","fontSize":"0.85rem"}),
                    dcc.Dropdown(
                        id="filter-type",
                        options=[
                            {"label":"All Types","value":"All"},
                            {"label":"Type A (Large)","value":"A"},
                            {"label":"Type B (Medium)","value":"B"},
                            {"label":"Type C (Small)","value":"C"},
                        ],
                        value="All",
                        clearable=False,
                        style={"fontSize":"0.85rem"}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Holiday Weeks Only",
                               style={"fontWeight":"600","fontSize":"0.85rem"}),
                    dbc.Switch(id="filter-holiday", value=False,
                               style={"marginTop":"8px"})
                ], width=3),
            ])
        ])
    ], style={"borderRadius":"12px","marginBottom":"16px",
              "boxShadow":"0 2px 8px rgba(0,0,0,0.06)"}),

    # ── TABS ──────────────────────────────────────────────────────────────────
    dbc.Tabs([

        # TAB 1: SALES OVERVIEW
        dbc.Tab(label="📊 Sales Overview", tab_id="tab-overview", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-trend"), width=8),
                dbc.Col(dcc.Graph(id="chart-seasonal"), width=4),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-type"), width=4),
                dbc.Col(dcc.Graph(id="chart-dept-top"), width=8),
            ], className="mt-2"),
        ]),

        # TAB 2: STORE PERFORMANCE
        dbc.Tab(label="🏪 Store Performance", tab_id="tab-stores", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-store-bar"), width=7),
                dbc.Col(dcc.Graph(id="chart-store-scatter"), width=5),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-store-heatmap"), width=12),
            ], className="mt-2"),
        ]),

        # TAB 3: HOLIDAY ANALYSIS
        dbc.Tab(label="🎄 Holiday Analysis", tab_id="tab-holiday", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-holiday-compare"), width=6),
                dbc.Col(dcc.Graph(id="chart-holiday-events"), width=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-markdown-holiday"), width=12),
            ], className="mt-2"),
        ]),

        # TAB 4: PREDICTIVE INSIGHTS
        dbc.Tab(label="🤖 Predictive Insights", tab_id="tab-predict", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-feat-imp"), width=5),
                dbc.Col(dcc.Graph(id="chart-corr"), width=7),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-forecast"), width=12),
            ], className="mt-2"),
        ]),

        # TAB 5: DATA TABLE
        dbc.Tab(label="📋 Data Explorer", tab_id="tab-data", children=[
            dbc.Row([
                dbc.Col([
                    html.P("Showing aggregated weekly sales by store and date.",
                           style={"color":COLORS["muted"],"marginTop":"16px",
                                  "fontSize":"0.85rem"}),
                    dash_table.DataTable(
                        id="data-table",
                        page_size=20,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX":"auto"},
                        style_header={
                            "backgroundColor": COLORS["primary"],
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "0.82rem"
                        },
                        style_cell={
                            "fontSize": "0.82rem",
                            "padding": "8px",
                            "fontFamily": "Segoe UI, Arial"
                        },
                        style_data_conditional=[{
                            "if": {"row_index": "odd"},
                            "backgroundColor": COLORS["light_bg"]
                        }]
                    )
                ])
            ], className="mt-3")
        ]),

    ], id="tabs", active_tab="tab-overview"),

    # FOOTER
    html.Hr(style={"marginTop":"24px"}),
    html.P("ICT702 Business Analytics & Visualisation | Victorian Institute of Technology | 2026",
           style={"textAlign":"center","color":COLORS["muted"],"fontSize":"0.78rem"})

], fluid=True, style={"backgroundColor": COLORS["light_bg"],
                       "padding": "20px", "fontFamily": "Segoe UI, Arial"})


# =============================================================================
# 7. CALLBACKS
# =============================================================================

def filter_df(store, year, store_type, holiday_only):
    """Apply dashboard filters to the master dataframe."""
    dff = df.copy()
    if store != 0:
        dff = dff[dff["Store"] == store]
    if year != 0:
        dff = dff[dff["Year"] == year]
    if store_type != "All":
        dff = dff[dff["Type"] == store_type]
    if holiday_only:
        dff = dff[dff["IsHoliday"] == True]
    return dff


# ── TAB 1: SALES OVERVIEW ────────────────────────────────────────────────────

@app.callback(
    Output("chart-trend", "figure"),
    Output("chart-seasonal", "figure"),
    Output("chart-type", "figure"),
    Output("chart-dept-top", "figure"),
    Input("filter-store", "value"),
    Input("filter-year", "value"),
    Input("filter-type", "value"),
    Input("filter-holiday", "value"),
)
def update_overview(store, year, store_type, holiday_only):
    dff = filter_df(store, year, store_type, holiday_only)

    # 1. Monthly trend
    trend = dff.groupby(dff["Date"].dt.to_period("M"))["Weekly_Sales"].sum().reset_index()
    trend["Date"] = trend["Date"].astype(str)
    fig_trend = px.area(
        trend, x="Date", y="Weekly_Sales",
        title="Monthly Total Sales Trend",
        labels={"Weekly_Sales": "Total Sales ($)", "Date": "Month"},
        color_discrete_sequence=[COLORS["primary"]]
    )
    fig_trend.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig_trend.update_traces(line_color=COLORS["primary"],
                            fillcolor=f"rgba(44,62,122,0.15)")
    fig_trend.update_yaxes(tickprefix="$", tickformat=",.0f")

    # 2. Seasonal bar
    seasonal = dff.groupby("Month")["Weekly_Sales"].mean().reset_index()
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    seasonal["MonthName"] = seasonal["Month"].map(month_map)
    fig_seas = px.bar(
        seasonal, x="MonthName", y="Weekly_Sales",
        title="Avg Sales by Month",
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "MonthName": ""},
        color="Weekly_Sales",
        color_continuous_scale=["#B3C6F7", COLORS["primary"]]
    )
    fig_seas.update_layout(**PLOTLY_TEMPLATE["layout"],
                           coloraxis_showscale=False)
    fig_seas.update_yaxes(tickprefix="$", tickformat=",.0f")

    # 3. Store type
    type_sales = dff.groupby("Type")["Weekly_Sales"].mean().reset_index()
    type_colors = {"A": COLORS["primary"], "B": COLORS["accent"],
                   "C": COLORS["success"]}
    fig_type = px.bar(
        type_sales, x="Type", y="Weekly_Sales",
        title="Avg Sales by Store Type",
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Type": "Store Type"},
        color="Type",
        color_discrete_map=type_colors
    )
    fig_type.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
    fig_type.update_yaxes(tickprefix="$", tickformat=",.0f")

    # 4. Top 10 departments
    dept_sales = dff.groupby("Dept")["Weekly_Sales"].mean().nlargest(10).reset_index()
    dept_sales["Dept"] = "Dept " + dept_sales["Dept"].astype(str)
    fig_dept = px.bar(
        dept_sales.sort_values("Weekly_Sales"),
        x="Weekly_Sales", y="Dept", orientation="h",
        title="Top 10 Departments by Avg Weekly Sales",
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Dept": ""},
        color="Weekly_Sales",
        color_continuous_scale=["#F7C6B3", COLORS["accent"]]
    )
    fig_dept.update_layout(**PLOTLY_TEMPLATE["layout"],
                           coloraxis_showscale=False)
    fig_dept.update_xaxes(tickprefix="$", tickformat=",.0f")

    return fig_trend, fig_seas, fig_type, fig_dept


# ── TAB 2: STORE PERFORMANCE ─────────────────────────────────────────────────

@app.callback(
    Output("chart-store-bar", "figure"),
    Output("chart-store-scatter", "figure"),
    Output("chart-store-heatmap", "figure"),
    Input("filter-store", "value"),
    Input("filter-year", "value"),
    Input("filter-type", "value"),
    Input("filter-holiday", "value"),
)
def update_stores(store, year, store_type, holiday_only):
    dff = filter_df(store, year, store_type, holiday_only)

    # 1. Store revenue bar (top & bottom)
    store_rev = dff.groupby(["Store","Type"])["Weekly_Sales"].sum().reset_index()
    store_rev = store_rev.sort_values("Weekly_Sales", ascending=False)

    top_stores = store_rev.head(10).copy()
    bot_stores = store_rev.tail(10).copy()
    top_stores["Category"] = "Top 10"
    bot_stores["Category"] = "Bottom 10"
    top10 = pd.concat([top_stores, bot_stores]).drop_duplicates()
    top10["Label"] = "Store " + top10["Store"].astype(str)
    fig_bar = px.bar(
        top10.sort_values("Weekly_Sales"),
        x="Weekly_Sales", y="Label", orientation="h",
        color="Type",
        title="Store Revenue: Top 10 vs Bottom 10",
        labels={"Weekly_Sales": "Total Sales ($)", "Label": ""},
        color_discrete_map={"A":COLORS["primary"],"B":COLORS["accent"],"C":COLORS["success"]}
    )
    fig_bar.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig_bar.update_xaxes(tickprefix="$", tickformat=",.2s")

    # 2. Size vs Sales scatter
    store_summary = dff.groupby(["Store","Type","Size"])["Weekly_Sales"].mean().reset_index()
    fig_scatter = px.scatter(
        store_summary, x="Size", y="Weekly_Sales", color="Type",
        size="Weekly_Sales", hover_data=["Store"],
        title="Store Size vs Avg Weekly Sales",
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Size": "Store Size (sq ft)"},
        color_discrete_map={"A":COLORS["primary"],"B":COLORS["accent"],"C":COLORS["success"]}
    )
    fig_scatter.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig_scatter.update_yaxes(tickprefix="$", tickformat=",.0f")

    # 3. Heatmap: store vs month
    heat_data = dff.groupby(["Store","Month"])["Weekly_Sales"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="Store", columns="Month", values="Weekly_Sales").fillna(0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [month_labels[c-1] for c in heat_pivot.columns]
    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=cols,
        y=[f"S{s}" for s in heat_pivot.index],
        colorscale="Blues",
        colorbar=dict(title="Avg Sales ($)", tickprefix="$")
    ))
    fig_heat.update_layout(
        title="Store x Month Sales Heatmap",
        xaxis_title="Month",
        yaxis_title="Store",
        height=500,
        **PLOTLY_TEMPLATE["layout"]
    )

    return fig_bar, fig_scatter, fig_heat


# ── TAB 3: HOLIDAY ANALYSIS ──────────────────────────────────────────────────

@app.callback(
    Output("chart-holiday-compare", "figure"),
    Output("chart-holiday-events", "figure"),
    Output("chart-markdown-holiday", "figure"),
    Input("filter-store", "value"),
    Input("filter-year", "value"),
    Input("filter-type", "value"),
    Input("filter-holiday", "value"),
)
def update_holiday(store, year, store_type, holiday_only):
    dff = filter_df(store, year, store_type, holiday_only)

    # 1. Holiday vs non-holiday
    hol = dff.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
    hol["Label"] = hol["IsHoliday"].map({True:"Holiday Week",False:"Non-Holiday"})
    fig_comp = px.bar(
        hol, x="Label", y="Weekly_Sales",
        title="Holiday vs Non-Holiday Avg Sales",
        labels={"Weekly_Sales":"Avg Weekly Sales ($)","Label":""},
        color="Label",
        color_discrete_map={"Holiday Week":COLORS["accent"],"Non-Holiday":COLORS["primary"]}
    )
    fig_comp.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
    fig_comp.update_yaxes(tickprefix="$", tickformat=",.0f")

    # 2. Events
    event_colors = {
        "Christmas":    COLORS["christmas"],
        "Thanksgiving": COLORS["thanks"],
        "Super Bowl":   COLORS["superbowl"],
        "Labor Day":    COLORS["laborday"],
        "Non-Holiday":  COLORS["nonholiday"],
    }
    ev = dff.groupby("HolidayEvent")["Weekly_Sales"].mean().reset_index()
    ev = ev.sort_values("Weekly_Sales", ascending=False)
    fig_ev = px.bar(
        ev, x="HolidayEvent", y="Weekly_Sales",
        title="Avg Sales by Holiday Event",
        labels={"Weekly_Sales":"Avg Weekly Sales ($)","HolidayEvent":""},
        color="HolidayEvent",
        color_discrete_map=event_colors
    )
    fig_ev.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
    fig_ev.update_yaxes(tickprefix="$", tickformat=",.0f")

    # 3. MarkDown vs Sales by week
    md_week = dff.groupby("Week")[["Weekly_Sales","TotalMarkDown"]].mean().reset_index()
    fig_md = make_subplots(specs=[[{"secondary_y": True}]])
    fig_md.add_trace(
        go.Scatter(x=md_week["Week"], y=md_week["Weekly_Sales"],
                   name="Avg Weekly Sales", line=dict(color=COLORS["primary"], width=2)),
        secondary_y=False
    )
    fig_md.add_trace(
        go.Bar(x=md_week["Week"], y=md_week["TotalMarkDown"],
               name="Avg Total MarkDown", marker_color=COLORS["accent"],
               opacity=0.5),
        secondary_y=True
    )
    fig_md.update_layout(
        title="Weekly Sales vs MarkDown Promotions",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["light_bg"],
        font=dict(family="Segoe UI, Arial", color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig_md.update_yaxes(title_text="Avg Weekly Sales ($)", secondary_y=False,
                        tickprefix="$", tickformat=",.0f")
    fig_md.update_yaxes(title_text="Avg Total MarkDown ($)", secondary_y=True,
                        tickprefix="$", tickformat=",.0f")

    return fig_comp, fig_ev, fig_md


# ── TAB 4: PREDICTIVE INSIGHTS ───────────────────────────────────────────────

@app.callback(
    Output("chart-feat-imp", "figure"),
    Output("chart-corr", "figure"),
    Output("chart-forecast", "figure"),
    Input("filter-store", "value"),
    Input("filter-year", "value"),
    Input("filter-type", "value"),
    Input("filter-holiday", "value"),
)
def update_predict(store, year, store_type, holiday_only):
    dff = filter_df(store, year, store_type, holiday_only)

    # 1. Feature importance
    fi_df = feat_imp.reset_index()
    fi_df.columns = ["Feature", "Importance"]
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        title="Random Forest Feature Importance",
        labels={"Importance":"Importance Score","Feature":""},
        color="Importance",
        color_continuous_scale=["#B3C6F7", COLORS["primary"]]
    )
    fig_fi.update_layout(**PLOTLY_TEMPLATE["layout"],
                         coloraxis_showscale=False)

    # 2. Correlation heatmap
    corr_cols = ["Weekly_Sales","Temperature","Fuel_Price",
                 "CPI","Unemployment","TotalMarkDown","Size"]
    corr_data = dff[corr_cols].dropna().corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns.tolist(),
        y=corr_data.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr_data.values,
        texttemplate="%{text}",
        colorbar=dict(title="r")
    ))
    fig_corr.update_layout(
        title="Feature Correlation Heatmap",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["light_bg"],
        font=dict(family="Segoe UI, Arial", color=COLORS["text"]),
        margin=dict(l=80, r=20, t=50, b=80),
        height=380
    )

    # 3. Actual vs Predicted for the filtered store (or store 20 by default)
    target_store = store if store != 0 else 20
    store_data = df[df["Store"] == target_store].copy()
    store_data["IsHoliday"] = store_data["IsHoliday"].astype(int)
    store_model = store_data[feature_cols + ["Weekly_Sales","Date"]].dropna()
    if len(store_model) > 10:
        store_model = store_model.sort_values("Date")
        X_s = store_model[feature_cols]
        y_pred_s = rf.predict(X_s)
        actual_ts = store_model.groupby("Date")["Weekly_Sales"].sum().reset_index()
        pred_agg = store_model.copy()
        pred_agg["Predicted"] = y_pred_s
        pred_ts = pred_agg.groupby("Date")["Predicted"].sum().reset_index()
        merged = actual_ts.merge(pred_ts, on="Date")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=merged["Date"], y=merged["Weekly_Sales"],
            name="Actual", line=dict(color=COLORS["primary"], width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=merged["Date"], y=merged["Predicted"],
            name="Predicted (RF)", line=dict(color=COLORS["accent"],
                                              width=2, dash="dash")
        ))
        fig_fc.update_layout(
            title=f"Actual vs Predicted Weekly Sales — Store {target_store}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLORS["light_bg"],
            font=dict(family="Segoe UI, Arial", color=COLORS["text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis_title="Date",
            yaxis_title="Weekly Sales ($)"
        )
        fig_fc.update_yaxes(tickprefix="$", tickformat=",.0f")
    else:
        fig_fc = go.Figure()
        fig_fc.update_layout(title="Not enough data for selected filters")

    return fig_fi, fig_corr, fig_fc


# ── TAB 5: DATA TABLE ────────────────────────────────────────────────────────

@app.callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Input("filter-store", "value"),
    Input("filter-year", "value"),
    Input("filter-type", "value"),
    Input("filter-holiday", "value"),
)
def update_table(store, year, store_type, holiday_only):
    dff = filter_df(store, year, store_type, holiday_only)
    table = dff.groupby(["Store","Type","Date","IsHoliday"])["Weekly_Sales"]\
               .sum().reset_index()
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    table["Weekly_Sales"] = table["Weekly_Sales"].round(2)
    table["IsHoliday"] = table["IsHoliday"].map({True:"Yes", False:"No"})
    table = table.rename(columns={
        "Store":"Store","Type":"Type","Date":"Date",
        "IsHoliday":"Holiday","Weekly_Sales":"Total Weekly Sales ($)"
    })
    table = table.sort_values("Date", ascending=False).head(500)
    cols = [{"name": c, "id": c} for c in table.columns]
    return table.to_dict("records"), cols


# =============================================================================
# 8. RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  RETAIL ANALYTICS DASHBOARD RUNNING")
    print("  Open your browser at: http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(debug=False, port=8050)