"""
Plastics Hunter — Ocean Plastic Cleanup Route Optimizer
Interactive Streamlit dashboard.
"""
import datetime
import io

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend import load_bundle, query_route, find_best_departure, fuel_deg_to_km, PORTS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plastics Hunter — Ocean Cleanup Optimizer",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    "<h2 style='margin-bottom:0'>🌊 Plastics Hunter — Ocean Cleanup Route Optimizer</h2>"
    "<p style='color:gray;margin-top:0'>Southeast Asia — Particle Drift Simulation & Vessel Route Optimization</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load bundle (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading optimization model (first run downloads ~190 MB)...")
def _load():
    return load_bundle()

try:
    bundle = _load()
    meta = bundle["meta"]
except Exception as e:
    st.error(f"Failed to load model bundle:\n\n```\n{e}\n```")
    st.info(
        "If this is the first deployment, make sure `BUNDLE_GDRIVE_ID` is set "
        "in `backend.py` (or as a Streamlit secret) and the Google Drive file "
        "is shared as 'Anyone with the link'."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Mission Parameters")

# Port selection
port_name = st.sidebar.selectbox(
    "Departure Port",
    list(PORTS.keys()) + ["Custom"],
)
if port_name == "Custom":
    col_lon, col_lat = st.sidebar.columns(2)
    port_lon = col_lon.number_input("Longitude", value=meta["PORT_LON"], format="%.1f")
    port_lat = col_lat.number_input("Latitude", value=meta["PORT_LAT"], format="%.1f")
    port = (port_lon, port_lat)
else:
    port = PORTS[port_name]

# Mission params — use calendar dates mapped to day-of-year
max_day = meta["n_prod_snapshots"] - 1
current_year = datetime.date.today().year
jan1 = datetime.date(current_year, 1, 1)
date_min = jan1
date_max = jan1 + datetime.timedelta(days=max_day)
today = datetime.date.today()
default_date = min(today, date_max)

departure_date = st.sidebar.date_input(
    "Departure Date", value=default_date, min_value=date_min, max_value=date_max
)
departure_day = (departure_date - jan1).days  # 0-based index into snapshots

trip_days = st.sidebar.slider("Trip Duration (days)", 3, 30, value=14)
n_vessels = st.sidebar.selectbox("Number of Vessels", [1, 2, 3, 4], index=1)

optimize_btn = st.sidebar.button("Optimize Routes", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Find Best Departure")
three_months_later = min(default_date + datetime.timedelta(days=90), date_max)
day_col1, day_col2 = st.sidebar.columns(2)
date_start = day_col1.date_input("From", value=default_date, min_value=date_min, max_value=date_max)
date_end = day_col2.date_input("To", value=three_months_later, min_value=date_min, max_value=date_max)
day_start = (date_start - jan1).days
day_end = (date_end - jan1).days
find_best_btn = st.sidebar.button("Search Best Day", use_container_width=True)

# ---------------------------------------------------------------------------
# Sidebar — QR code for mobile access (for live demos)
# ---------------------------------------------------------------------------
st.sidebar.divider()
with st.sidebar.expander("📱 Share via QR code"):
    default_url = "https://plastic-hunter-dashboard.streamlit.app"
    share_url = st.text_input("URL to encode", value=default_url)
    if st.button("Generate QR code", use_container_width=True):
        try:
            import qrcode
            img = qrcode.make(share_url)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="Scan to open on your phone", use_container_width=True)
        except ImportError:
            st.error("`qrcode` package not installed. Add it to requirements.txt.")

# ---------------------------------------------------------------------------
# Helper: build Plotly map figure
# ---------------------------------------------------------------------------
VESSEL_COLORS = ["#0077B6", "#2A9D8F", "#E76F51", "#9B5DE5"]

def build_map(routes, plastic, fuel, fc, port):
    """Build a Plotly Mapbox figure with density heatmap + routes."""
    fig = go.Figure()

    # --- density heatmap from forecast (last timestep) ---
    if fc is not None and fc.size > 0:
        bbox = meta["BBOX"]
        density_2d = fc[:, :, -1] if fc.ndim == 3 else fc
        n_lat, n_lon = density_2d.shape
        lons = np.linspace(bbox["lon_min"], bbox["lon_max"], n_lon)
        lats = np.linspace(bbox["lat_min"], bbox["lat_max"], n_lat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        mask = density_2d > 0
        if mask.any():
            fig.add_trace(go.Scattermapbox(
                lon=lon_grid[mask].ravel(),
                lat=lat_grid[mask].ravel(),
                mode="markers",
                marker=dict(
                    size=6,
                    color=np.log1p(density_2d[mask].ravel()),
                    colorscale="YlOrRd",
                    opacity=0.5,
                    colorbar=dict(title="Density (log)"),
                ),
                name="Plastic Density",
                hovertemplate="Lon: %{lon:.2f}<br>Lat: %{lat:.2f}<br>Density: %{marker.color:.2f}<extra></extra>",
            ))

    # --- vessel routes ---
    for i, route in enumerate(routes):
        if not route:
            continue
        color = VESSEL_COLORS[i % len(VESSEL_COLORS)]
        r_lon = [p[0] for p in route]
        r_lat = [p[1] for p in route]
        fig.add_trace(go.Scattermapbox(
            lon=r_lon, lat=r_lat,
            mode="lines+markers",
            marker=dict(size=7, color=color),
            line=dict(width=3, color=color),
            name=f"Vessel {i+1}",
            hovertemplate=f"Stop %{{pointNumber}}<br>Lon: %{{lon:.2f}}<br>Lat: %{{lat:.2f}}<extra>Vessel {i+1}</extra>",
        ))

    # --- port marker ---
    fig.add_trace(go.Scattermapbox(
        lon=[port[0]], lat=[port[1]],
        mode="markers+text",
        marker=dict(size=15, color="black", symbol="star"),
        text=["Port"],
        textposition="top center",
        name="Port",
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lon=float(port[0]), lat=float(port[1])),
            zoom=4,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig

# ---------------------------------------------------------------------------
# Helper: ensure each route returns to port
# ---------------------------------------------------------------------------
def ensure_return_to_port(routes, port):
    """Append port to end of each route if not already there."""
    fixed = []
    for route in routes:
        if not route:
            fixed.append(route)
            continue
        last = route[-1]
        if abs(last[0] - port[0]) > 0.01 or abs(last[1] - port[1]) > 0.01:
            route = list(route) + [port]
        fixed.append(route)
    return fixed

# ---------------------------------------------------------------------------
# Helper: route details table
# ---------------------------------------------------------------------------
def route_table(routes, plastic, fuel):
    """Create a per-vessel summary DataFrame."""
    rows = []
    for i, route in enumerate(routes):
        if not route:
            continue
        dist_deg = 0.0
        for j in range(1, len(route)):
            dx = route[j][0] - route[j-1][0]
            dy = route[j][1] - route[j-1][1]
            dist_deg += np.sqrt(dx**2 + dy**2)
        dist_km = fuel_deg_to_km(dist_deg)
        rows.append({
            "Vessel": f"Vessel {i+1}",
            "Waypoints": len(route),
            "Distance (km)": f"{dist_km:.1f}",
            "Distance (deg)": f"{dist_deg:.2f}",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Styled KPI card
# ---------------------------------------------------------------------------
KPI_CSS = """
<style>
.kpi-card {
    background: linear-gradient(135deg, #0077B6 0%, #023047 100%);
    border-radius: 10px; padding: 14px 18px; text-align: center; color: white;
}
.kpi-card .value { font-size: 1.8rem; font-weight: 700; margin: 2px 0; }
.kpi-card .label { font-size: 0.85rem; opacity: 0.85; }
.spacer { margin-top: 12px; }
</style>
"""

def kpi_card(label, value):
    return f'<div class="kpi-card"><div class="label">{label}</div><div class="value">{value}</div></div>'

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if optimize_btn:
    with st.spinner("Optimizing routes... this may take a moment."):
        routes, plastic, fuel, fc = query_route(departure_day, port, trip_days, n_vessels)

    # Normalise routes to list-of-lists
    if routes and not isinstance(routes[0], list):
        routes = [routes]
    routes = ensure_return_to_port(routes, port)

    fuel_km = fuel_deg_to_km(fuel)
    efficiency = plastic / fuel_km if fuel_km > 0 else 0.0

    # --- KPI row (styled cards) ---
    st.markdown(KPI_CSS, unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("Plastic Collected (score)", f"{plastic:.2f}"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Total Distance", f"{fuel_km:.1f} km"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Collection Efficiency", f"{efficiency:.4f} /km"), unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # --- Map + route table side by side ---
    map_col, table_col = st.columns([3, 1])
    with map_col:
        fig = build_map(routes, plastic, fuel, fc, port)
        st.plotly_chart(fig, use_container_width=True)
    with table_col:
        st.markdown("**Route Details**")
        df = route_table(routes, plastic, fuel)
        st.dataframe(df, use_container_width=True, hide_index=True, height=380)

elif find_best_btn:
    with st.spinner(f"Searching best departure from {date_start} to {date_end}..."):
        results = find_best_departure(port, trip_days, int(day_start), int(day_end), n_vessels)

    if not results:
        st.warning("No results found for the given range.")
    else:
        best = results[0]
        best_date = jan1 + datetime.timedelta(days=int(best['departure_day']))

        # --- KPI for best result ---
        st.markdown(KPI_CSS, unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.markdown(kpi_card("Best Departure", best_date.strftime("%b %d, %Y")), unsafe_allow_html=True)
        k2.markdown(kpi_card("Plastic Collected (score)", f"{best['plastic_collected']:.2f}"), unsafe_allow_html=True)
        k3.markdown(kpi_card("Collection Efficiency", f"{best['normalized_plastic']:.4f}"), unsafe_allow_html=True)
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # --- Chart + table side by side ---
        df = pd.DataFrame(results[:20])
        df["date"] = df["departure_day"].apply(lambda d: (jan1 + datetime.timedelta(days=int(d))).strftime("%Y-%m-%d"))
        if "fuel_consumed" in df.columns:
            df["fuel_km"] = df["fuel_consumed"].apply(fuel_deg_to_km)

        # Filter out zero-plastic results (failed runs) and keep top 10 by score
        df_valid = df[df["plastic_collected"] > 0].copy()
        top10 = df_valid.head(10).copy()
        # Sort chronologically for chart readability
        top10 = top10.sort_values("departure_day").reset_index(drop=True)

        chart_col, tbl_col = st.columns([2, 1])
        with chart_col:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=top10["date"], y=top10["plastic_collected"],
                name="Plastic Collected", marker_color="#0077B6",
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=top10["date"],
                y=top10["fuel_km"] if "fuel_km" in top10.columns else top10["fuel_consumed"],
                name="Distance (km)", mode="lines+markers",
                line=dict(color="#E76F51", width=2),
                marker=dict(size=8),
            ), secondary_y=True)
            fig.update_layout(
                xaxis_title="Departure Date",
                height=320, margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(yanchor="top", y=1.12, xanchor="left", x=0, orientation="h"),
                xaxis=dict(type="category"),
            )
            fig.update_yaxes(title_text="Plastic Collected (score)", secondary_y=False)
            fig.update_yaxes(title_text="Distance (km)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with tbl_col:
            st.markdown("**Top Departure Days**")
            st.dataframe(df_valid[["date", "plastic_collected", "fuel_km", "normalized_plastic"]],
                         use_container_width=True, hide_index=True, height=320)

else:
    # Default state — show instructions + port map
    st.info(
        "Configure mission parameters in the sidebar and click **Optimize Routes** "
        "to compute optimal vessel routes, or **Search Best Day** to find the ideal departure date."
    )
    fig = go.Figure(go.Scattermapbox(
        lon=[port[0]], lat=[port[1]],
        mode="markers+text",
        marker=dict(size=15, color="black", symbol="star"),
        text=[port_name if port_name != "Custom" else "Custom Port"],
        textposition="top center",
    ))
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lon=float(port[0]), lat=float(port[1])),
            zoom=4,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
