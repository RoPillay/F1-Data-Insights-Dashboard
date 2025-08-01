import fastf1
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots

# Enable cache
import os

cache_dir = "./fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

st.set_page_config(layout="wide")
st.title("🏎️ F1 Race Analysis Dashboard")

# Inputs
year = st.number_input("Enter Year", min_value=2018, max_value=2025, value=2025)
round_input = st.text_input("Enter Grand Prix (name or round number)", "Monaco")
session_type = st.selectbox("Session Type", ["Q", "R"])
compare_mode = st.checkbox("Compare Two Drivers")
analysis_mode = st.checkbox("Race and Qualifying Analysis (Singular Driver)")

# If compare mode 
if compare_mode:
    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.text_input("Driver 1 Code (e.g. LEC)", "LEC")
    with col2:
        driver2 = st.text_input("Driver 2 Code (e.g. VER)", "VER")
else:
    driver = st.text_input("Driver Code (e.g. LEC, VER, NOR)", "LEC")

# When analyze button is clicked
if st.button("Analyze"):
    try:
        session = fastf1.get_session(year, round_input, session_type.upper())
        session.load()

        # Creating a Results Table
        # Shows entire session results, driver, numer, position, and team
        try:
            results = pd.DataFrame()
            if 'Position' in session.results.columns:
                results['Position'] = session.results['Position']

            if 'Abbreviation' in session.results.columns:
                results['Driver'] = session.results['Abbreviation']
            elif 'BroadcastName' in session.results.columns:
                results['Driver'] = session.results['BroadcastName']
            elif 'DriverNumber' in session.results.columns:
                results['Driver'] = session.results['DriverNumber']

            if 'TeamName' in session.results.columns:
                results['Team'] = session.results['TeamName']

            if 'LapTime' in session.results.columns:
                results['LapTime'] = session.results['LapTime'].dt.total_seconds().map(lambda x: f"{x:.3f}")
            if 'TimeDelta' in session.results.columns:
                results['TimeDelta'] = session.results['TimeDelta'].dt.total_seconds().map(lambda x: f"{x:.3f}")

            if not results.empty:
                st.subheader("📋 Session Results")
                st.dataframe(results)
            else:
                st.warning("Could not build results table: Missing expected fields.")

        except Exception as e:
            st.warning(f"Could not load the session results table: {e}")

        # Time Formatting 
        def format_time(t):
            if pd.isnull(t):
                return "N/A"
            total_seconds = t.total_seconds()
            return f"{int(total_seconds // 60)}:{total_seconds % 60:06.3f}"

        # If Compare Mode is selected 
        # Shows side by side fastest lap comparison, sector times, finishing position
        if compare_mode:
            laps1 = session.laps.pick_driver(driver1.upper())
            laps2 = session.laps.pick_driver(driver2.upper())

            if laps1.empty or laps2.empty:
                st.warning("One or both drivers have no data.")
            else:
                lap1, lap2 = laps1.pick_fastest(), laps2.pick_fastest()

                if lap1 is None or lap2 is None:
                    st.warning("One or both drivers have no valid lap data (possible DNF or crash).")
                else:
                    st.subheader(f"🚗 Comparison: {driver1.upper()} vs {driver2.upper()} – {round_input} {year} ({session_type})")

                    col1, col2 = st.columns(2)
                    for col, lap, drv, color in zip([col1, col2], [lap1, lap2], [driver1, driver2], ["darkorange", "crimson"]):
                        with col:
                            st.markdown(f"### {drv.upper()}")
                            st.write(f"Lap Time: {format_time(lap['LapTime'])}")
                            st.write(f"Sector 1: {format_time(lap['Sector1Time'])}")
                            st.write(f"Sector 2: {format_time(lap['Sector2Time'])}")
                            st.write(f"Sector 3: {format_time(lap['Sector3Time'])}")
                            if pd.notnull(lap.get('Position')):
                                st.write(f"Finishing Position: {int(lap['Position'])}")

                
                    # Telemetry Plot
                    # Shows telemetry comparison (speed, throttle, brake, gear)
                    try:
                        tel1 = lap1.get_car_data()
                        tel2 = lap2.get_car_data()

                        if tel1 is None or tel2 is None:
                            st.warning("One or both drivers don't have telemetry data availble for this session (possible DNF)")
                        else:
                            tel1 = tel1.add_distance()
                            tel2 = tel2.add_distance()

                            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=("Speed", "Throttle", "Brake", "Gear"))

                            fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Speed'], name=f"{driver1.upper()} Speed", line=dict(color="darkorange")), row=1, col=1)
                            fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Speed'], name=f"{driver2.upper()} Speed", line=dict(color="crimson")), row=1, col=1)

                            fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Throttle'], name=f"{driver1.upper()} Throttle", line=dict(color="darkorange")), row=2, col=1)
                            fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Throttle'], name=f"{driver2.upper()} Throttle", line=dict(color="crimson")), row=2, col=1)

                            fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Brake'], name=f"{driver1.upper()} Brake", line=dict(color="darkorange")), row=3, col=1)
                            fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Brake'], name=f"{driver2.upper()} Brake", line=dict(color="crimson")), row=3, col=1)

                            fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['nGear'], name=f"{driver1.upper()} Gear", line=dict(color="darkorange")), row=4, col=1)
                            fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['nGear'], name=f"{driver2.upper()} Gear", line=dict(color="crimson")), row=4, col=1)

                            fig.update_layout(height=900, title="Telemetry Comparison")
                            st.plotly_chart(fig)

                            # Delta Plot
                            # Also shows delta time over the lap(postive delta: driver code 1 is ahead of 2, negative delta: driver code 2 is ahead of 1)
                            ref_dist = tel1['Distance']
                            t1 = tel1['Time'].reset_index(drop=True)
                            t2 = tel2.set_index('Distance')['Time'].reindex(ref_dist, method='nearest').reset_index(drop=True)
                            delta = (t2 - t1).dt.total_seconds()

                            fig_delta = go.Figure()
                            fig_delta.add_trace(go.Scatter(x=ref_dist, y=delta,
                                               mode='lines',
                                               name=f"Δ Time ({driver2.upper()} - {driver1.upper()})",
                                               line=dict(color='purple')))
                            fig_delta.update_layout(title="Time Delta Over Lap Distance",
                                        xaxis_title="Distance (m)",
                                        yaxis_title="Time Delta (s)",
                                        annotations=[
                                            dict(
                                                text=f"Positive = {driver1.upper()} ahead, Negative = {driver2.upper()} ahead",
                                                xref="paper", yref="paper",
                                                x=0.5, y=1.07,
                                                showarrow=False,
                                                font=dict(size=12)
                                            )
                                        ])
                            st.plotly_chart(fig_delta)
                    except Exception as e:
                        st.error("Telemetry data is unavailable for one or both drivers.")


        # Single Driver Mode
        # Not compare mode, shows data for singlular driver chosen
        else:
            laps = session.laps.pick_driver(driver.upper())

            clean_laps = laps[laps['LapTime'].notnull()].copy()
            if not clean_laps.empty:
                clean_laps['LapTime_s'] = clean_laps['LapTime'].dt.total_seconds()

            if laps.empty or laps.pick_fastest() is None:
                st.warning(f"{driver.upper()} has no valid lap data for this session - possible DNF or no timed laps")
            else:
                fastest_lap = laps.pick_fastest()

                # Tabs for organization
                tabs = st.tabs(["🏁 Session Overview", "📊 Telemetry Analysis", "🛠️ Race Consistency", "🗺️ Track Map", "🤔Predictions"])

                # Session Overview
                with tabs[0]:
                    st.subheader(f"Fastest Lap Summary – {driver.upper()}")
                    is_race = session.session_info['Name'].lower() == 'race'

                    # Extracting lap times
                    sector1 = fastest_lap['Sector1Time'].total_seconds()
                    sector2 = fastest_lap['Sector2Time'].total_seconds()
                    sector3 = fastest_lap['Sector3Time'].total_seconds()
                    compound = fastest_lap['Compound'] if 'Compound' in fastest_lap else "Unknown"
                    
                    if is_race:
                        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    else:
                        col1, col2, col3, col4, col5 = st.columns(5)

                    # Display metrics
                    col1.metric("Best Lap", format_time(fastest_lap['LapTime']))
                    col2.metric("Sector 1", f"{sector1:.3f}s")
                    col3.metric("Sector 2", f"{sector2:.3f}s")
                    col4.metric("Sector 3", f"{sector3:.3f}s")
                    col5.metric("Compound", compound)
                    
                    # Show Stint and Lap only for Race sessions
                    if is_race:
                        stint = int(fastest_lap.get('Stint', 1))
                        lap_number = int(fastest_lap.get('LapNumber', -1))
                        col6.metric("Stint", stint)
                        col7.metric("Lap #", lap_number)


                # Telemetry Analysis Graphs
                tel = fastest_lap.get_car_data().add_distance()
                with tabs[1]:
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        subplot_titles=("Speed", "Throttle", "Brake", "Gear"))
                    fig.add_trace(go.Scatter(x=tel['Distance'], y=tel['Speed'], name="Speed"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=tel['Distance'], y=tel['Throttle'], name="Throttle"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=tel['Distance'], y=tel['Brake'], name="Brake"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=tel['Distance'], y=tel['nGear'], name="Gear"), row=4, col=1)
                    fig.update_layout(height=900, title="Telemetry for Fastest Lap")
                    st.plotly_chart(fig)

                # Race Consistency Tab (if race selected)
                if session_type.upper() == "R":
                    with tabs[2]:
                        st.subheader(f"Lap-by-Lap Consistency for {driver.upper()}")
                        clean_laps = laps[laps['LapTime'].notnull()].copy()
                        clean_laps['LapTime_s'] = clean_laps['LapTime'].dt.total_seconds()

                        # Boxplot by compound
                        fig_box = px.box(clean_laps, x='Compound', y='LapTime_s', points="all",
                                         color='Compound', title="Lap Time Distribution by Tyre Compound",
                                         labels={'LapTime_s': 'Lap Time (s)'})
                        st.plotly_chart(fig_box)

                        # Line plot of lap time vs lap number
                        fig_line = go.Figure()
                        for comp in clean_laps['Compound'].unique():
                            stint = clean_laps[clean_laps['Compound'] == comp]
                            fig_line.add_trace(go.Scatter(x=stint['LapNumber'], y=stint['LapTime_s'],
                                                          mode='lines+markers', name=f"{comp} Tyre",
                                                          line=dict(width=2)))
                        fig_line.update_layout(title="Lap-by-Lap Pace and Degradation",
                                               xaxis_title="Lap Number",
                                               yaxis_title="Lap Time (s)")
                        st.plotly_chart(fig_line)

                        # Stint Strategy Summary Table
                        st.subheader("Stint Strategy Summary")
                        all_laps = laps.copy()
                        all_laps['LapTime_s'] = all_laps['LapTime'].dt.total_seconds()
                        stint_summary = (all_laps.groupby(['Stint', 'Compound'])
                                         .agg(Stint_Length=('LapNumber', 'count'),
                                              Avg_LapTime_s=('LapTime_s', 'mean'))
                                         .reset_index()
                                         .sort_values('Stint'))
                        st.dataframe(stint_summary.rename(columns={
                            'Stint': 'Stint Number',
                            'Compound': 'Tyre Compound',
                            'Stint_Length': 'Stint Length (laps)',
                            'Avg_LapTime_s': 'Avg Lap Time (s)'}))

                # Track Map Visualization
                with tabs[3]:
                    try:
                         # Get positional and telemetry data
                        pos = fastest_lap.get_pos_data().dropna(subset=['X', 'Y'])
                        tel = fastest_lap.get_telemetry().dropna(subset=['Time', 'Speed'])

                        # Ensure numeric data
                        pos[['X', 'Y']] = pos[['X', 'Y']].apply(pd.to_numeric, errors='coerce')
                        tel['Speed'] = pd.to_numeric(tel['Speed'], errors='coerce')

                        # Merge data
                        pos = pos.merge(tel[['Time', 'Speed']], on='Time', how='inner').dropna()

                        # Plot track map
                        fig_map = px.scatter(pos,
                            x='X', y='Y',
                            color='Speed',
                            color_continuous_scale='Turbo',
                            title=f"Track Map – Speed Overlay ({driver.upper()})",
                            labels={'Speed': 'Speed (km/h)'},
                            hover_data=['Speed']
                        )
                        fig_map.update_traces(marker=dict(size=6))
                        fig_map.update_yaxes(scaleanchor="x", scaleratio=1)  # Keep proportions correct
                        fig_map.update_layout(height=700)
                        st.plotly_chart(fig_map, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Could not generate track map: {e}")

                # Predictive Mode
                # Predicting the next few laps after race ends based on degradation
                with tabs[4]:
                    if analysis_mode:
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.preprocessing import LabelEncoder

                        model_laps = clean_laps.copy()
                        if not model_laps.empty:
                            model_laps['Compound_encoded'] = LabelEncoder().fit_transform(model_laps['Compound'].astype(str))
                            model_laps['TrackStatus_encoded'] = LabelEncoder().fit_transform(model_laps['TrackStatus'].astype(str))

                            features = ['LapNumber', 'Compound_encoded', 'Stint', 'TrackStatus_encoded']
                            target = model_laps['LapTime'].dt.total_seconds()

                            # Training model
                            model = RandomForestRegressor(n_estimators=100, random_state=21)
                            model.fit(model_laps[features], target)

                            # Starting point
                            last_lap = model_laps['LapNumber'].max()
                            last_stint = model_laps['Stint'].iloc[-1]
                            last_compound = model_laps['Compound'].iloc[-1]
                            stint_laps = model_laps[(model_laps['Stint'] == last_stint) & 
                                    (model_laps['Compound'] == last_compound)]
                            
                            # --- Calculate data-driven degradation rate (s/lap) ---
                            if len(stint_laps) >= 4:
                                x = stint_laps['LapNumber']
                                y = target[stint_laps.index]
                                slope, intercept = np.polyfit(x, y, 1)  # linear fit
                                degradation_rate = max(0, slope)  # ensure non-negative
                            else:
                                degradation_rate = 0.05  # fallback value for short stints

                            # Estimated Degradation Rate
                            st.markdown(
                                f"**Estimated Degradation Rate:** {degradation_rate:.3f} sec/lap "
                                f"(Compound: {last_compound}, Stint: {last_stint})"
                            )
                            
                            # Create future data
                            future_laps = 5
                            future_data = pd.DataFrame({
                                'LapNumber': range(int(last_lap) + 1, int(last_lap) + future_laps + 1),
                                'Compound_encoded': model_laps['Compound_encoded'].iloc[-1],
                                'Stint': last_stint,
                                'TrackStatus_encoded': model_laps['TrackStatus_encoded'].iloc[-1]
                            })

                            # Predict base lap times
                            base_preds = model.predict(future_data)
                            # Apply degradation incrementally
                            lap_offsets = np.arange(1, future_laps + 1)
                            adjusted_preds = base_preds + (lap_offsets * degradation_rate)

                            # Combine past and predicted data
                            past_df = pd.DataFrame({'LapNumber': model_laps['LapNumber'], 'LapTime': target, 'is_prediction': False})
                            pred_df = pd.DataFrame({'LapNumber': future_data['LapNumber'], 'LapTime': adjusted_preds, 'is_prediction': True})
                            combined = pd.concat([past_df, pred_df])

                            # Classify degradation levels
                            median = past_df['LapTime'].median()
                            std = past_df['LapTime'].std()
                            def classify(x):
                                if x <= median:
                                    return 'Optimal'
                                elif x <= median + 0.5 * std:
                                    return 'Normal'
                                else:
                                    return 'Worn'
                            combined['Degradation'] = combined['LapTime'].apply(classify)
                            combined['Label'] = combined.apply(lambda row: f"{'Predicted' if row['is_prediction'] else 'Actual'} - {row['Degradation']}", axis=1)

                            # Plot
                            fig_pred = px.scatter(combined, x='LapNumber', y='LapTime', color='Label',
                                title='Lap Time Prediction and Degradation Classification',
                                labels={'LapTime': 'Lap Time (s)', 'LapNumber': 'Lap Number'})
                            st.plotly_chart(fig_pred)

    except Exception as e:
        st.error(f"Error loading data: {e}")
