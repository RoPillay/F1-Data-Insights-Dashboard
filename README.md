# Interactive F1 Race Data Dashboard 
**See it in Action:** [Launch on Streamlit](https://f1-data-insights-dashboard-2vzi6vwbfddhuqkzqbspqo.streamlit.app/)

## Features
- **Session Overview**
  - Fastest lap summary for selected driver(s)
- **Driver Telemetry Comparison**
  - Looks into overlay speed, throttle, brake, and gear data for two drivers on their fastest laps
- **Delta Time Visualization**
  - Analyze where one driver gains or loses time against another throughout the selected session
- **Race Consistency**
  - A lap-by-lap analysis for a singular driver during the race looking at:
    - Lap time distribution based on tire compound
    - Lap-by-lap pace and degradation
    - Stint strategy summary
- **Track Map**
  - Feature heat map based on speed
    - Circuit is pictured with a color gradient representing car's speed at each location
- **Stint Strategy Analysis**
  - Used a **Random Forest Regressor** to predict stint degradation for future laps and classify each lap as optimal, normal, or worn/traffic-affected
- **Interactive Plots** using **Plotly** with zoom, hover tooltips, and multi-driver selection
- **Real-time Data Fetching** from the FastF1 API
- **Streamlit Web App Deployment** - No installation required, can be run directly in browser

## Tech Stack
- **Programming Language:** Python 3.11
- **Framework:** [Streamlit](https://streamlit.io/)  
- **Data Source:** [FastF1](https://theoehrly.github.io/Fast-F1/) – Live F1 car telemetry and timing data
- **Data Visualization:** Plotly, Matplotlib  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **Other Libraries:** Pandas, NumPy

## How to Use
1. **Click the live app link:**
   [Open the Dashboard](https://f1-data-insights-dashboard-2vzi6vwbfddhuqkzqbspqo.streamlit.app/)
2. **Select a race session:**
   - Select year (2018-present), round (number or name), Session type (Qualifying or Race)
3. **Choose one or two drivers** to compare and view telemetry data
4. Enable **Predictive Mode** when looking at a singular driver to analyze tire degradation and stint quality, view map, etc.
5. **Explore interactive charts**, lap times, sector breakdowns, delta plots, etc.

## Author
- Rohan Pillay UC Davis Class of 2026
- [LinkedIn](https://www.linkedin.com/in/rohan-pillay-098902323/)

## Credits
- **[FastF1 Library](https://github.com/theOehrly/Fast-F1)**
  - Provided open-sourced F1 data
- **ChatGPT**
  - Assistance with code structure (e.g. recommending tabs, RandomForest help)
- **GitHub Copilot**
  - Helped resolving code errors and improving syntax (e.g. debugging caching errors, efficient pandas merging)

## Future Improvements
- Add **live session pace evolution plots**
  - Tracks cars through the circuit live and displays data
- Include a **driver consistency/performance score** using variance in lap times and overall race weekend performance
- Create a **pit stop strategy simulator**
  - Create a simulator to analyze when the best time to pit stop is
- Create **predictive race strategies**
  - Look at historical data and simulator data to predict lap times based on degradation, weather, tire strategy, etc.
  - Based on that determine optimal race strategies for tire compounds, car setup, pit stops, etc. before a race weekend
