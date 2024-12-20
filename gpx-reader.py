# This app will read gpx files and a kml file and calculate travel times, visualized route over time, and display points on a map.
# the app is built using Git and hosted on Streamlit.

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import Draw
import branca
import branca.colormap as cm
import os
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import pytz

# Define Salt Lake City's timezone
LOCAL_TZ = pytz.timezone('America/Denver')

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GPX Reader',
    page_icon=':car:',
)

# Add title and logo
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <img src="https://avenueconsultants.com/wp-content/themes/avenuecustom/webflow/images/a-blue.png" 
             style="width: 100px; margin-left: 20px;">
        <div>
            <h1>GPX Route Analysis Tool</h1>
            <p>Upload GPX files and a KML file to analyze travel times and visualize routes.</p>
        </div>
        
    </div>
    """, 
    unsafe_allow_html=True
)

# Add some vertical space
st.markdown("<br>", unsafe_allow_html=True)

# File upload section
st.header('File Upload')
st.write('The GPX files should come from a GPX app that has a time and location at each point.')
st.write(
    '''
    The KML file should consist of points (in order) that represent the intersections on the route.
    If travel times at the end points are important for the analysis, consider adding a point at the start and end of the route.
    '''
)
uploaded_gpx_files = st.file_uploader('Upload GPX files', type=['gpx'], accept_multiple_files=True)
uploaded_kml = st.file_uploader('Upload KML file', type=['kml'])

# Direction selector
direction = st.radio('Select Direction:', ['NS', 'EW'])


# Function to calculate travel time between two timestamps
def calculate_travel_time(start_time, end_time):
    start_datetime = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    end_datetime = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
    return (end_datetime - start_datetime).total_seconds()


# Function to parse GPX file and calculate speed
@st.cache_data
def parse_gpx(file, _key_intersections):
    tree = ET.parse(file)
    root = tree.getroot()

    namespaces = {
        'gpx': 'http://www.topografix.com/GPX/1/1'
    }
    
    data = []
    previous_point = None
    previous_time = None
    previous_intersection = None
    previous_intersection_time = None
    travel_times = []
    distance_threshold = 0.009  # approximately 50 feet in miles
    
    for trkpt in root.findall('.//gpx:trkpt', namespaces):
        timestamp = trkpt.find('gpx:time', namespaces).text
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        
        try:
            time_of_day_utc = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            time_of_day_utc = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
        
        time_of_day_local = time_of_day_utc.replace(tzinfo=pytz.utc).astimezone(LOCAL_TZ)
        
        # Find closest intersection
        closest_intersection = None
        min_distance = float('inf')
        for _, intersection in _key_intersections.iterrows():
            distance = haversine(lat, lon, intersection['Latitude'], intersection['Longitude'])
            if distance < min_distance and distance <= distance_threshold:
                min_distance = distance
                closest_intersection = intersection
        
        # If we found a close intersection and had a previous intersection
        if closest_intersection is not None and previous_intersection is not None:
            # Only record if it's a different intersection
            if closest_intersection['Name'] != previous_intersection['Name']:
                travel_times.append({
                    'origin_index': previous_intersection['Intersection'],
                    'route_id': str(previous_intersection['Intersection']) + '-' + str(closest_intersection['Intersection']),
                    'direction': '2' if (previous_intersection['Intersection'] - closest_intersection['Intersection']) > 0 else '1',
                    'start_intersection': previous_intersection['Name'],
                    'end_intersection': closest_intersection['Name'],
                    'start_time': previous_intersection_time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'end_time': time_of_day_local.strftime('%Y-%m-%dT%H:%M:%S'),
                    'travel_time_seconds': (time_of_day_local - previous_intersection_time).total_seconds()
                })
                previous_intersection = closest_intersection
                previous_intersection_time = time_of_day_local
        elif closest_intersection is not None:
            previous_intersection = closest_intersection
            previous_intersection_time = time_of_day_local
        
        if previous_point is not None and previous_time is not None:
            distance = haversine(previous_point[0], previous_point[1], lat, lon)
            time_diff = (time_of_day_local - previous_time).total_seconds() / 3600.0
            speed = distance / time_diff if time_diff != 0 else 0
        else:
            speed = 0.0
        
        data.append([time_of_day_local.strftime('%H:%M:%S'), lat, lon, speed])
        previous_point = (lat, lon)
        previous_time = time_of_day_local
    
    df = pd.DataFrame(data, columns=['TimeOfDay', 'Latitude', 'Longitude', 'Speed'])
    travel_times_df = pd.DataFrame(travel_times)
    
    bins = [-float('inf'), 3, 15, 30, float('inf')]
    labels = ['Below 3 mph', '3-15 mph', '15-30 mph', 'Above 30 mph']
    df['SpeedCategory'] = pd.cut(df['Speed'], bins=bins, labels=labels)

    speed_category_order = ['Below 3 mph', '3-15 mph', '15-30 mph', 'Above 30 mph']
    df['SpeedCategory'] = pd.Categorical(df['SpeedCategory'], categories=speed_category_order, ordered=True)

    return df, travel_times_df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0*0.621371  # Earth radius in miles
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

@st.cache_data
def parse_kml(file):
    gdf = gpd.read_file(file, driver='KML')
    key_intersections = gdf[['geometry', 'Name']].copy()
    key_intersections['Longitude'] = key_intersections['geometry'].x
    key_intersections['Latitude'] = key_intersections['geometry'].y
    key_intersections['Intersection'] = range(len(key_intersections))
    return key_intersections

def map_to_intersections(df, key_intersections, direction):
    intersection_to_lat = key_intersections.set_index('Name')['Latitude'].to_dict()
    intersection_to_lon = key_intersections.set_index('Name')['Longitude'].to_dict()

    def get_nearest_intersection(lat, lon):
        min_distance = float('inf')
        nearest_intersection = None
        for _, row in key_intersections.iterrows():
            dist = haversine(lat, lon, row.geometry.y, row.geometry.x)
            if dist < min_distance:
                min_distance = dist
                nearest_intersection = row['Intersection']
        return nearest_intersection
    
    df['Intersection'] = df.apply(lambda row: get_nearest_intersection(row['Latitude'], row['Longitude']), axis=1)
    df = df.merge(key_intersections[['Intersection', 'Name']], on='Intersection', how='left')

    df['LatitudeForPlot'] = df['Name'].map(intersection_to_lat)
    df['LongitudeForPlot'] = df['Name'].map(intersection_to_lon)
    
    return df, intersection_to_lat if direction == 'NS' else intersection_to_lon

def make_time_plot(data, intersection_coords, direction):
    fig = go.Figure()
    
    y_column = 'Latitude' if direction == 'NS' else 'Longitude'
    
    fig.add_trace(go.Scatter(
        x=data['TimeOfDay'],
        y=data[y_column],
        mode='markers',
        marker=dict(
            color=data['Speed'],
            colorscale=[[0, 'red'], [0.05, 'yellow'], [0.3, 'lightgreen'], 
                       [0.5, 'green'], [0.8, 'darkgreen'], [1, 'purple']],
            colorbar=dict(title='Speed Scale'),
            size=10
        ),
        text=data.apply(lambda row: f"Time: {row['TimeOfDay']}<br>Speed: {row['Speed']:.1f} mph", axis=1),
        hoverinfo='text'
    ))

    fig.update_yaxes(
        tickvals=list(intersection_coords.values()),
        ticktext=list(intersection_coords.keys()),
        title='Key Intersections'
    )

    fig.update_xaxes(
        tickformat='%H:%M',
        dtick=360
    )

    fig.update_layout(
        title=f'Key Intersections vs Time of Day with Speed Gradient ({direction} Direction)',
        width=1000,
        height=800,
        autosize=True
    )

    return fig  # Return the figure instead of showing it

def plot_data_on_map(df, min_speed, max_speed, start_time, end_time):
    m = folium.Map(tiles="cartodb positron", 
                  location=[df['Latitude'].mean(), df['Longitude'].mean()], 
                  zoom_start=12)
    
    # Filter data based on speed range and time range
    df_filtered = df[
        (df['Speed'] >= min_speed) & 
        (df['Speed'] <= max_speed) &
        (df['TimeOfDay'] >= start_time) &
        (df['TimeOfDay'] <= end_time)
    ]
    
    colors = {
        'Below 3 mph': 'red',
        '3-15 mph': 'orange',
        '15-30 mph': 'yellow',
        'Above 30 mph': 'green'
    }

    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=colors[row['SpeedCategory']],
            fill_color=colors[row['SpeedCategory']],
            fill_opacity=0.7,
            legend_name='Speed from GPX Data',
            popup=f"Time: {row['TimeOfDay']}<br>Speed: {row['Speed']} mph<br>Lat: {row['Latitude']}<br>Lon: {row['Longitude']}"
        ).add_to(m)
    
    return m

def process_travel_times(travel_times_df):
    # Format into table
    filtered_df = travel_times_df[travel_times_df['start_intersection'] != travel_times_df['end_intersection']]
    filtered_df.loc[:,'route'] = filtered_df['start_intersection'] + '_to_' + filtered_df['end_intersection']
    filtered_df['run_number'] = filtered_df.groupby('route').cumcount() + 1

    # Convert 'start_time' to datetime and extract just the time component
    filtered_df['start_time'] = pd.to_datetime(filtered_df['start_time'])
    filtered_df['time_of_day'] = filtered_df['start_time'].dt.strftime('%H:%M')
    
    # Round to nearest 15 minutes
    filtered_df['time_bin'] = pd.to_datetime(filtered_df['start_time']).dt.round('15min').dt.strftime('%H:%M')

    # Pivot the table based on the time_bin column
    pivoted_df = filtered_df.pivot_table(
        index=['direction', 'origin_index', 'route'], 
        columns='time_bin', 
        values='travel_time_seconds',
        aggfunc='mean'  # Changed to mean to handle multiple values in same time bin
    )

    # Sort the columns chronologically
    time_columns = sorted(pivoted_df.columns)
    pivoted_df = pivoted_df[time_columns]

    # Calculate the average and std_deviation
    pivoted_df['average'] = pivoted_df.mean(axis=1)
    pivoted_df['std_deviation'] = pivoted_df.std(axis=1)

    return pivoted_df
# -----------------------------------------------------------------------------------------
# Main app logic --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

if uploaded_gpx_files and uploaded_kml:
    st.header('Analysis Results')
    
    try:    
        # Process KML file with caching
        key_intersections = parse_kml(uploaded_kml)

        # Process GPX files and calculate travel times with caching
        all_gpx_data = []
        all_travel_times_df = []
        for gpx_file in uploaded_gpx_files:
            df, travel_times_df = parse_gpx(gpx_file, key_intersections)
            all_gpx_data.append(df)
            all_travel_times_df.append(travel_times_df)
        gpx_data = pd.concat(all_gpx_data, ignore_index=True)
        travel_times_df = pd.concat(all_travel_times_df, ignore_index=True)
        
        # Map intersections
        gpx_data, intersection_coords = map_to_intersections(gpx_data, key_intersections, direction)
        
        # Sort data
        gpx_data = gpx_data.sort_values(by='TimeOfDay')

        # Display travel times
        st.subheader('Travel Times')
        formatted_travel_times_df = process_travel_times(travel_times_df)
        st.dataframe(formatted_travel_times_df)
        
        # Display time plot
        st.subheader('Route Over Time Plot')
        fig = make_time_plot(gpx_data, intersection_coords, direction)
        fig.update_layout(width=1200, height=800)  # Set both height and width
        st.plotly_chart(fig, use_container_width=True)
        
        # Display map
        # Add filter sliders before the map
        st.subheader('Route Map')
        
        # Speed filter
        min_speed = float(gpx_data['Speed'].min())
        max_speed = float(gpx_data['Speed'].max())
        speed_range = st.slider(
            'Filter by Speed (mph)',
            min_value=min_speed,
            max_value=max_speed,
            value=(min_speed, max_speed),
            step=1.0
        )
        
        # Time filter - convert to datetime.time objects for better handling
        time_values = pd.to_datetime(gpx_data['TimeOfDay']).dt.time.unique()
        time_values = sorted(time_values)
        time_range = st.select_slider(
            'Filter by Time of Day',
            options=time_values,
            value=(time_values[0], time_values[-1]),
            format_func=lambda x: x.strftime('%H:%M:%S')
        )
        
        # Convert selected times to strings for comparison
        start_time_str = time_range[0].strftime('%H:%M:%S')
        end_time_str = time_range[1].strftime('%H:%M:%S')
        
        # Update map with filtered data
        m = plot_data_on_map(gpx_data, 
                           speed_range[0], 
                           speed_range[1],
                           start_time_str,
                           end_time_str)
        st.components.v1.html(m._repr_html_(), height=800)

        
        
        # Optional: Display data table
        if st.checkbox('Show Raw GPX Data'):
            st.dataframe(gpx_data)
            
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
else:
    st.info('Please upload your GPX and KML files to begin analysis.')