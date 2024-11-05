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

# Add title and description
st.title('GPX Route Analysis Tool')
st.write('Upload GPX files and a KML file to analyze travel times and visualize routes.')

# File upload section
st.header('File Upload')
uploaded_gpx_files = st.file_uploader('Upload GPX files', type=['gpx'], accept_multiple_files=True)
uploaded_kml = st.file_uploader('Upload KML file', type=['kml'])

# Direction selector
direction = st.radio('Select Direction:', ['NS', 'EW'])

# Function to parse GPX file and calculate speed
def parse_gpx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    namespaces = {
        'gpx': 'http://www.topografix.com/GPX/1/1'
    }
    
    data = []
    previous_point = None
    previous_time = None
    
    for trkpt in root.findall('.//gpx:trkpt', namespaces):
        timestamp = trkpt.find('gpx:time', namespaces).text
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        
        try:
            time_of_day_utc = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            time_of_day_utc = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
        
        time_of_day_local = time_of_day_utc.replace(tzinfo=pytz.utc).astimezone(LOCAL_TZ)
        
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

    bins = [-float('inf'), 3, 15, 30, float('inf')]
    labels = ['Below 3 mph', '3-15 mph', '15-30 mph', 'Above 30 mph']
    df['SpeedCategory'] = pd.cut(df['Speed'], bins=bins, labels=labels)

    speed_category_order = ['Below 3 mph', '3-15 mph', '15-30 mph', 'Above 30 mph']
    df['SpeedCategory'] = pd.Categorical(df['SpeedCategory'], categories=speed_category_order, ordered=True)

    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0*0.621371  # Earth radius in miles
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

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
        text=data['Name'],
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
        title=f'Key Intersections vs Time of Day with Speed Gradient ({direction} Direction)'
    )

    return fig  # Return the figure instead of showing it

def plot_data_on_map(df):
    m = folium.Map(tiles="cartodb positron", 
                  location=[df['Latitude'].mean(), df['Longitude'].mean()], 
                  zoom_start=12)
    
    colors = {
        'Below 3 mph': 'red',
        '3-15 mph': 'orange',
        '15-30 mph': 'yellow',
        'Above 30 mph': 'green'
    }

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=colors[row['SpeedCategory']],
            fill_color=colors[row['SpeedCategory']],
            fill_opacity=0.7,
            legend_name='Speed from GPX Data',
            popup=f"Time: {row['TimeOfDay']}<br>Speed: {row['Speed']} mph<br>Lat: {row['Latitude']}<br>Lon: {row['Longitude']}"
        ).add_to(m)
    
    return m  # Return the map object instead of saving it

# -----------------------------------------------------------------------------------------
# Main app logic --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

if uploaded_gpx_files and uploaded_kml:
    st.header('Analysis Results')
    
    try:
        # Process GPX files
        all_gpx_data = []
        for gpx_file in uploaded_gpx_files:
            df = parse_gpx(gpx_file)
            all_gpx_data.append(df)
        
        gpx_data = pd.concat(all_gpx_data, ignore_index=True)
        
        # Process KML file
        key_intersections = parse_kml(uploaded_kml)
        
        # Map intersections
        gpx_data, intersection_coords = map_to_intersections(gpx_data, key_intersections, direction)
        
        # Sort data
        gpx_data = gpx_data.sort_values(by='TimeOfDay')
        
        # Display time plot
        st.subheader('Time vs. Location Plot')
        fig = make_time_plot(gpx_data, intersection_coords, direction)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display map
        st.subheader('Route Map')
        m = plot_data_on_map(gpx_data)
        st.components.v1.html(m._repr_html_(), height=600)
        
        # Optional: Display data table
        if st.checkbox('Show Raw Data'):
            st.dataframe(gpx_data)
            
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
else:
    st.info('Please upload your GPX and KML files to begin analysis.')