import requests
import folium
from geopy.distance import geodesic
import math
import os
from folium.plugins import MarkerCluster

def get_user_location():
    # Return hardcoded location for testing
    return 23.5470, 87.2902

def find_nearby_hospitals(latitude, longitude, radius=10000):
    # Use Overpass API to find nearby hospitals
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"="hospital"](around:{radius},{latitude},{longitude});
    out body;
    """
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    return data['elements']

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    initial_bearing = math.atan2(x, y)
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360
    return compass_bearing

def create_map(latitude, longitude, hospitals):
    user_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=13,
        tiles='CartoDB positron',  # More realistic tile style
        attr='&copy; <a href="http://stamen.com">Stamen Design</a> | Data by <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
    )

    folium.Marker(
        [latitude, longitude], 
        tooltip="You are here", 
        icon=folium.Icon(color="blue", icon='info-sign')
    ).add_to(user_map)

    marker_cluster = MarkerCluster().add_to(user_map)

    for hospital in hospitals:
        hosp_lat = hospital['lat']
        hosp_lon = hospital['lon']
        hosp_name = hospital.get('tags', {}).get('name', 'Unnamed Hospital')
        hosp_address = hospital.get('tags', {}).get('addr:full', 'Address not available')
        hosp_phone = hospital.get('tags', {}).get('phone', 'No phone number available')  # Extract phone number

        user_location = (latitude, longitude)
        hospital_location = (hosp_lat, hosp_lon)

        distance = geodesic(user_location, hospital_location).kilometers
        bearing = calculate_bearing(latitude, longitude, hosp_lat, hosp_lon)

        direction = ""
        if bearing >= 0 and bearing < 90:
            direction = "NE"
        elif bearing >= 90 and bearing < 180:
            direction = "SE"
        elif bearing >= 180 and bearing < 270:
            direction = "SW"
        else:
            direction = "NW"

        popup_text = f"""
        <div style="font-size: 16px; line-height: 1.5; color: #333; background-color: #fff; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
            <strong style="font-size: 18px;">{hosp_name}</strong><br>
            <em>Address: {hosp_address}</em><br>
            Phone: <strong>{hosp_phone}</strong><br>
            Distance: <strong style="color: green;">{distance:.2f} km</strong><br>
            Direction: {direction}<br>
            <a href='https://www.openstreetmap.org/directions?mlat={hosp_lat}&mlon={hosp_lon}#map=17/{hosp_lat}/{hosp_lon}' target='_blank' style="color: blue; text-decoration: underline;">Get Directions</a>
        </div>
        """
        folium.Marker(
            [hosp_lat, hosp_lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="red", icon='hospital-o')
        ).add_to(marker_cluster)

    # Legend for the map
    legend_html = """
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 240px; height: auto; 
                background-color: rgba(255, 255, 255, 0.8); z-index:9999; border:2px solid grey; 
                font-size:14px; padding: 10px; border-radius: 5px;">
    <i class="fa fa-map-marker" style="color:blue"></i>&nbsp;You are here<br>
    <i class="fa fa-map-marker" style="color:red"></i>&nbsp;Nearby Hospitals<br>
    <p>Click on the markers for details!</p>
    </div>
    """
    user_map.get_root().html.add_child(folium.Element(legend_html))

    return user_map

def save_map(map_object, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    map_object.save(file_path)
    print(f"Map has been saved as '{file_path}'.")

# Main execution
latitude, longitude = get_user_location()
nearby_hospitals = find_nearby_hospitals(latitude, longitude)
hospital_map = create_map(latitude, longitude, nearby_hospitals)

# Corrected path
path = os.path.join("MedRecommendation", "project", "static", "hospitals.html")
save_map(hospital_map, path)
