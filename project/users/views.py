from flask import render_template, url_for, flash, redirect, request, Blueprint, jsonify
from flask_login import login_user, current_user, logout_user, login_required
from project import db
from project.models import User, Data
from project.users.forms import RegistrationForm, LoginForm, UpdateUserForm
from project.users.picture_handler import add_profile_pic
import stripe
import re
import pandas as pd
import numpy as np
import os
import joblib as jb
import requests
import folium
from geopy.distance import geodesic
import math
import os
from folium.plugins import MarkerCluster



model = jb.load('D:\RnD, CCA\Parichay 2024\MedRecommendation\diseasepred.pkl')




from model.req import symptoms
from model.req import disease_dict
from model.req import description_dict
from model.req import medications_description

for i in range(len(symptoms)):
    if '_' in symptoms[i]:
        symptoms[i] = symptoms[i].replace('_', ' ')
        


def getMedicines(dic, prediction):
    df = pd.read_csv('MedRecommendation\model\medications.csv')
    for key, value in dic.items():
        if value == prediction[0]:
            for i in range(len(df)):
                row = df.iloc[i, :2]
                if row[0] == key:
                    return row[1]
    return "No medicine found"


def suggestedDiets(dic, prediction):
    df = pd.read_csv('MedRecommendation\model\diets.csv')
    for key, value in dic.items():
        if value == prediction[0]:
            for i in range(len(df)):
                row = df.iloc[i, :2]
                if row[0] == key:
                    return row[1]
                
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
# Main execution
latitude, longitude = get_user_location()
nearby_hospitals = find_nearby_hospitals(latitude, longitude)
hospital_map = create_map(latitude, longitude, nearby_hospitals)

# Load Stripe keys from environment variables
public_key = os.getenv('STRIPE_PUBLIC_KEY')
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

users = Blueprint('users', __name__)

# Register
@users.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('users.data'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data,
                    username=form.username.data,
                    password=form.password.data)

        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f"Account created successfully! You are now logged in as {user.username}", category='success')
        return redirect(url_for('users.data'))

    return render_template('register.html', form=form)

# Login
@users.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Log in Success!', category='success')

            next = request.args.get('next')
            if not next or not next.startswith('/'):
                next = url_for("users.data")

            return redirect(next)
        else:
            flash('Invalid email or password', category='danger')

    return render_template('login.html', form=form)

# Logout
@users.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("core.index"))

# Account (update UserForm)
@users.route('/account', methods=['GET', 'POST'])
def account():
    form = UpdateUserForm()
    if form.validate_on_submit():
        if form.picture.data:
            username = current_user.username
            pic = add_profile_pic(form.picture.data, username)
            current_user.profile_image = pic

        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('User Account Updated!', category='success')
        return redirect(url_for('users.account'))

    elif request.method == "GET":
        form.username.data = current_user.username
        form.email.data = current_user.email

    profile_image = url_for('static', filename='profile_pics/' + current_user.profile_image)
    return render_template('account.html', profile_image=profile_image, form=form)




        
@users.route('/data', methods=['GET'])
def data():
    # Corrected path
    path = os.path.join("MedRecommendation", "project", "static", "hospitals.html")
    save_map(hospital_map, path)
    return render_template('data.html')

@users.route('/predict', methods=['POST'])
def predict():
    disease_name = None
    description = None
    medicine = None
    diet = None
    descriptions = None

    # Get symptoms input from form
    paragraph = request.form.get('paragraph')
    print("Received symptoms:", paragraph)
    
    if paragraph:
        # Find symptoms mentioned in the input paragraph (ignoring case)
        found_symptoms = [symptom for symptom in symptoms if re.search(r'\b{}\b'.format(re.escape(symptom)), paragraph.lower())]

        # Create the symptom vector (1 if found, 0 if not)
        symptom_vector = [1 if symptom in found_symptoms else 0 for symptom in symptoms]
        symptom_vector = np.array(symptom_vector).reshape(1, -1)
        
        # Make prediction based on symptom vector
        prediction = model.predict(symptom_vector)
        predicted_disease_value = prediction[0]  # Get the predicted disease value
        
        # Get disease name from disease dictionary
        disease_name = [key for key, value in disease_dict.items() if value == predicted_disease_value]
        
        if disease_name:
            key = disease_name[0]  # The disease name (first found match)

            # Get the corresponding details
            description = description_dict.get(key, 'Disease description not found')
            diet = suggestedDiets(disease_dict, prediction)  # Suggested diet for the disease
            medicine = getMedicines(disease_dict, prediction)  # Get suggested medicine
            
            # Get medication descriptions
            medicine_key = str(medicine)  # Convert medicine to string as key
            descriptions = medications_description.get(medicine_key, "Medication description not found")
        
    else:
        print("Could not determine the disease from the input.")
    return render_template('prediction.html', disease_name=disease_name, description=description, medicine=medicine, diet=diet, descriptions=descriptions)


