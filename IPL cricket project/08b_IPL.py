import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("IPL_model.pkl", "rb") as file:
    model = pickle.load(file)

# title
st.title("IPL Match Outcome Predictor")

# Team and City Lists
selected_team = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bengaluru', 'Sunrisers Hyderabad']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# User Inputs
batting_team = st.selectbox("Select Batting Team", selected_team)
bowling_team = st.selectbox("Select Bowling Team", selected_team)
city = st.selectbox("Select City", cities)
target_runs = st.number_input("Target Runs", min_value=50, max_value=300, step=1)
balls_left = st.number_input("Balls Left", min_value=0, max_value=120, step=1)
wickets_left = st.slider("Wickets Left", min_value=0, max_value=10, step=1)
total_runs = st.number_input("Total Runs Scored", min_value=0, max_value=300, step=1)

# Calculate Required Run Rate (RRR)
if balls_left > 0:
    rrr = (target_runs - total_runs) / (balls_left / 6)
else:
    rrr = 0

# Encode Teams and Cities
teams = {team: idx for idx, team in enumerate(selected_team)}
cities_encoding = {city: idx for idx, city in enumerate(cities)}

if batting_team in teams and bowling_team in teams and city in cities_encoding:
    batting_team_encoded = teams[batting_team]
    bowling_team_encoded = teams[bowling_team]
    city_encoded = cities_encoding[city]
else:
    st.error("Invalid Selection")
    st.stop()

# Create input array
input_data = np.array([[batting_team_encoded, bowling_team_encoded, city_encoded, target_runs, balls_left, wickets_left, total_runs, rrr]])

# Prediction
if st.button("Predict Outcome"):
    prediction_prob = model.predict_proba(input_data)[0]
    batting_team_win_prob = prediction_prob[1] * 100
    bowling_team_win_prob = prediction_prob[0] * 100
    
    st.success(f"{batting_team} Win Probability: {batting_team_win_prob:.2f}%")
    st.success(f"{bowling_team} Win Probability: {bowling_team_win_prob:.2f}%")
