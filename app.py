import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the saved model, scaler, and training column structure
model = pickle.load(open('ipl_model.pkl', 'rb'))
scaler = pickle.load(open('ipl_scaler.pkl', 'rb'))
model_columns = pickle.load(open('ipl_columns.pkl', 'rb'))

# List of active IPL teams and host cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

cities = [
    'Hyderabad', 'Bengaluru', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'UAE', 'Rajkot', 'Kanpur', 'Mohali', 'Guwahati', 'Lucknow'
]

# 2. Configure the Streamlit web application title and user inputs
st.title('🏏 IPL Live Win Probability Predictor')
st.write('Enter live second-innings match details to predict real-time win probabilities.')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team (Chasing)', sorted(teams))
with col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

selected_city = st.selectbox('Match City', sorted(cities))
target = st.number_input('Target Runs', min_value=1, max_value=350, value=180)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, max_value=350, value=90)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=2)

# 3. Calculate match features and generate win probability on button click
if st.button('Predict Win Probability'):
    if batting_team == bowling_team:
        st.error('Batting and Bowling teams must be different!')
    else:
        # Feature engineering for live match state
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = (score * 6) / (120 - balls_left) if (120 - balls_left) > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Create a single-row DataFrame matching training features
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'target_runs': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Apply One-Hot Encoding and align columns with training data
        input_encoded = pd.get_dummies(input_df, columns=['batting_team', 'bowling_team', 'city'])
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Scale features and predict probabilities
        input_scaled = scaler.transform(input_encoded)
        result = model.predict_proba(input_scaled)

        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Display predicted probabilities
        st.subheader('Predicted Match Win Probability:')
        st.success(f"**{batting_team}:** {round(win_prob * 100, 2)}%")
        st.info(f"**{bowling_team}:** {round(loss_prob * 100, 2)}%")
