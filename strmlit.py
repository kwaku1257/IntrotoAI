import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_filename = 'best_random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Function to scale the input data
def scale_data(input_data, scaler):
    data_df = pd.DataFrame(input_data, index=[0])
    scaled_data = scaler.transform(data_df)
    return pd.DataFrame(scaled_data, columns=data_df.columns)

# Load the scaler used during training
scaler = StandardScaler()
scaler.fit_transform(pd.DataFrame({
    'movement_reactions': [0], 'potential': [0], 'dribbling': [0], 'defending': [0], 'goalkeeping_skills': [0],
    'age': [0], 'attacking_crossing': [0], 'physic': [0], 'shooting_skills': [0], 'mentality': [0],
    'attacking_short_passing': [0], 'passing': [0], 'skills': [0], 'potential_x_movement_reactions': [0],
    'dribbling_x_defending': [0]
}))

# Streamlit app
st.title('Player Overall Rating Predictor')

st.header('Enter Player Attributes')

movement_reactions = st.number_input('Movement Reactions', min_value=0, max_value=100, value=50)
potential = st.number_input('Potential', min_value=0, max_value=100, value=50)
dribbling = st.number_input('Dribbling', min_value=0, max_value=100, value=50)
defending = st.number_input('Defending', min_value=0, max_value=100, value=50)
goalkeeping_skills = st.number_input('Goalkeeping Skills', min_value=0, max_value=100, value=50)
age = st.number_input('Age', min_value=15, max_value=45, value=25)
attacking_crossing = st.number_input('Attacking Crossing', min_value=0, max_value=100, value=50)
physic = st.number_input('Physic', min_value=0, max_value=100, value=50)
shooting_skills = st.number_input('Shooting Skills', min_value=0, max_value=100, value=50)
mentality = st.number_input('Mentality', min_value=0, max_value=100, value=50)
attacking_short_passing = st.number_input('Attacking Short Passing', min_value=0, max_value=100, value=50)
passing = st.number_input('Passing', min_value=0, max_value=100, value=50)
skills = st.number_input('Skills', min_value=0, max_value=100, value=50)

# Derived features
potential_x_movement_reactions = potential * movement_reactions
dribbling_x_defending = dribbling * defending

input_data = {
    'movement_reactions': movement_reactions, 'potential': potential, 'dribbling': dribbling, 'defending': defending,
    'goalkeeping_skills': goalkeeping_skills, 'age': age, 'attacking_crossing': attacking_crossing, 'physic': physic,
    'shooting_skills': shooting_skills, 'mentality': mentality, 'attacking_short_passing': attacking_short_passing,
    'passing': passing, 'skills': skills, 'potential_x_movement_reactions': potential_x_movement_reactions,
    'dribbling_x_defending': dribbling_x_defending
}

if st.button('Predict Overall Rating'):
    scaled_input_data = scale_data(input_data, scaler)
    prediction = loaded_model.predict(scaled_input_data)
    st.write(f'Predicted Overall Rating: {prediction[0]:.2f}')
