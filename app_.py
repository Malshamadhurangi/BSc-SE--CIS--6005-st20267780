import pandas as pd
import streamlit as st
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model
xgb_model = joblib.load('C:\\Users\\acer\\OneDrive\\Desktop\\Horses\\xgboost_model_new.pkl')

# Load label encoders
with open('C:\\Users\\acer\\OneDrive\\Desktop\\Horses\\label_encoders_new.pkl', 'rb') as f:
    label_encoders = joblib.load(f)


# Remove 'outcome' label encoder
if 'outcome' in label_encoders:
    del label_encoders['outcome']
    
    
# Function to preprocess input data
def preprocess_input(data):
    # Perform label encoding for categorical variables
    for column, encoder in label_encoders.items():
        print(column, encoder)
        data[column] = encoder.transform(data[column])
    return data

# Function to make predictions
def predict_outcome(model, data):
    preprocessed_data = preprocess_input(data)
    prediction = model.predict(preprocessed_data)
    return prediction

def convert_int_to_object(data, columns):
    for column in columns:
        if data[column].dtype == 'int64':
            data[column] = data[column].astype('object')

def reorder_columns(input_data):
    # Ensure columns are in the same order as during training
    
    expected_columns = [
    'surgery', 'age', 'rectal_temp', 'pulse', 'respiratory_rate',
    'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane',
    'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention',
    'nasogastric_tube', 'nasogastric_reflux', 'nasogastric_reflux_ph',
    'rectal_exam_feces', 'abdomen', 'packed_cell_volume', 'total_protein',
    'abdomo_appearance', 'abdomo_protein', 'surgical_lesion', 'lesion_1',
    'lesion_2', 'lesion_3', 'cp_data']
    
    return input_data.reindex(columns=expected_columns)


# Convert numerical prediction to text
def get_outcome_text(prediction_value):
    outcome_mapping = {0: 'Died', 1: 'Euthanized', 2: 'Lived'}
    return outcome_mapping.get(prediction_value, 'Unknown')

# Streamlit web app
def main():
    
    titleside = """
    <style>
    [data-testid="stMarkdownContainer"] {
        text-align: center;
        
        
    }
    
    </style>
    """
    
    st.markdown(titleside, unsafe_allow_html=True)
    st.sidebar.title('Horse Health Prediction')



    buttonside = """
    <style>
    [data-testid="stButton"]{
        padding-left: 100px;
    }
    </style>
    """
    
    st.markdown(buttonside, unsafe_allow_html=True)
   



   
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
        background-image: url("https://wallpapers.com/images/featured/horse-h3azzzaaorg8c9ay.jpg");
        background-size: cover;}
    
    
    </style>
    
    
    
    """    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
  
    headerbg = """
    <style>
    [data-testid="stHeader"]{
        background-image: url("https://wallpapers.com/images/featured/horse-h3azzzaaorg8c9ay.jpg");
        background-size: cover;}
    
    
    </style>
    
    
    
    """    
    st.markdown(headerbg, unsafe_allow_html=True)
    
    
    container = """
    <style>
    [data-testid="stAppViewBlockContainer"]{
 
        padding: 0rem 1rem 10rem;
    
    </style>
    
    
    
    """    
    st.markdown(container, unsafe_allow_html=True)
    
    sidet = """
    <style>
    [data-testid="stSidebarUserContent"]{
        font-size: 2px;
        
        
    }
    </style>
    """
    st.markdown(sidet, unsafe_allow_html=True)


    # Load DataFrame
    df = pd.read_csv('C:\\Users\\acer\\OneDrive\\Desktop\\Horses\\train.csv')
    
    df.dropna(inplace = True)
    
    #lesion_columns = ['lesion_1', 'lesion_2', 'lesion_3']

    # Remove unnecessary columns
    df = df.drop(['id', 'hospital_number','outcome'], axis=1)
    
    # Applying the function to traindata
    #convert_int_to_object(df, lesion_columns)

    # Create dropdowns for categorical variables
    categorical_vars = df.select_dtypes(include=['object']).columns
    categorical_values = {col: df[col].unique().tolist() for col in categorical_vars}
    
    
    
    
    
    # Split categorical values into two halves
    half_len = len(categorical_values) // 2
    categorical_values_left = dict(list(categorical_values.items())[:half_len])
    categorical_values_right = dict(list(categorical_values.items())[half_len:])
    
    
    user_input = {} 
    left_column, right_column = st.columns(2)
    
    with left_column:
        
        for feature, values in categorical_values_left.items():
            user_input[feature] = st.selectbox(feature.capitalize(), values)


    with right_column:
        
        for feature, values in categorical_values_right.items():
            user_input[feature] = st.selectbox(feature.capitalize(), values)
            
            
    # Add numerical input fields
    numerical_vars = df.select_dtypes(include=['float64','int64']).columns
    for feature in numerical_vars:
        user_input[feature] = st.number_input(feature.capitalize(), min_value=0.0)
        
    st.sidebar.markdown("""
        <style>
            .centered-button {
                display: block;
                margin: 0 auto;
                width: 5%; /* Adjust the width as needed */
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    
    
    
    if st.sidebar.button('Predict', key='predict_button', help='Click to predict',):
        # Convert user input to DataFrame
        input_data = pd.DataFrame([user_input])

        # Reorder columns
        input_data = reorder_columns(input_data)

        # Make prediction
        prediction = predict_outcome(xgb_model, input_data)


        # Convert numerical prediction to text
        predicted_outcome = get_outcome_text(prediction[0])
        # Style the predicted outcome text
        # Define colors based on predicted outcome
        outcome_colors = {'Died': 'red', 'Euthanized': 'orange', 'Lived': 'green'}
        outcome_color = outcome_colors.get(predicted_outcome, 'black')
    
        # Style the predicted outcome text with color
        predicted_outcome_html = f"<div style='position: absolute; top: 150px; left: 50%; transform: translateX(-50%); text-align:center; background-color: #D3D3D3; padding: 30px; border-radius: 5px; width: 300px;'><h1 style='color:#262730; font-size:20px;'>Predicted Outcome:</h1><h1 style='color:{outcome_color}; font-size:50px;'>{predicted_outcome}</h1></div>"
    
        # Display styled predicted outcome
        st.sidebar.markdown(predicted_outcome_html, unsafe_allow_html=True)
        
        st.sidebar.write('XGboost Model with 76% accuracy!')
        
      

if __name__ == '__main__':
    main()




