import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Diabetes_dataset.csv')
df = data.copy()
df.head(3)

# def transformer(dataframe):
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     scaler = StandardScaler()
#     encoder = LabelEncoder()

#     for i in dataframe.columns:
#         if dataframe[i].dtypes != 'O':
#             dataframe[i] = scaler.fit_transform(dataframe[[i]])
#         else:
#             dataframe[i] = encoder.fit_transform(dataframe[i])
#     return dataframe


# df = transformer(df.drop('Diabetes', axis = 1))
# df.head()

# from imblearn.over_sampling import SMOTE
# import pandas as pd
# import seaborn as sns

# # Assuming you have a Datadf df with features (x) and target variable (y)
# # Replace 'Diabetes' with the actual name of your target variable

# # Extract features (x) and target variable (y)
# y = data.Diabetes
# x = df

# # Initialize SMOTE
# smote = SMOTE(sampling_strategy='auto',  random_state=42)  # You can adjust the sampling_strategy as needed

# # Apply SMOTE to generate synthetic samples
# x_resampled, y_resampled = smote.fit_resample(x, y)

# # Create a new Datadf with the resampled data
# ds = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), pd.Series(y_resampled, name='Diabetes')], axis=1)

# # Plot the count of samples for each class in the resampled data
# sns.countplot(x=ds['Diabetes'])


# # - split into train and test
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# y = ds.Diabetes
# x = ds.drop('Diabetes', axis =1)


# x_train, x_test, y_train, y_test = train_test_split(ds, y, test_size = 0.20, random_state = 79, stratify = y)
# print(f'x_train: {x_train.shape}')
# print(f'x_test: {x_test.shape}')
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))

# model = tf.keras.Sequential([ #........................ Instantiate the model creating class.
#     tf.keras.layers.Dense(units=8, activation='relu'), #... Input layer of 12 features
#     tf.keras.layers.Dense(20, activation='relu'), #.... Add the second 20 layer, and instantiate the activation to be used.
#     tf.keras.layers.Dense(15, activation='relu'), #..... Add the third layer.
#     tf.keras.layers.Dense(10, activation='relu'), #..... Add the third layer.
#     tf.keras.layers.Dense(5, activation='relu'), #..... Add the third layer.
#     tf.keras.layers.Dense(1, activation='sigmoid') #... Add the last output layer
# ])
# model.compile(optimizer='adam', # ..................... The optimizer that adjusts weight and bias for a given neuron
#               loss = 'binary_crossentropy', #...... Loss calculates the error of the prediction
#               metrics=['accuracy']) #.................. Accuracy calculates the precision of the prediction.

# model.fit(x_train, y_train, epochs=20) #..... Fit the model on the dataset and define the number of epochs


# y_pred = model.predict(x_test)
# y_pred = (y_pred > 0.5).astype(int)  #............................................... set a 50% confidence level that the customer doesnt stop buying
# outcome = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
# outcome

# y_pred = model.predict(x_test)
# y_pred = (y_pred > 0.5).astype(int)  #............................................... set a 50% confidence level that the customer doesnt stop buying
# outcome = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
# outcome

import streamlit as st
import pickle
model = pickle.load(open('Model.pkl', "rb"))

st.sidebar.image('pngwing.com (12).png', width = 300,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Diabetes Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>Developing an Accurate Diabetes Prediction Model Using Machine Learning to Enhance Early Detection and Improve Patient Outcomes.</h6>",unsafe_allow_html=True)
    st.image('main.png',  width = 650)

    # Background story
    st.markdown("<h3 style = 'margin: -15px; color: #2B2A4C; text-align: left; font-family:montserrat'>Background to the story</h3>",unsafe_allow_html=True)
    st.markdown("<p>This project is personal to me because my grandpa had diabetes for a long time. I want to create a computer program that can tell if someone might get diabetes in the future. By using information about a person's health, the program will try to help them know if they need to be careful. I'm doing this so that others don't have to go through what my grandpa did. Let's work together to use technology to help people stay healthy and avoid diabetes.</p>", unsafe_allow_html = True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>The Need for Precision</h3>", unsafe_allow_html=True)
    st.markdown("<p>Imagine a world where waiting for a package is a thing of the past. In this era of instant gratification, customers expect not just timely deliveries but the exact moment their package will arrive...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Navigating Logistics Challenges</h3>", unsafe_allow_html=True)
    st.markdown("<p>The logistics landscape is riddled with challenges — traffic variations, unpredictable weather conditions, and dynamic delivery routes. Conventional delivery time estimates often fall short in accounting for these factors, resulting in frustrated customers and operational inefficiencies...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Machine Learning as the Navigator</h3>", unsafe_allow_html=True)
    st.markdown("<p>At the heart of our initiative is machine learning, a powerful tool that learns and adapts from historical data. By analyzing patterns in delivery times, considering external variables, and continuously learning from each delivery experience, our model becomes a navigator...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Benefits Beyond Accuracy</h3>", unsafe_allow_html=True)
    st.markdown("<p>Beyond the immediate benefits of precise delivery time estimates, our model aims to optimize delivery routes, reduce operational costs, and enhance overall customer satisfaction...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: left; font-family:montserrat'>The Future of Delivery</h3>", unsafe_allow_html=True)
    st.markdown("<p>As we delve into this project, we envision a future where waiting for a delivery is a thing of the past, replaced by a seamless experience where customers know exactly when their package will arrive. Our time of delivery prediction model is not just a tool; it's a glimpse into the future of logistics...</p>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: LEFT;'>Join us on this exciting venture as we redefine the logistics landscape, one accurate delivery prediction at a time.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by Datapsalm</p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(df.head())
    # st.sidebar.image('pngwing.com (13).png', width = 300,  caption = 'customer and deliver agent info')


if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()



if selected_page == "Modeling":
    st.sidebar.markdown("Add your modeling content here")
    Gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
    Age = st.sidebar.number_input("Age", 0,1000)
    Hypertension = st.sidebar.selectbox("Hypertension", df['Hypertension'].unique())
    Heart_disease = st.sidebar.selectbox("Heart_disease", df['Heart_disease'].unique())
    Smoking_history = st.sidebar.selectbox('Smoking_history', df['Smoking_history'].unique())
    Body_Mass_Index = st.sidebar.number_input("Body_Mass_Index", 0,1000)      
    Hemoglobin_A1c = st.sidebar.number_input("Hemoglobin_A1c", 0,1000)
    Blood_glucose_level = st.sidebar.number_input("Blood_glucose_level",0,1000)
    st.sidebar.markdown('<br>', unsafe_allow_html= True)


    input_variables = pd.DataFrame([{
        'Gender':Gender,
        'Age': Age,
        'Hypertension': Hypertension,
        'Heart_disease': Heart_disease,
        'Smoking_history': Smoking_history, 
        'Body_Mass_Index': Body_Mass_Index, 
        'Hemoglobin_A1c': Hemoglobin_A1c,
        'Blood_glucose_level':Blood_glucose_level 
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #2B2A4C;'>Inputed Variables</h2>", unsafe_allow_html=True)
    st.write(input_variables)
    # st.write(input_variables)
    cat = input_variables.select_dtypes(include = ['object', 'category'])
    num = input_variables.select_dtypes(include = 'number')

    # Standard Scale the Input Variable.
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    for i in input_variables.columns:
        if i in num.columns:
            input_variables[i] = StandardScaler().fit_transform(input_variables[[i]])
    for i in input_variables.columns:
        if i in cat.columns: 
            input_variables[i] = LabelEncoder().fit_transform(input_variables[i])

    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Press To Predict'):
        st.markdown("<h4 style = 'color: #2B2A4C; text-align: left; font-family: montserrat '>Model Report</h4>", unsafe_allow_html = True)
        predicted = model.predict(input_variables)
        st.toast('bank_account Predicted')
        st.image('check icon.png', width = 100)
        st.success(f'Model Predicted {predicted}')
        if predicted == 0:
            st.success('The person is not likely to have Diabetes')
        else:
            st.success('The person is likely to have Diabetes')


    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>FINANCIAL INCLUSION BUILT BY DATAPSALM</h8>",unsafe_allow_html=True)


    
