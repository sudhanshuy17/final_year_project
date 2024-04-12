import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow import keras
import numpy as np
import streamlit as st
import folium
from streamlit_folium import folium_static
from weatherbit.api import Api
import pandas as pd

# Set the Streamlit page configuration
st.set_page_config(page_title="Cyclone Location Prediction", layout="wide")

# Load your machine learning model
model = tf.keras.models.load_model('DeepCNN_model.h5', compile=False)

# Define the Streamlit app
st.title("Cyclone Intensity With Future Location's + Weather Forecasting")

# Create a multiselect widget to navigate between pages
page = st.selectbox("Click below to nevigate", ["Find Cyclone Intensity", "Cyclone Location Prediction's", "Weather Forecast"])

# Page 1
if page == "Find Cyclone Intensity":
    # Create an upload button for the image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform inference on the image
        st.write("Predicting velocity...")

        # Preprocess the image: Resize to 201x201 and convert to a NumPy array
        image = image.resize((201, 201))

        # Convert the RGB image to 2 channels (e.g., red and green)
        image_array = np.array(image)[:, :, :2]

        # Make predictions using your model
        # Replace this with your own prediction logic
        velocity_prediction = model.predict(np.expand_dims(image_array, axis=0))[0][0]
         # Format the velocity to display only three decimal places
        formatted_velocity = "{:.3f}".format(velocity_prediction)

        # Divide the formatted velocity by 10
        divided_velocity = float(formatted_velocity) / 10
        velocity = int(divided_velocity)
        intensity_mapping = {
            (0, 38): "Tropical Depression",
            (39, 73): "Tropical Storm",
            (74, 95): "Typhoon",
            (96, 112): "Major Typhoon",
            (113, 136): "Super Typhoon",
        }

        # Classify the velocity into intensity categories
        intensity_category = "Unknown"
        for velocity_range, category in intensity_mapping.items():
            min_velocity, max_velocity = velocity_range
            if min_velocity <= float(divided_velocity) <= max_velocity:
                intensity_category = category
                break

        st.write(f"The Cyclone is approching with the velocity of : {velocity} KMPH")
        st.write(f"Predicted Intensity Category: {intensity_category}")


# Page 2
if page == "Cyclone Location Prediction's":
    # Create content for the second page
    lat = st.number_input('Enter the current Latitude of the Cyclone:')
    long = st.number_input('Enter the current Longitude of the Cyclone:')
    vmax = st.number_input('Enter the Velocity of the Cyclone:')
    mslp = st.number_input('Enter the Sea Level Pressure:')

    if __name__ == '__main__':
        model = keras.models.load_model('FinalLSTM.h5', compile=False)
        predictions = model.predict([[[lat, long, vmax, mslp], [lat, long, vmax, mslp], [lat, long, vmax, mslp]]])
        predicted_long, predicted_lat = predictions[0]
        st.write('Predicted Longitude:', predicted_long)
        st.write('Predicted Latitude:', predicted_lat)
        
        long = predicted_long
        lat = predicted_lat
        
        if st.button("Next Predictions"):
            for i in range(1, 10):
                model2 = keras.models.load_model('E:/cyclone/app/FinalLSTM.h5', compile=False)
                input_data = np.array([[[lat, long, vmax, mslp], [lat, long, vmax, mslp], [lat, long, vmax, mslp]]])
                predictions2 = model2.predict(input_data)
                
                predicted_long2, predicted_lat2 = predictions2[0]
                st.write(i, 'Predicted Longitude:', predicted_long2)
                st.write(i, 'Predicted Latitude:', predicted_lat2)
                long = predicted_long2
                lat = predicted_lat2
                
        if st.button("Mark Locations"):
            # Create a map centered at the predicted latitude and longitude
            m = folium.Map(location=[predicted_lat, predicted_long], zoom_start=5)
            
            for i in range(1, 10):
                model3 = keras.models.load_model('E:/cyclone/app/FinalLSTM.h5', compile=False)
                input_data = np.array([[[lat, long, vmax, mslp], [lat, long, vmax, mslp], [lat, long, vmax, mslp]]])
                predictions2 = model3.predict(input_data)
                
                predicted_lat2, predicted_long2 = predictions2[0]
                tooltip = f"Predicted Location {i}"
                folium.Marker([predicted_lat2, predicted_long2], popup=f"Predicted Location {i}", tooltip=tooltip).add_to(m) 
                lat = predicted_lat2
                long = predicted_long2
                
            # Render the map in Streamlit
            folium_static(m, width=1360)

#page 3
if page == "Weather Forecast":
    # weather application
    st.title('A Seven-Day Weather Forecast')
    # key = st.text_input('Enter key')
    key = '1e47672fdd544354b2a2a1ca6c7a1aa5'
    api = Api(key)


    @st.cache_resource
    def weather_forecast(city, state, country):
        api = Api(key)
        forecast = api.get_forecast(city=city, state=state, country=country)
        return forecast

    left, right = st.columns(2)
    city = left.text_input('Enter City')
    state = right.text_input('Enter City State')
    country = st.text_input('Enter Country')
    op = st.selectbox('Make a choice', ['DataFrame', 'BarChart', 'LineChart'])
    if st.button('Fetch'):
        forecast = weather_forecast(city, state, country)
        dates = []
        precips = []
        weather = []
        wind = []
        temp = []
        for day in forecast.get_series(['temp', 'precip', 'weather','wind_spd' ,'datetime']):
            dates.append(day['datetime'].date())
            temp.append(day['temp'])
            precips.append(day['precip'])
            weather.append(day['weather']['description'])
            wind.append(day['wind_spd'])
        df = pd.DataFrame({'Date': dates, 'Temp': temp, 'Precip': precips, 'Wind Speed': wind, 'Weather': weather})
        if op == 'DataFrame':
            st.dataframe(df)
        elif op == 'BarChart':
            st.bar_chart(df.Temp)
        else: 
            st.line_chart(df.Temp)