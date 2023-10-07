import streamlit as st
import requests
import datetime
import joblib

# Weather API key and endpoint
WEATHER_API_KEY = '377aacecc1592f1be075beb71ab22ea0'
WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'

# RapidAPI key for ChatGPT
RAPIDAPI_KEY = '3965b429bfmsh160e49419126035p1d84efjsn4c96f585201e'  # Replace with your RapidAPI key

st.markdown(
    f"""
    <style>
    body {{
        font-family: Arial, sans-serif;
    }}
    /* Common styles */
    .container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    }}

    /* Weather-specific styles */
    .normal-weather {{
        background-color: #87ceeb; /* Light blue for normal weather */
        color: #333; /* Dark text color for visibility */
    }}
    .cloudy-weather {{
        background-color: #ddd; /* Gray for cloudy weather */
        color: #333;
    }}
    .sunny-weather {{
        background-color: #ffdb58; /* Yellow for sunny weather */
        color: #333;
    }}
    .partly-cloudy-weather {{
        background-color: #87ceeb; /* Light blue for partly cloudy */
        color: #333;
    }}
    .hot-weather {{
        background-color: #ff5733; /* Orange for hot weather */
        color: #fff; /* White text for visibility */
    }}
    .rainy-weather {{
        background-color: #4682b4; /* Steel blue for rainy weather */
        color: #fff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


def get_weather(location):
    params = {
        'q': location,
        'appid': WEATHER_API_KEY,
        'units': 'metric'  # Request temperature in Celsius
    }

    response = requests.get(WEATHER_API_URL, params=params)
    response_json = response.json()

    if 'main' in response_json and 'weather' in response_json:
        weather_info = {
            'temperature': response_json['main']['temp'],
            'humidity': response_json['main']['humidity'],
            'wind_speed': response_json['wind']['speed'],
            'pressure': response_json['main']['pressure'],
            'sunrise_timestamp': response_json['sys']['sunrise'],
            'sunset_timestamp': response_json['sys']['sunset'],
            'cloudiness': response_json['clouds']['all']
        }
        if 'rain' in response_json:
            weather_info['precipitation'] = response_json['rain']
        else:
            weather_info['precipitation'] = 'N/A'
        if 'uv' in response_json:
            weather_info['uv_index'] = response_json['uv']
        else:
            weather_info['uv_index'] = 'N/A'

        # Convert Unix timestamps to datetime objects
        weather_info['sunrise_time'] = datetime.datetime.fromtimestamp(weather_info['sunrise_timestamp']).strftime(
            '%Y-%m-%d %H:%M:%S')
        weather_info['sunset_time'] = datetime.datetime.fromtimestamp(weather_info['sunset_timestamp']).strftime(
            '%Y-%m-%d %H:%M:%S')

        return weather_info

def predict_weather(features):
    # Reshape the 1D features list into a 2D array
    features_2d = [features]

    loaded_model = joblib.load('model.joblib')
    prediction = loaded_model.predict(features_2d)
    return prediction[0]

def generate_response(prompt):
    url = "https://open-ai21.p.rapidapi.com/conversationgpt"
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "3965b429bfmsh160e49419126035p1d84efjsn4c96f585201e",  # Use your RapidAPI key
        "X-RapidAPI-Host": "open-ai21.p.rapidapi.com"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            response_data = response.json()
            return response_data
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return f"Failed to get a response from the ChatGPT API. Status code: {response.status_code}"

def main():
    st.title('AI Weather and Chatbot App')

    location = st.text_input('Enter location:')

    if st.button('Get Weather'):
        weather_data = get_weather(location)

        if weather_data is not None:

            st.write(f"Temperature: {weather_data.get('temperature', 'N/A')}°C")
            st.write(f"Humidity: {weather_data.get('humidity', 'N/A')}%")
            st.write(f"Wind Speed: {weather_data['wind_speed']} km/h")
            st.write(f"Pressure: {weather_data['pressure']} hPa")
            st.write(f"Cloudiness: {weather_data['cloudiness']}%")
            st.write(f"Sunrise Time: {weather_data['sunrise_time']}")
            st.write(f"Sunset Time: {weather_data['sunset_time']}")

            image_path = 'weather-images.jpg'
            st.image(image_path, caption='Types of Weather', use_column_width=True)

            features = [
                weather_data['temperature'],
                weather_data['humidity'],
                weather_data['wind_speed'],
                weather_data['pressure'],
                weather_data['cloudiness']
            ]
            predicted_condition = predict_weather(features)

            st.markdown(f"<h2>Today's Predicted Weather Condition: {predicted_condition}</h2>", unsafe_allow_html=True)
            if predicted_condition == 'normal':
                st.markdown(
                    "<p style='font-size: 18px;'>It seems like a normal day. You can go about your usual activities.</p>",
                    unsafe_allow_html=True)
            elif predicted_condition == 'cloudy':
                st.markdown(
                    "<p style='font-size: 18px;'>The sky is cloudy today. It might be a good idea to carry an umbrella.</p>",
                    unsafe_allow_html=True)
            elif predicted_condition == 'sunny':
                st.markdown(
                    "<p style='font-size: 18px;'>It's a sunny day! Don't forget to wear sunscreen and stay hydrated.</p>",
                    unsafe_allow_html=True)
            elif predicted_condition == 'partly cloudy':
                st.markdown(
                    "<p style='font-size: 18px;'>The weather is partly cloudy. Enjoy your day with a mix of sun and clouds.</p>",
                    unsafe_allow_html=True)
            elif predicted_condition == 'hot':
                st.markdown("<p style='font-size: 18px;'>It's going to be a hot day. Stay cool and stay hydrated.</p>",
                            unsafe_allow_html=True)
            elif predicted_condition == 'rainy':
                st.markdown("<p style='font-size: 18px;'>Expect rain today. Don't forget your umbrella!</p>",
                            unsafe_allow_html=True)
            else:
                st.write("The weather condition is uncertain. Stay prepared for any changes.")
        else:
            st.warning(
                "Oops! The weather information for the provided location is not available. Please make sure the location is valid and try again.")
            return None

    st.subheader('Chat with AI:')
    st.markdown(
        "Feel free to ask any weather-related questions.")
    user_input = st.text_input("You: ")
    if st.checkbox("Chat"):
        if user_input:
            response = generate_response(user_input)
            assistant_reply = response.get('GPT', 'N/A')  # Extract GPT response content
            st.write(f"AI: {assistant_reply}")

if __name__ == '__main__':
    main()
