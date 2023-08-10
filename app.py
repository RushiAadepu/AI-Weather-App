import streamlit as st
import requests
import datetime
import joblib
from transformers import pipeline
from sklearn.tree import DecisionTreeClassifier
import transformers
import torch

# Load pre-trained GPT2 model
model_name = "gpt2"  # You can also try other models like "gpt2-medium", "gpt2-large", etc.
tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

# Set the seed for reproducibility
torch.manual_seed(42)

# Load the trained weather prediction model
loaded_model = joblib.load('model.joblib')

    # Display an image from a file path
image_path = 'weather-images.png'  
st.image(image_path, caption='Image Caption', use_column_width=True)  # Adjust caption and width settings
    
# Display an image from a URL
image_url = 'weather-images.jpg'
st.image(image_url, caption='Image Caption', use_column_width=True)


# Weather API key and endpoint
WEATHER_API_KEY = '377aacecc1592f1be075beb71ab22ea0'
WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'

st.markdown(
    f"""
    <style>
    body {{
        background-color: #87ceeb;
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
    prediction = loaded_model.predict(features)
    return prediction


def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


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

            features = [
                weather_data['temperature'],
                weather_data['humidity'],
                weather_data['wind_speed'],
                weather_data['pressure'],
                weather_data['cloudiness']
            ]
            predicted_condition = predict_weather([features])[0]
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
        user_input = st.text_input("You: ")
        if st.checkbox("Chat"):
            if user_input:
                response = generate_response(user_input)
                st.write(f"AI: {response}")


if __name__ == '__main__':
    main()
