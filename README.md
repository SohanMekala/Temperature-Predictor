# Temperature-Predictor
Basic regression model built with tensorflow and keras

GlobalWeatherRepository.csv contains data regarding the weather for many locations in the world as of 09/18/2023
temp-predictor.py is a python script that uses tensorflow and keras to make a basic linear regression model that can accurately predict temperature

the model takes in wind(mph), pressure(mb), and humidity(%) to predict the temperature(°f)

this dataset is from this [link](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) and is updated daily. Huge thanks to the creator of this dataset

this program is obviously not perfect because it takes in training data from only one point throughout the entire year
