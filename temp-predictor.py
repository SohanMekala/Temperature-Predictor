import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers, callbacks
from keras.callbacks import EarlyStopping

df = pd.read_csv('GlobalWeatherRepository.csv')
selectedCols = df[['temperature_fahrenheit', 'wind_mph', 'pressure_mb', 'humidity']]

df_train = selectedCols.sample(frac=0.7, random_state = 0)
df_valid = selectedCols.drop(df_train.index)

X_train = df_train.drop('temperature_fahrenheit', axis=1)
X_valid = df_valid.drop('temperature_fahrenheit', axis=1)
y_train = df_train['temperature_fahrenheit']
y_valid = df_valid['temperature_fahrenheit']

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu', input_shape=[3]),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mae',
)


#take weather data from frisco, tx and predict temperature
#https://weather.com/weather/today/l/33.15,-96.82?par=google

#current time/date 10:04 PM CDT Monday 9/18/2023
#actual temperature: 82 degrees farenheight

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=100,
    verbose = 0
)

wind_speed = 10
pressure = 1013
humidity_level = 33
X_single = np.array([[wind_speed, pressure, humidity_level]])

prediction = model.predict(X_single)
print(prediction[0][0])
#consitently about 7-8 degrees farenheight off