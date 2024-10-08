# The City College of New York, City University of New York
# Kelin Mav
# August, 2020
# Simplified in 2024
# NLP: Predicting financial stock movement based on tweets from Pres. Trump
import re
import datetime 
from datetime import date
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# Function to remove punctuation and web links from tweets
# @\S+|https?://\S+ - matches either a substring which starts with @ 
# and contains non-whitespace characters \S+ OR a link(url) which 
# starts with http(s)://

def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', 
                ' ', tweet).split())

def str_to_date(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def print_weights(weights):
    # weights = model.get_weights();
    print('\n******* WEIGHTS OF ANN *******\n') 
    for i in range(int(len(weights)/2)):
        print('Weights W%d:\n' %(i), weights[i*2])
        print('Bias b%d:\n' %(i), weights[(i*2)+1])
        
def normalize_column(dataframe, col_name):
    maximum = max(dataframe[col_name])
    minimum = min(dataframe[col_name])
    if maximum != minimum:
        dataframe[col_name] = (dataframe[col_name] - minimum)/(maximum-minimum)
    return dataframe[col_name]

# Load in file, remove punctuation and weblinks from all tweets and then 
#   perform sentiment analysis.  
print('\n\n********** CLEANING TWEETS **********\n\n')
filename = "trumptwitter_2019_01_01_to_2020_03_11.csv"
df = pd.read_csv(filename)
df["Clean Tweet"] = df['text'].apply(lambda x: clean_tweet(x).lower())
df['created_at'] = pd.to_datetime(df['created_at']).dt.date

# Data cleaning and processing  
# Check if the tweet contains any keywords - remove all tweets that don't  
print('\n\n********** IDENTIFYING KEY WORDS **********\n\n')
key_words = ['Europe', 'China', 'tariff', 'Stock Market', 'economy', 'bank', 
         'trade', 'jobs', 'money', 'dollar','currency','Xi','deal','growth']

df['noof_keywords'] = np.where(df.text.str.contains('|'.join(key_words)),1,0)

for key_word in key_words:
    df[key_word] = np.where(df.text.str.contains(key_word), 1, 0)

columns_to_aggregate = {'noof_keywords':'sum'}
for key_word in key_words:
    columns_to_aggregate[key_word] = 'sum'
grouped_df=df.groupby(by='created_at').agg(columns_to_aggregate).reset_index()
grouped_df.set_index('created_at', inplace=True)
#% Stock prices
filename = "S&P500_2019_01_01_to_2020_03_11.csv"
snp_df = pd.read_csv(filename)
snp_df['date'] = pd.to_datetime(snp_df['Date']).dt.date
snp_df.set_index('date', inplace=True)
# return at close each day
snp_df['percent change'] = snp_df['Close'].pct_change() 
grouped_df = grouped_df.join(snp_df, how='inner')
features = [ 'Open', 'Volume','noof_keywords',] + key_words
features = features + ['percent change']
grouped_df = grouped_df[features]
grouped_df = grouped_df.dropna()

# ANN 
# before putting any data into NN, we normalize the data (between 0 and 1)
# Normailize only the open and volume traded (beacuse they are large compared
# to the other input values)
grouped_df['day_opening'] = normalize_column(grouped_df, 'Open')
grouped_df['units_traded'] = normalize_column(grouped_df,'Volume')

# Prepare input and output features for ANN
features = ['day_opening', 'units_traded', 'noof_keywords'] + key_words
x = np.array(grouped_df[features])
y = np.array(grouped_df['percent change'])

#ANN model
neural_net = tf.keras.Sequential()

# NN inputs: how many times each keyword was mentioned by President, 
# opening price of the stock and volume traded
neural_net.add(tf.keras.layers.Dense(10 , activation='relu', 
                                     input_dim=x.shape[1]))

## add another hidden layer with 4 neurons to the NN
neural_net.add(tf.keras.layers.Dense(4, activation='relu'))

## add another hidden layer with 8 neurons to the NN
neural_net.add(tf.keras.layers.Dense(8, activation='relu'))

## add an output layer with a single output (percent change)
neural_net.add(tf.keras.layers.Dense(1, activation='linear'))

neural_net.compile(optimizer='adam', loss='mean_absolute_error')

## train the ANN model using 1200 iterations
print('\n\n********** Begin ANN training **********\n\n')
neural_net.fit(x, y, epochs=1200)
weights = neural_net.get_weights()
print_weights(weights)
print('\n\n********** ANN training complete **********\n\n')


# Test on the first 10 days
days_to_test = 10
noof_correct_movement = 0
diffs = []
noof_predictions = 0

print('\n\n********** ANN PREDICTIONS FOR LAST 10 DAYS **********\n\n')
for i in range(1, days_to_test + 1):
    input_features = []
    noof_predictions += 1
    # Actual change from S&P data for the i-th day from the end
    actual_change = y[i]
    # Predict the change using the neural network
    predicted_change = neural_net.predict(np.array([x[i]]))[0, 0]
    # Print predicted vs actual changes
    print(f"The predicted change for {grouped_df.index[0]} was: {predicted_change:.5f}")
    print(f"Actual change for {grouped_df.index[0]} was: {round(actual_change, 4)}\n")
    
    # Compare prediction with actual movement direction
    if predicted_change * actual_change > 0:
        diffs.append(abs(predicted_change - actual_change))
        noof_correct_movement += 1
# Calculate accuracy and average error
percent_correct = (noof_correct_movement / noof_predictions) * 100
average_diff = sum(diffs) / len(diffs) if diffs else 0

print(f"ANN was correct in predicting the movement {percent_correct:.2f}% of the time in {noof_predictions} predictions.")
print(f"The average error of the correct predictions was {average_diff * 100.0:.1f}%")
# Save results to file
with open('output.txt', 'a') as output_file:
    output_file.write(f'{date.today()} percentage of correct predictions: {round(percent_correct, 1)}% with error {average_diff * 100.0:.1f}%\n')

grouped_df.to_csv('TrumpTweet_Input.csv')