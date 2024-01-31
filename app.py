import streamlit as st
import pickle
import time

# load the model
model = pickle.load(open('twitter_sentiment.pkl', 'rb'))
second_model = pickle.load(open('fake_news_detect.pkl','rb'))

st.title('Sentiment Analysis On Fake news')

text = st.text_input('Enter your text')

submit = st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict([text])
    prediction_second = second_model.predict([text])
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

    sentiment_message = 'The sentiment is: ' + str(prediction[0])
    false_news_message = 'The news is predicted to be: ' + str(prediction_second[0])
     
    print(sentiment_message)
    print(false_news_message)
    st.write(sentiment_message) 
    st.write(false_news_message)
