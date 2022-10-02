
import streamlit as st
import random


st.title('This is a Tweeter Outrage Radar.')
st.text('Gotta emotionally discard daily Internet bull****')
    
tweet = st.text_input('Now please gently input the tweet:')

whenTweetBad = ["this is unsafe", "not recommended", "risky, buddy", "eyebleach please", "this is toxic","NSFW", "emotional Damage", ":(","why do we exist, just to suffer"]
whenTweetGoodOrNeutral = ["wholesome", "approved", "certified", "10/10", "safe"]



if tweet == "lil shithead, ur banned":
    st.text("bad,")
    st.text(random.choice(whenTweetBad))

pad = st.text('lol')
