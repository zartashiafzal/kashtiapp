# Name = Zartashia Afzal      # ML chilla

import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# making container (parts on website)
header=st.container()
data_sets=st.container()
features=st.container()
model_training=st.container()


with header:
    st.title("Kashti ki app")
    st.text("In this project we are working on Titanic data")


with data_sets:
    st.header("Titanic")
    st.text("In this project we are working on Titanic data_set")
    # importing data
    df=sns.load_dataset("titanic")
    df=df.dropna()
    st.write(df.head(10))
    st.subheader("wrt gender")
    st.bar_chart(df['sex'].value_counts())
    st.subheader("class k hisab say faraq")
    #other plot
    st.bar_chart(df['class'].value_counts())
    st.bar_chart(df['age'].sample(10)) # or head(10)


with features:
    st.header("These are our app features")
    st.text("In this project we are working on features")
    st.markdown("feature: ** will tell us ...**")


with model_training:
    st.header("kya bna?")
    st.text("In this project we are working on Titanic model training, is mein hum parameters ko kam ya zayada kraingay")

    # Columns making (2)

    input, display= st.columns(2)

    # pehlay column mein selection points lanay k liyay
    max_depth= input.slider("How many people do u know?", min_value=10, max_value=100, value=20, step=5)


#n-estimators
n_estimators= input.selectbox("how many trees should b there in random forest?", options=[50, 100, 200, 300, "No limit"])

# adding list of features
input.write(df.columns)

# input features from user
input_features= input. text_input("Which input feature should we use?")

# ML model

model= RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# we will add condition so that system could what to do if option of "no limit" is selected
if n_estimators == "No limit":
    model= RandomForestRegressor(max_depth=max_depth) # if no limit then apply random forest regressor  without specified number of trees
else:
    model= RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# for fitting model define x & y
x=df[[input_features]]
y=df[['fare']]  # 2d array when we are y=using for prediction

# fitting
model.fit(x,y)
pred=model.predict(y)

# Display model metrices
display.subheader("Mean absolute error of model is ...")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean squared error of model is ...")
display.write(mean_squared_error(y, pred))
display.subheader("R square score of model is ...")
display.write(r2_score(y, pred))