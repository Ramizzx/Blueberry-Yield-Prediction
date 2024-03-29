import streamlit as st
import joblib
import sklearn
import pandas

rf = joblib.load('randomforest.joblib')

col = ['Row#', 'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
       'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
       'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
       'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds']

def prediction_yield(row, clonesize, honeybee, bumbles, osmia,maxutrange, minutrange,avgutrange,maxltrange,minltrange,avgltrange,rainingdays,avgrainingdays,fruitset,fruitmass,seeds):

    prediction = rf.predict([[row, clonesize, honeybee, bumbles, osmia,maxutrange, minutrange,avgutrange,maxltrange,minltrange,avgltrange,rainingdays,avgrainingdays,fruitset,fruitmass,seeds]])
    print('prediction done')
    return prediction


def main():

    st.header('Blueberry Yield Prediction App')

    row = st.text_input('Row','Row# Value')
    clonesize = st.text_input('clonesize','clonesize value')
    honeybee = st.text_input('honeybee','bumbles_value')
    bumbles = st.text_input('andrena','andrena values')
    osmia = st.text_input('osmia','osmia')
    maxutrange = st.text_input('maxutrange','maxutrange')
    minutrange = st.text_input('minutrange','minutrange')
    avgutrange = st.text_input('avgutrange','avgutrange')
    maxltrange = st.text_input('maxltrange','maxltrange')
    minltrange = st.text_input('minltrange','minltrange')
    avgltrange = st.text_input('avgltrange','avgltrange')
    rainingdays = st.text_input('rainingdays','rainingdays')
    avgrainingdays= st.text_input('avgrainingdays','avgrainingdays')
    fruitset = st.text_input('fruitset','fruitset')
    fruitmass = st.text_input('fruitmass','fruitmass')
    seeds = st.text_input('seeds','seeds')
    pred=""


    if st.button('Predict Yield'):
        pred = prediction_yield(row,clonesize, honeybee, bumbles, osmia,maxutrange, minutrange,avgutrange,maxltrange,minltrange,avgltrange,rainingdays,avgrainingdays,fruitset,fruitmass,seeds)

        st.success('The yield is {}'.format(pred))

if __name__ == '__main__':

    main()