from random import Random
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


st.write("""
# :rocket:Predicting Data on Diabetes Patients 

"""
)
#Displaying an image
image = Image.open('diabetic_patient.jpg')

st.image(image,caption='diabetic patient')

# if st.checkbox('show dataset'):
#     st.table(ds)

#     ds

# st.line_chart(ds)

st.write("""
## Sample Data on which the patient are tested : 
""")
df = pd.read_csv('diabetes.csv')

show_ds = df.iloc[:5,:-1]
st.table(show_ds)

st.write('## showing statistics and chart :')

st.write(show_ds.describe())

st.write('## showing bar chart :')

chart = st.bar_chart(df)

#spliting the dataset into train and test

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.25, random_state=0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies' , 0 , 17 , 3)
    Glucose = st.sidebar.slider('glucose' , 0 , 200 , 30)
    BloodPressure = st.sidebar.slider('bloodpressure' , 75 , 123 , 50)
    SkinThickness = st.sidebar.slider('skinthickness' , 75 , 123 , 30)
    Insulin = st.sidebar.slider('insulin' , 0 , 847 , 120)
    bmi = st.sidebar.slider('bmi' , 0 , 847 , 120)
    DPF = st.sidebar.slider('spf', 0.08, 2.42 , 1.0)
    Age = st.sidebar.slider('age', 21 , 81)

    user_data = {'pregnancies':pregnancies,
                 'glucose':Glucose,
                 'bloodpressure': BloodPressure,
                 'skinthickness': SkinThickness,
                 'insulin':Insulin,
                 'bmi':bmi,
                 'dpf':DPF,
                 'age':Age
                    }
    
    features = pd.DataFrame(user_data,index=[0])

    return features

#giving the subheader and displaying the user input
user_input = get_user_input()
st.write("""
## Displaying the user input
""")

st.write(user_input)

#creating and training the model
RFClassifiers = RandomForestClassifier()
RFClassifiers.fit(x_train, y_train)



#store the prediction of the model in a variable
st.subheader('Result :')
predict = RFClassifiers.predict(user_input)

if predict == 0:
    st.write("# you don't have Diabetes :smile: :thumbsup:")
else:
    st.write("# Sorry , You have Diabetes 😢")
st.write(predict)

#showing the accuracy score
st.subheader('## Model Accuracy score :')
st.write(str(accuracy_score(y_test, RFClassifiers.predict(x_test))* 100 )+ '%')

st.write("""
## A Project by [shadman](www.github.com/shady4real)
""")

