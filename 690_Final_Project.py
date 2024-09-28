#https://towardsdatascience.com/make-dataframes-interactive-in-streamlit-c3d0c4f84ccb
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns 
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')
import pydeck as pdk
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix,roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import plotly.figure_factory as ff
import pickle
from sklearn.svm import SVC 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import requests
import sqlite3
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import datetime
import requests
from streamlit_option_menu import option_menu


#tab1, tab2 = st.tabs(["Dataset", "Prediction"])
def streamlit_menu(example=1):
    selected = option_menu(
                menu_title=None,  # required
                options=["Dataset", "Prediction"],  # required
                # icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )
    return selected

selected = streamlit_menu()

if selected == "Dataset":
    #st.title(f"You have selected {selected}")
    st.title("STROKE PREDICTION DATASET")
    r=requests.get('http://127.0.0.1:5000/getusers')
    re= r.json()
    y= re['data']
    df= pd.DataFrame.from_dict(y, orient ='columns')
    
    # st.markdown('Dataset')
    st.write(df)

# with tab1:
    
#     # st.header("STROKE PREDICTION DATASET")
#     # r=requests.get('http://127.0.0.1:5000/getusers')
#     # re= r.json()
#     # y= re['data']
#     # df= pd.DataFrame.from_dict(y, orient ='columns')
    
#     # # st.markdown('Dataset')
#     # st.write(df)
if selected == "Prediction":
    #st.title(f"You have selected {selected}")

#with tab2: 
    # a1c levels explanation

# avg_glucose_level < 140 is Normal
# avg_glucose_level between 140 and 200 is Prediabetes
# avg_glucose_level > 200 is diabetes

##Adding new column to categorize BMI
# BMI levels explanation

# BMI < 18.5 is Underweight
# BMI between 18.5 and 24.9 is Healthy weight
# bmi between 25.0 and 29.9 is Over weight
# bmi >= 30 is Obesity
    st.title("STROKE PREDICTION")
    col1,col2,col3 = st.columns(3)
    col4,col5,col6 = st.columns(3)
    col7,col8= st.columns(2)
    col9, col10 = st.columns(2)
    #conn= create_connection('ant.db')
    with col1:   
        gen = st.selectbox('Gender',['Male','Female'])
    with col2:
        Age = st.number_input('Age', 0, 99)
    with col3:
        tension= st.selectbox('Hypertension',['0','1'])
    with col4:
        heart = st.selectbox('Heart Disease',['0','1'])
    with col5:
        married = st.selectbox('Married?',['Yes','No'])
    with col6:
        work = st.selectbox('Type of Work', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    with col7:
        res = st.selectbox('Residence type', ['Urban', 'Rural'])
    with col8:
        glu = st.slider('Average Glucose level', 50.0, 300.0, 75.0)
        #glu = st.number_input('Average Glucose level',50.0,300.0)
    with col9:
        bmi = st.slider('Bmi',10.0,100.0, 25.0)
        #bmi = st.number_input('Bmi',10.0,100.0)
    with col10:
        smoking = st.selectbox('Smoking status',['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    
    def predict(event):
        resp = requests.post('http://127.0.0.1:5000/prediction', json={'data':[[gen, Age, tension, heart, married, work, res, glu, bmi, smoking]]})
        print(resp.json())
        val = resp.json()
        stroke = val["Prediction"]
        print(stroke)
        return stroke

    stroke = predict(1)
    print(stroke)
    col_1, col_2 = st.columns(2)
    with col_1:
        if st.button('Predict'):
            st.write("The predicted value is:", str(stroke))
            if stroke == 0:
                st.write("YOU ARE FINE....")
            else:
                st.write("CONSULT A DOCTOR...!!!!!")
    #with col_2:
    if st.button('Insert'):
        ins = requests.put('http://127.0.0.1:5000/insert', json = {"gender": gen, "age": Age,"hypertension": tension,\
                            "heart_disease": heart,"ever_married": married,"work_type": work,\
                                "Residence_type": res,"avg_glucose_level": glu,"bmi": bmi,\
                                    "smoking_status": smoking,"stroke": stroke })

        st.write(ins.text)
            
        r2=requests.get('http://127.0.0.1:5000/getusers')
        re2= r2.json()
        y2= re2['data']
        df2= pd.DataFrame.from_dict(y2, orient ='columns')
        
        st.markdown('Dataset with inserted row')
        st.write(df2.tail())
    
    #db_conn.close()

