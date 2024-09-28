from flask import Flask
from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker
from flask import abort, request, escape, make_response
import numpy as np
import pickle
import pandas as pd
import sqlite3

app = Flask(__name__)
model = pickle.load(open('projpredictgcv.pkl', 'rb'))

#Setup of db, we use sqlite for convenience.
engine = create_engine("sqlite+pysqlite:///health_proj.db", echo=True)
conn= engine.connect()

loan = pd.read_csv('healthcare-dataset-stroke-data.csv')
loan.to_sql('health_proj', conn, if_exists='replace', index = False)

@app.route('/getusers', methods=['GET'])
def getusers():
    """
    Gets all users in the db and passes them as json
    """
    global engine
    if request.method == "GET":
        with engine.connect() as conn:
            results=conn.execute(text("select * from health_proj"))
            return {'data': [{'gender': r[0], 'age': r[1],
                            'hypertension': r[2], 'heart_disease': r[3],
                            'ever_married': r[4], 'work_type': r[5],
                            'Residence_type': r[6], 'avg_glucose_level': r[7],
                            'bmi': r[8], 'smoking_status': r[9],'stroke':r[10]} for r in results.fetchall()]}
    else: abort(405)

# Here we post json of maniputlate it and return a resulat.
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        if request.json and 'data' in request.json:
            print(request.json)
            df_predict_live = pd.DataFrame(request.json.get("data"),columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'avg_glucose_level', 'bmi','smoking_status'])
            return {"Prediction": int(model.predict(df_predict_live)[0])}
            # return {"Prediction": int(model.predict(request.json['data']))}
        else: abort(405)
    else: abort(405)


@app.route('/insert', methods=['PUT'])
def insert():
    """
    Takes json of the form {"username": username, "id": id} and puts in db
    """
    global engine
    if request.method == "PUT":
        #print(1)
        print(request.json)
        if request.json:
            try:
                print(2)
                with engine.connect() as conn:
                    print(3)
                    print(request.json['age'])
                    conn.execute(
                    text("INSERT INTO health_proj (gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,\
                                        avg_glucose_level, bmi, smoking_status, stroke) VALUES (:gender, :age,:hypertension,\
                                            :heart_disease,:ever_married,:work_type,:Residence_type,:avg_glucose_level,:bmi,:smoking_status,:stroke)"),
                            [{"gender": request.json['gender'], "age": request.json['age'],"hypertension": request.json['hypertension'],\
                                "heart_disease": request.json['heart_disease'],"ever_married": request.json['ever_married'],"work_type": request.json['work_type'],\
                                    "Residence_type": request.json['Residence_type'],"avg_glucose_level": request.json['avg_glucose_level'],"bmi": request.json['bmi'],\
                                        "smoking_status": request.json['smoking_status'],"stroke": request.json['stroke']}]
                        )
                    return "record created"
            except:
                abort(400)
        else: 
            abort(405)
    else: abort(405)



@app.errorhandler(405)
def malformed_query(error):
    """
    Redirects 405 errors
    """
    resp = make_response("Malformed Query")
    return resp
