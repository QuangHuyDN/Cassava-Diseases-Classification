import joblib
import io
import uvicorn
import numpy as np
import json
import datetime

from PIL import Image
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

model = load_model('cassava-model.h5')
test_datagen, class_names = joblib.load('model-data.pkl')

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get('/')
async def root():
    return {'detail': 'API\'s root'}

@app.get('/records/')
async def records():
    with open('data.json', 'r') as file:
        data = file.read()
    
    try:
        data = json.loads(data)
    except json.decoder.JSONDecodeError:
        data = []

    return data

@app.post('/classfication/')
async def classify_image(file: bytes = File(...)):
    img = Image.open(io.BytesIO(file))
    scaled_img = img.resize((224, 224))

    img_pred = test_datagen.flow(np.expand_dims(np.asarray(scaled_img), axis=0))

    prediction = model.predict(img_pred)
    proba = list(prediction[0])
    res_val = max(proba)
    res_idx = proba.index(res_val)

    with open('data.json', 'r') as file:
        data = file.read()
    
    try:
        data = json.loads(data)
    except json.decoder.JSONDecodeError:
        data = []
    finally:
        n = len(data)
        data += [{
            'id': n+1,
            'Class': class_names[res_idx],
            'Proba': float(res_val),
            'created_on': str(datetime.datetime.now().date())
        }]
    data = json.dumps(data, indent=4)

    with open('data.json', 'w') as file:
        file.write(data)
    
    res = {
        'Class': class_names[res_idx],
        'Proba': float(res_val),
    }
    return res

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
