from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = load_model("brain.h5")
labels = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
image_size = 150

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=['*']
)

def preprocess_image(img):
    img = cv2.resize(img, (image_size, image_size))
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

@app.get("/")  
async def home():
    return {"message": "Brain Tumor Final"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prediction = predict(img)
    return {"prediction": prediction}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
