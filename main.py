from typing import Union, List
from fastapi import FastAPI, File, UploadFile , HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import pytesseract
from fastapi import FastAPI
import cv2
import numpy as np

import pickle
from face_landmark_detection import get_face_embbed, cosine_similarity

filename = 'models/model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

    
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/iris/")
async def iris_predict(data: List[float]):
    print(data)
    result = loaded_model.predict([data])
    return JSONResponse(content={"predict": str(result[0])})
    
@app.post("/ocr/")
async def ocr_text(file: UploadFile = File(None)):
    if file:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
    else:
        raise HTTPException(status_code=400, detail="No file uploaded")

    custom_config = r'-l eng+khm --oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(img, config=custom_config).replace("\n", "")#.split("\n")
    # save as log and monitoring
    # image, its text, 0.9%
    # "", 0.6%
    # extracted_text = ""
    return JSONResponse(content={"recognized_texts": extracted_text})

@app.post("/face_verify/")
async def face_verify(file: UploadFile = File(None), file_2: UploadFile = File(None)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_encodings = get_face_embbed(img)[0]

    image_bytes = await file_2.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_2_encodings = get_face_embbed(img_2)[0]

    scroing = cosine_similarity(img_2_encodings, img_encodings)    
    
    # save image, image 2, score

    THRESHOLD = 0.90
    if scroing > THRESHOLD:
        return JSONResponse(content={"matched": True, "scroing": scroing})
    else:
        return JSONResponse(content={"matched": False, "scroing": scroing})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
