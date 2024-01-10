from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import pickle
import resampy

app = FastAPI()


def extract_feature(file_stream, mfcc):
    X, sample_rate = librosa.load(file_stream.file, res_type='kaiser_fast')
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    return result


with open('Emotion_Voice_Detection_Model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.post("/upload")
async def upload_file(audio_file: UploadFile = File(...), mfcc: bool = True):
    # Check if the file is empty
    if not audio_file:
        return {"message": "No file provided"}, 400

    new_feature = extract_feature(audio_file, mfcc)
    ans = []
    ans.append(new_feature)
    ans = np.array(ans)
    prediction = model.predict(ans)
    return JSONResponse(content={"prediction": prediction.tolist()}, status_code=200)
