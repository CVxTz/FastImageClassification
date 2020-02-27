from fastapi import FastAPI, File, UploadFile
from predictor import ImagePredictor

app = FastAPI()

predictor_config_path = "../example/predictor_config.yaml"

predictor = ImagePredictor.init_from_config_path(predictor_config_path)


@app.post("/scorefile/")
def create_upload_file(file: UploadFile = File(...)):
    return predictor.predict_from_file(file.file)
