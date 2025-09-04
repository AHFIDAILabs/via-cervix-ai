from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from predict import Predictor
from config import TRAINED_MODEL_PATH

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load the predictor
predictor = Predictor(TRAINED_MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image and returns the prediction."""
    contents = await file.read()
    with open("temp_image.jpg", "wb") as f:
        f.write(contents)

    predicted_class, confidence = predictor.predict("temp_image.jpg")
    return {"prediction": predicted_class, "confidence": f"{confidence:.2f}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)