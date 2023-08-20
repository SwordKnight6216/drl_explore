from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import inference
import training
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from time import sleep

app = FastAPI()

# This will allow you to serve the index.html as the root page
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Define a variable to store training progress
training_progress = 0

class InputData(BaseModel):
    data: list

@app.get("/")
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/predict/")
async def get_prediction(input_data: InputData):
    try:
        prediction = inference.suggest_action(input_data.data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to start training
@app.get("/start-training/")
async def start_training(background_tasks: BackgroundTasks):
    global training_progress
    training_progress = 0
    background_tasks.add_task(run_training)
    return {"status": "started"}

# Simulated training function
def run_training():
    global training_progress
    training.train()

@app.get("/get-progress/")
async def get_progress():
    global training_progress
    return {"progress": training_progress}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
