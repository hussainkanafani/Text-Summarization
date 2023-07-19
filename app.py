from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
#from initializer import LoggerInstance
from fastapi.routing import APIRouter
from transformers import pipeline
import zipfile
from inference_pipeline import InferencePipeline
import logging
import logging.config

app= FastAPI(title="text summarizer", version="0.1")

inference_pipeline = None

router = APIRouter()
#logger = LoggerInstance().get_logger(__name__)
logging.config.fileConfig('logging_config.ini')
logger = logging.getLogger(__name__)

gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

#def load_model_files(zip_file):
#    with zipfile.ZipFile(zip_file,'r') as zip_ref:
#        zip_ref.extractall('unzipped_model')


@app.on_event("startup")
async def on_startup():
    logger.info('start server')
    global inference_pipeline
    logger.info('load model')
    inference_pipeline=InferencePipeline().inference_pipe
    logger.info('app is ready!')


@app.get("/")
async def root():
    return PlainTextResponse('Welcome to Pegasus Text Summarizer!',200)


@app.get("/predict")
async def predict(text:str):
    logger.info('start inference')
    output=inference_pipeline(text, **gen_kwargs)[0]["summary_text"]
    return PlainTextResponse(output,200)