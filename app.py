from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRouter
import logging
import logging.config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from initializer import LoggerInstance
# from transformers import pipeline
# import zipfile
# from inference_pipeline import InferencePipeline

app = FastAPI(title="text summarizer", version="0.1")

inference_pipeline = None

router = APIRouter()
# logger = LoggerInstance().get_logger(__name__)
logging.config.fileConfig('logging_config.ini')
logger = logging.getLogger(__name__)

gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

# def load_model_files(zip_file):
#     with zipfile.ZipFile(zip_file,'r') as zip_ref:
#         zip_ref.extractall('unzipped_model')


@app.on_event("startup")
async def on_startup():
    logger.info('start server')
    global inference_pipeline
    logger.info('load model')
    # uncomment when to use your finetuned model
    # inference_pipeline=InferencePipeline().inference_pipe
    global tokenizer, model
    checkpt = "google/pegasus-cnn_dailymail"
    tokenizer = AutoTokenizer.from_pretrained(checkpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpt)
    logger.info('app is ready!')


@app.get("/")
async def root():
    return PlainTextResponse('Welcome to Pegasus Text Summarizer!', 200)


@app.get("/predict")
async def predict(text: str):
    logger.info('start inference')
    # uncomment to use your model
    # output=inference_pipeline(text, **gen_kwargs)[0]["summary_text"]

    # Generate Summary from pretrained model
    inputs = tokenizer([text], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"])
    output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return PlainTextResponse(output, 200)
