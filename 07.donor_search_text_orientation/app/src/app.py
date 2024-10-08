import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from service import rotate_and_save

app = FastAPI()

tmppath = r'tmp/'
resultpath = r'tmp/result/'
if not os.path.exists(tmppath): os.makedirs(tmppath)
if not os.path.exists(resultpath): os.makedirs(resultpath)

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

INP_SIZE = 224
DEVICE = "cpu"

LOGOPATH = 'tmp/logo4.jpg'  #'templates/img/logo3.jpg'

# image size according to other applications

@app.get("/health")
def health():
    return {"status": "OK"}


@app.get('/')
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request,
                                       "logopath": LOGOPATH})


@app.post("/predict-detect")
def process_request(file: UploadFile, request: Request):
    """save file to the local folder and send the image to the process function"""

    # путь к тестовой картинке
    input_image_path = tmppath + file.filename
    # путь к папке куда падает уже перевернутая картинка
    output_image_path = resultpath + file.filename


    #save_pth = "tmp/" + file.filename
    app_logger.info(f'processing file - segmentation {input_image_path}')
    with open(input_image_path, "wb") as fid:
        fid.write(file.file.read())
    result = rotate_and_save(input_image_path, output_image_path)

    return templates.TemplateResponse('detect_form.html',
                                      {"request": request,
                                       "result": result,
                                       "outpath": output_image_path,
                                       "inputpath": input_image_path})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)