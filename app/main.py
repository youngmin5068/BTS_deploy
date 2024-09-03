from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image
import torch
import os
import cv2
from io import BytesIO
import base64
import numpy as np
from .model import load_model
from .data_utils import load_dicom_image, save_png

app = FastAPI(title="딥러닝 모델 API")

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 모델 로드
model = load_model("/Users/kimyoungmin/my_fastapi_app/models/Breast_MTL.pth")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str

@app.post("/process/")
async def process_dicom(request: Request, file: UploadFile = File(...)):
    try:
        # DICOM 파일을 서버에 저장
        dicom_file_path = f"/Users/kimyoungmin/my_fastapi_app/input_data/{file.filename}"
        with open(dicom_file_path, "wb") as f:
            f.write(await file.read())

        # DICOM 이미지 로드 및 전처리
        image_array = load_dicom_image(dicom_file_path)

        # 입력 이미지 준비
        input_image = Image.fromarray((image_array.squeeze() * 255).astype(np.uint8))
        input_image_base64 = image_to_base64(input_image)

        dcm_forOverlay = (image_array.squeeze() * 255.0).astype(np.uint8)
        dcm_forOverlay = np.expand_dims(dcm_forOverlay, axis=2)
        dcm_forOverlay = cv2.cvtColor(dcm_forOverlay, cv2.COLOR_GRAY2BGR)
        
        # 모델에 입력할 형태로 변환 (예: 배치 크기 추가)
        input_tensor = torch.tensor(image_array, dtype=torch.float32)

        # 모델 추론
        with torch.no_grad():
            _, _, prediction, output_img = model(input_tensor)
        prediction = prediction.item()

        output_img = torch.sigmoid(output_img).squeeze().cpu().numpy()
        output_img = (output_img * 255.0).astype(np.uint8)
        output_img = np.expand_dims(output_img, axis=2)
        output_img = cv2.applyColorMap(output_img, cv2.COLORMAP_JET)
        output_img[:, :, 0] = 0
        output_img = output_img[:, :, ::-1]

        alpha = 0.7  # 원본 이미지와 overlay 이미지의 가중치 조절
        overlay_result = cv2.addWeighted(dcm_forOverlay, alpha, output_img, 1 - alpha, 0)

        # 결과 이미지를 PIL 이미지로 변환
        result_image = Image.fromarray(overlay_result)
        result_image_base64 = image_to_base64(result_image)

        prediction = f"{prediction:.4f}"

        # 입력 이미지와 결과 이미지를 HTML 페이지에 표시
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction,
            "input_image": input_image_base64,
            "result_image": result_image_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
