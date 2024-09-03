from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch
import os
import cv2
from .model import load_model
import numpy as np
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

@app.post("/process/")
async def process_dicom(request: Request, file: UploadFile = File(...)):
    try:
        # DICOM 파일을 서버에 저장
        dicom_file_path = f"/Users/kimyoungmin/my_fastapi_app/input_data/{file.filename}"
        with open(dicom_file_path, "wb") as f:
            f.write(await file.read())

        # DICOM 이미지 로드 및 전처리
        image_array = load_dicom_image(dicom_file_path)

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

        # 결과 PNG 파일 저장
        png_file_path = f"/Users/kimyoungmin/my_fastapi_app/output_data/{os.path.splitext(file.filename)[0]}.png"
        save_png(overlay_result, png_file_path)

        prediction = f"{prediction:.4f}"

        # 예측 결과와 PNG 파일 경로를 HTML 페이지로 반환
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction,
            "png_file": f"/download_png/?file_path={png_file_path}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_png/")
async def download_png(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png", filename=os.path.basename(file_path))
    else:
        raise HTTPException(status_code=404, detail="File not found")
