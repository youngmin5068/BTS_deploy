def load_model(model_path: str):
    import torch
    from app.gmic_v2 import GMIC


    parameters  = {
    "device_type":"cpu",
    "cam_size": (32, 32),
    "K": 6,
    "crop_shape": (128, 128),
    "percent_t":0.005,
    "post_processing_dim": 512,
    "num_classes": 1
    }

    model = GMIC(parameters).to("cpu",dtype=torch.float32)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 전환
    
    return model