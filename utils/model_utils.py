from segment_anything import sam_model_registry, SamPredictor

basic_SAM_model_string = "vit_b"


def load_model(model_path: str):
    sam = sam_model_registry[basic_SAM_model_string](checkpoint=model_path)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor
