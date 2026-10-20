"""API du prototype EYE Médicale. Les poids sont chargés depuis EYE_MODEL_DIR."""
import io
import json
import logging
import os
from pathlib import Path
import traceback

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from text_utils import clean

MODEL_DIR = Path(os.environ.get("EYE_MODEL_DIR", "model/eye-medicale"))
required = ["adapter_config.json", "tokenizer_config.json",
            "vision_projector.pt", "multimodal_config.json"]
missing = [name for name in required if not (MODEL_DIR / name).is_file()]
if missing:
    raise FileNotFoundError("Export du modèle incomplet : " + ", ".join(missing))
model_config = json.loads((MODEL_DIR / "multimodal_config.json").read_text())
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

device     = "cuda" if torch.cuda.is_available() else "cpu"
device_map = {"": 0} if device == "cuda" else {"": "cpu"}
LORA_PATH = str(MODEL_DIR)
logging.info("LORA_PATH = %s | %s fichiers", LORA_PATH, len(os.listdir(LORA_PATH)))

gemma, tokenizer = FastLanguageModel.from_pretrained(
    model_config["base_model"],
    load_in_4bit=True,
    trust_remote_code=True,
    device_map=device_map,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if model_config["image_token"] not in tokenizer.get_vocab():
    raise ValueError("Le tokenizer exporté ne contient pas le token image.")
gemma.resize_token_embeddings(len(tokenizer))
gemma = PeftModel.from_pretrained(gemma, LORA_PATH)

clip_model     = CLIPModel.from_pretrained(model_config["clip_model"]).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_config["clip_model"])

class GemmaMultimodal(nn.Module):
    def __init__(self, gemma, clip):
        super().__init__()
        self.gemma, self.clip = gemma, clip
        self.proj = nn.Linear(clip.config.projection_dim,
                              gemma.config.hidden_size, bias=False).to(device)

    def forward(self, input_ids, attention_mask, pixel_values=None, labels=None):
        txt_emb = self.gemma.get_input_embeddings()(input_ids)
        with torch.no_grad():
            img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        img_emb  = self.proj(img_feat).unsqueeze(1).to(txt_emb.dtype)
        img_tok  = tokenizer.convert_tokens_to_ids("<|image|>")
        mix      = torch.where((input_ids == img_tok).unsqueeze(-1), img_emb, txt_emb)
        return self.gemma(inputs_embeds=mix,
                          attention_mask=attention_mask,
                          labels=labels)

mmodel = GemmaMultimodal(gemma, clip_model).to(device)
mmodel.proj.load_state_dict(torch.load(
    MODEL_DIR / "vision_projector.pt", map_location=device, weights_only=True
))
mmodel.eval()

app = FastAPI(title="EYE Médicale — CLIP / Gemma",
              description="Gemma-2B + LoRA multimodal CLIP",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logging.error("⚠️ %s → %s", request.url.path, exc)
        traceback.print_exc(limit=2)
        return JSONResponse(status_code=500, content={"error": "L’analyse a échoué."})

@app.get("/")
async def root():
    return {"status": "OK (Gemma VLM)"}

def infer(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    px  = clip_processor(images=img, return_tensors="pt").pixel_values.to(device)

    prompt = (tokenizer.bos_token or "") + "<|image|> " \
           + "You are a medical radiologist. Analyse precisely."
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    ids, attn = enc.input_ids, enc.attention_mask

    gen = []
    with torch.no_grad():
        for _ in range(128):
            out = mmodel(input_ids=ids, attention_mask=attn, pixel_values=px)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            if nxt.item() == tokenizer.eos_token_id: break
            gen.append(nxt.item())
            ids  = torch.cat([ids, nxt], 1)
            attn = torch.cat([attn, torch.ones_like(nxt)], 1)

    return clean(tokenizer.decode(gen, skip_special_tokens=True))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    txt = infer(await file.read())
    return {"result": txt}
