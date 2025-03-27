from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# โหลดโมเดลจากโฟลเดอร์ที่มีไฟล์ safetensors ทั้งหมด
model_path = 'C:/models_llama/Llama-3.2-11B-Vision'

# โหลดรูปภาพ
image_path = r"D:/work/python/ocr_utl3_use_llm_from_huggingface/images/$S00119AA-11D18ED26.000000E.jpg"
# image_path = r"D:/work/python/ocr_utl3_use_llm_from_huggingface/images/T3508EF-02-1.jpg"

model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16, 
    # device_map="auto"
)
model.tie_weights()
model.to("cuda")

processor = AutoProcessor.from_pretrained(model_path)

# prompt = "<|image|>If I had to read an OCR for this one"
prompt = "<|image|>read OCR:"
raw_image = Image.open(image_path)

inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)

# ทำ inference
output = model.generate(**inputs, max_new_tokens=50)

generate_output = processor.decode(output[0], skip_special_tokens=True)

# แสดงผล
print(generate_output)
