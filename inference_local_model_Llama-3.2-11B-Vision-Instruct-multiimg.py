from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import os
import re

# โหลดโมเดลจากโฟลเดอร์ที่มีไฟล์ safetensors ทั้งหมด
model_path = 'C:/models_llama/Llama-3.2-11B-Vision-Instruct'

# โหลดโมเดลและโปรเซสเซอร์แค่ครั้งเดียว
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
model.tie_weights()
model.to("cuda")

processor = AutoProcessor.from_pretrained(model_path)

# ฟังก์ชันสำหรับประมวลผลแต่ละภาพ
def process_image(image_path):
    image = Image.open(image_path)

    # กำหนดข้อความที่ต้องการ
    messages = [
        {
            "role": "system", "content": [
                {"type": "text", "text": "You are an OCR extraction assistant. Extract only the raw text from the image exactly as it appears."
                "The characters are brighter from the background, and the image is an IC unit mark, case sensitive."
                "Preserve all uppercase and lowercase letters without modification."
                "Ensure that elements such as punctuation marks, numbers, and words with spaces are handled correctly in separate lines if they appear visually distinct."}
            ]
        },
        {
            "role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract only the raw text that is bright on a dark background from the image without any additional descriptions."}
            ]
        }
    ]

    # ใช้ processor เพื่อสร้าง input
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    # สร้าง input ที่ส่งเข้าโมเดล
    inputs = processor(image, text_input, return_tensors="pt").to(model.device)

    # ทำ inference
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # ถอดรหัสผลลัพธ์
    generate_output = processor.decode(output[0], skip_special_tokens=True)

    # return generate_output

    match = re.search(r"assistant\n\n(.*)", generate_output, re.DOTALL)

    if match:
        final_output = match.group(1).strip()
        return final_output
    else:
        print("ไม่พบคำตอบ")

# โฟลเดอร์ที่มีรูปภาพ
image_folder = 'D:/work/python/ocr_utl3_use_llm_from_huggingface/images'

# ลูปผ่านทุกไฟล์ในโฟลเดอร์และประมวลผล
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)

    # ตรวจสอบว่าเป็นไฟล์รูปภาพ
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing image: {image_file}")
        result = process_image(image_path)
        print(f"Result for {image_file}:\n{result}\n--------")
