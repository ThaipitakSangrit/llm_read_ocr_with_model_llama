from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import re

# โหลดโมเดลจากโฟลเดอร์ที่มีไฟล์ safetensors ทั้งหมด
model_path = 'C:/models_llama/Llama-3.2-11B-Vision-Instruct'

# โหลดรูปภาพ
image_path = r"D:/work/python/ocr_utl3_use_llm_from_huggingface/images/$S00119AA-11D18ED26.000000E.jpg"
# image_path = r"D:/work/python/ocr_utl3_use_llm_from_huggingface/images/T3508EF-02-1.jpg"

image = Image.open(image_path)

# โหลดโมเดลและโปรเซสเซอร์
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    # device_map="auto"
)
model.tie_weights()
model.to("cuda")

processor = AutoProcessor.from_pretrained(model_path)

messages = [
    # {
    #     "role": "system", "content": [
    #         {"type": "text", "text": "You are an OCR extraction assistant. Extract only the raw text from the image exactly as it appears."
    #         "The characters are brighter from the background, and the image is an IC unit mark, case sensitive."
    #         "Preserve all uppercase and lowercase letters without modification."
    #         "If the text is visually separated into distinct lines in the image, ensure each line is assigned a unique line_number and preserve the order of the lines."
    #         "Ensure that elements such as punctuation marks, numbers, and words with spaces are handled correctly in separate lines if they appear visually distinct."
    #         "Respond **ONLY** with a compact JSON object, in text without extra spaces or line breaks."
    #         "Ensure the text is extracted correctly, preserving the original case, line breaks."
    #         "Use this format:\n" '{"text":"<full extracted text>","lines":[{"line_number":<line>,"content":"<text>"}]}'
    #         "Ensure each line has 'line_number' and 'content' fields."
    #         "Respond ONLY in the specified JSON format."
    #         "Ensure every distinct word or segment that visually appears separated in the image is treated as an individual line."
    #         "Preserve the correct separation of lines, ensuring that visually distinct words or characters are not merged together."
    #         "Ensure that each distinct word, phrase, or character group is assigned its own 'line_number'."}
    #     ]
    # },
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
generate_output = processor.decode(output[0],skip_special_tokens=True)

# แสดงผล
# print(generate_output)
match = re.search(r"assistant\n\n(.*)", generate_output, re.DOTALL)

if match:
    final_output = match.group(1).strip()
    print(final_output)
else:
    print("ไม่พบคำตอบ")