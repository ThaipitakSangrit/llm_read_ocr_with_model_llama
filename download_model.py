from huggingface_hub import snapshot_download

# ตั้งค่าพาธที่ต้องการเก็บโมเดล
model_path = "./models/Llama-3.2-11B-Vision-Instruct"

# ดาวน์โหลด LLaMA 3.2-Vision-11B จาก Hugging Face
snapshot_download(repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct", 
                  local_dir=model_path, 
                  local_dir_use_symlinks=False,
                  resume_download=True)