from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import subprocess
import os
from pathlib import Path

app = FastAPI()

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/lip_sync/")
async def lip_sync(video: UploadFile = File(...), audio: UploadFile = File(...)):
    video_path = OUTPUT_DIR / video.filename
    audio_path = OUTPUT_DIR / audio.filename
    output_path = OUTPUT_DIR / "result_voice.mp4"
    
    try:
        # Lưu file video và audio
        with video_path.open("wb") as f:
            f.write(video.file.read())
        with audio_path.open("wb") as f:
            f.write(audio.file.read())

        # Chạy Wav2Lip
        command = [
            "python", "inference.py",
            "--checkpoint_path", "wav2lip.pth",
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--wav2lip_batch_size", "256"
        ]
        subprocess.check_call(command, shell=True)
        
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Lỗi: Video đầu ra không tồn tại!")
        
        return {"video_url": f"{output_path.name}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

# @app.get("/download/{filename}")
# def download_file(filename: str):
#     file_path = OUTPUT_DIR / filename
#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="File không tồn tại")
#     return file_path.read_bytes()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
