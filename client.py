import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/lip_sync/"  # Địa chỉ API FastAPI

def call_lip_sync(video, audio):
    if video is None or audio is None:
        return "Vui lòng chọn cả video và audio!", None
    
    files = {
        "video": (video.name, open(video.name, "rb"), "video/mp4"),
        "audio": (audio.name, open(audio.name, "rb"), "audio/wav"),
    }

    response = requests.post(API_URL, files=files)
    
    if response.status_code == 200:
        video_url = response.json().get("video_url")
        return "Xử lý thành công!", f"results/{video_url}"
    else:
        return f"Lỗi: {response.json().get('detail')}", None

with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ LipSync với Wav2Lip")
    
    with gr.Row():
        video_input = gr.File(label="Chọn Video (MP4)")
        audio_input = gr.File(label="Chọn Audio (WAV/MP3)")

    process_button = gr.Button("🔄 Bắt đầu đồng bộ")
    output_message = gr.Textbox(label="Trạng thái")
    output_video = gr.Video(label="Video sau xử lý")

    process_button.click(call_lip_sync, inputs=[video_input, audio_input], outputs=[output_message, output_video])

if __name__ == "__main__":
    demo.launch()
