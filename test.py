import sys, os
from pathlib import Path

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    base = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    paths = [
        base / 'cudnn' / 'bin',
        base / 'cuda_runtime' / 'bin',
        base / 'cublas' / 'bin',
    ]
    for var in ['PATH', 'CUDA_PATH']:
        os.environ[var] = os.pathsep.join([str(p) for p in paths]) + os.pathsep + os.environ.get(var, '')

set_cuda_paths()


import os
import torch
import torchaudio
import gradio as gr
import subprocess
import threading
import asyncio
import re
import datetime
from tqdm import tqdm
from faster_whisper import WhisperModel
from demucs import pretrained
from demucs.apply import apply_model
import torchaudio.transforms as transforms
from googletrans import Translator, LANGUAGES
import pytesseract
from PIL import Image
from underthesea import sent_tokenize
from huggingface_hub import hf_hub_download, snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from num2words import num2words

# --------------------------
# Phần 3: text-to-Speech (XTTS)
# --------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "checkpoint/GPT_XTTS_FT-March-17-2025_01+28PM-8e59ec3"

CHECKPOINTS = {
#    "vi": {
#        "checkpoint": os.path.join(MODEL_DIR, "best_model_1352094.pth"),
#        "config": os.path.join(MODEL_DIR, "config.json"),
#        "vocab": os.path.join(MODEL_DIR, "vocab.json"),
#    },
    "vi": {
        "checkpoint": os.path.join(MODEL_DIR, "model2.pth"),
        "config": os.path.join(MODEL_DIR, "config2.json"),
        "vocab": os.path.join(MODEL_DIR, "vocab2.json"),
    },
    "default": {
        "checkpoint": os.path.join(MODEL_DIR, "model1.pth"),
        "config": os.path.join(MODEL_DIR, "config1.json"),
        "vocab": os.path.join(MODEL_DIR, "vocab1.json"),
    }
}

SUPPORTED_LANGUAGES = ["vi", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja"]

XTTS_MODEL = None
current_lang = None


def load_model(lang):
    global XTTS_MODEL, current_lang

    if current_lang == lang and XTTS_MODEL is not None:
        print(f"✅")
        return

    ckpt_set = CHECKPOINTS["vi"] if lang == "vi" else CHECKPOINTS["default"]
    checkpoint_path = ckpt_set["checkpoint"]
    config_path = ckpt_set["config"]
    vocab_path = ckpt_set["vocab"]

    if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
        raise FileNotFoundError("❌ Model checkpoint hoặc config không tồn tại!")

    config = XttsConfig()
    config.load_json(config_path)
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=checkpoint_path, vocab_path=vocab_path, use_deepspeed=False)
    XTTS_MODEL.to(device)

    current_lang = lang
    print(f"✅ Model cho ngôn ngữ đã được load!")

def normalize_vietnamese_text(text):
    def replace_date(match):
        day, month, year = match.groups()
        day_text = num2words(int(day), lang='vi', to='ordinal')
        month_text = num2words(int(month), lang='vi')
        year_text = num2words(int(year), lang='vi')
        return f'ngày {day_text} tháng {month_text} năm {year_text}'
    def replace_number(match):
        number = match.group()
        return num2words(int(number.replace(',', '').replace('.', '')), lang='vi')
    text = text.lower()
    text = text.replace("..", ".").replace("!.", "!").replace("?.", "?").replace("(", " ").replace(")", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', replace_date, text)
    text = re.sub(r'\b\d{1,3}(?:[.,]\d{3})*\b', replace_number, text)
    text = text.replace("AI", "Ây Ai").replace("A.I", "Ây Ai").replace("@", "a còng").replace("%", "Phần trăm").replace("&", " và")
    return text


def split_text_by_language(text, lang):
    if lang in ["ja", "zh"]:
        sentences = text.split("。")
    else:
        sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def run_tts(lang, text, speaker_audio_path):
    global XTTS_MODEL

    # Luôn load model theo ngôn ngữ
    load_model(lang)

    if lang not in SUPPORTED_LANGUAGES:
        return f"Ngôn ngữ '{lang}' chưa được hỗ trợ!"
    if not os.path.exists(speaker_audio_path):
        return "File tham chiếu không tồn tại!"

    if lang == "vi":
        text = normalize_vietnamese_text(text)

    sentences = split_text_by_language(text, lang)

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_path,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    wav_chunks = []
    for sentence in tqdm(sentences, desc="🔊 Generating Speech"):
        if not sentence.strip():
            continue
        wav_chunk = XTTS_MODEL.inference(
            text=sentence,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
    output_path = "output.wav"
    torchaudio.save(output_path, out_wav, 24000)
    return output_path


def tts_interface(text=None, text_file=None, lang="vi", speaker_audio=None):
    if text_file is not None:
        if hasattr(text_file, "read"):
            text = text_file.read().decode("utf-8")
        elif isinstance(text_file, str):
            if os.path.exists(text_file):
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = text_file
    if not text or not text.strip():
        return "Lỗi: Không có nội dung văn bản hợp lệ!"
    if speaker_audio is None:
        return "Lỗi: Vui lòng tải lên tệp giọng mẫu!"
    speaker_audio_path = "speaker_ref.wav"
    if hasattr(speaker_audio, "read"):
        with open(speaker_audio_path, "wb") as f:
            f.write(speaker_audio.read())
    else:
        speaker_audio_path = speaker_audio
    return run_tts(lang, text, speaker_audio_path)

#load_model("vi")

# --------------------------
# Phần 1: Speech-to-Text
# --------------------------
model_size = "large"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)




def get_next_filename():
    output_dir = "Transcript"
    os.makedirs(output_dir, exist_ok=True)

    counter_file = os.path.join(output_dir, "counter.txt")
    try:
        with open(counter_file, "r", encoding="utf-8") as f:
            count = int(f.read().strip())
    except FileNotFoundError:
        count = 0
    count += 1
    with open(counter_file, "w", encoding="utf-8") as f:
        f.write(str(count))
    return os.path.join(output_dir, f"transcription_{count}.txt")



# --------------------------
# Phần 2: Video và Audio Processing
# --------------------------
def get_unique_filename(base_name, extension):
    counter = 1
    filename = f"{base_name}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}_{counter}{extension}"
    return filename

def split_audio_video(input_video):
    import os

    video_path = input_video['name'] if isinstance(input_video, dict) else input_video
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 🗂️ Định nghĩa các thư mục đích
    audio_dir = "Extracted Audio"
    video_dir = "Video without Audio"
    clip_dir = "10s Audio Clip"

    # 📁 Tạo thư mục nếu chưa có
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    # 📄 Tạo file path với thư mục tương ứng
    output_audio = get_unique_filename(os.path.join(audio_dir, base_name), ".mp3")
    output_video = get_unique_filename(os.path.join(video_dir, base_name), "_no_audio.mp4")
    output_10s = get_unique_filename(os.path.join(clip_dir, base_name), "_10s.mp3")

    def process():
        subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_audio],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ffmpeg", "-i", video_path, "-an", output_video],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ffmpeg", "-i", output_audio, "-ss", "15", "-t", "10", output_10s],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    thread = threading.Thread(target=process)
    thread.start()
    thread.join()

    return output_audio, output_video, output_10s


def separate_vocals(audio_path):
    import os

    torchaudio.set_audio_backend("soundfile")
    model = pretrained.get_model('htdemucs')
    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    TARGET_SR = 44100
    if sr != TARGET_SR:
        resampler = transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        wav = resampler(wav)
        sr = TARGET_SR

    wav = wav.unsqueeze(0)

    with torch.no_grad():
        sources = apply_model(model, wav, device="cuda" if torch.cuda.is_available() else "cpu")

    # 🗂️ Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = "Separated Vocals"
    os.makedirs(output_dir, exist_ok=True)

    # 📄 Tạo tên file có thư mục
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocal_path = get_unique_filename(os.path.join(output_dir, base_name), "_vocals.wav")

    # 💾 Lưu vocals
    torchaudio.save(vocal_path, sources[0, model.sources.index('vocals')], sr)

    return vocal_path


def extract_text_from_file(file):
    if file is None:
        return ""
    if file.name.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file.name)
        text = pytesseract.image_to_string(image)
    elif file.name.endswith('.txt'):
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        return "Unsupported file format. Please upload an image or a text file."
    return text

def translate(text, file, src_lang, dest_lang):
    if text:
        input_text = text
    elif file:
        input_text = extract_text_from_file(file)
    else:
        return "Please enter text or upload a file."
    translator = Translator()
    try:
        translated = asyncio.run(translator.translate(input_text, src=src_lang, dest=dest_lang))
        return translated.text
    except Exception as e:
        return f"Lỗi: {str(e)}"

language_choices = [(code, name.title()) for code, name in LANGUAGES.items()]

# --------------------------
# Phần 4: LipSync
# --------------------------
def run_wav2lip(video, audio):
    if video is None or audio is None:
        return "Vui lòng chọn cả video và audio!"
    video_path = video.name
    audio_path = audio.name
    output_path = "results/result_voice.mp4"
    command = [
        "python", "inference.py",
        "--checkpoint_path", "wav2lip.pth",
        "--face", video_path,
        "--audio", audio_path,
        "--wav2lip_batch_size", "256"
    ]
    try:
        subprocess.check_call(command, shell=True)
        if os.path.exists(output_path):
            return output_path
        else:
            return "Lỗi: Video đầu ra không tồn tại!"
    except subprocess.CalledProcessError as e:
        return f"Lỗi: {str(e)}"

# --------------------------
# Giao diện Gradio
# --------------------------
video_interface = gr.Interface(
    fn=split_audio_video,
    inputs=gr.Video(label="📤 Upload Video"),
    outputs=[
        gr.Audio(type="filepath", label="🎵 Extracted Audio"),
        gr.Video(label="📼 Video without Audio"),
        gr.Audio(type="filepath", label="🔟 10s Audio Clip")
    ],
    title="🎬 Split Audio/Video + Extract 10s Audio",
    description="Upload a video file to extract audio, create a silent video, and extract a 10s audio clip from second 15."
)


audio_interface = gr.Interface(
    fn=separate_vocals,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=gr.Audio(type="filepath", label="Separated Vocals"),
    title="Separate Vocals from Audio",
    description="Upload an audio file to extract vocals using Demucs."
)

def transcribe_audio(audio_file):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = "".join([segment.text for segment in segments])
    filename = get_next_filename()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(transcription)

    return filename, transcription


transcription_interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=[
        gr.File(label="📄 Download Transcript (.txt)"),
        gr.Textbox(label="📝 Transcript Content", lines=10, interactive=False)
    ],
    title="🗣️ Speech to Text",
    description="Upload an audio file to convert speech into text and view the result directly."
)


translation_interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter Text to Translate"),
        gr.File(label="Upload Text or Image File"),
        gr.Dropdown(choices=language_choices, value="auto", label="Source Language"),
        gr.Dropdown(choices=language_choices, value="en", label="Target Language")
    ],
    outputs=gr.Textbox(label="Translation Result"),
    title="Translation",
    description="Enter text or upload a text/image file to translate."
)

tts_tab = gr.Interface(
    fn=tts_interface,
    inputs=[
        gr.Textbox(label="Enter Text"),
        gr.File(label="Or Upload a Text File"),
        gr.Dropdown(SUPPORTED_LANGUAGES, label="Select Language", value="vi"),
        gr.Audio(type="filepath", label="🎙️ Upload Reference Voice"),
    ],
    outputs=gr.Audio(label="🔊 Generated Speech"),
    title="🗣️ viXTTS - Text to Speech",
    description="Nhập nội dung và tải lên hoặc ghi âm mẫu giọng nói để tạo giọng nhân tạo tương tự."
)

lipsync_interface = gr.Interface(
    fn=run_wav2lip,
    inputs=[
        gr.File(label="Select Video", type="filepath", file_types=[".mp4"]),
        gr.File(label="Select Audio", type="filepath", file_types=[".mp3", ".wav"]),
    ],
    outputs=gr.Video(label="Processed Video"),
    title="LipSync with Wav2Lip",
    description="Select a video and audio to synchronize lips with the sound."
)

gui = gr.TabbedInterface(
    [video_interface, audio_interface, transcription_interface, translation_interface, tts_tab, lipsync_interface],
    ["Split Audio + 10s Clip", "Audio Processing", "Speech to Text", "Translation", "Text to Speech", "LipSync"]
)
# --------------------------

def preload_wav2lip_checkpoint():
    checkpoint_path = "wav2lip.pth"
    if not os.path.exists(checkpoint_path):
        print("❌ Wav2Lip checkpoint not found!")
        return
    try:
        print("⏳ Preloading Wav2Lip checkpoint...")
        torch.load(checkpoint_path, map_location="cpu")  # Hoặc "cuda" nếu bạn muốn preload vào GPU
        print("✅ Wav2Lip checkpoint loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading Wav2Lip checkpoint: {e}")

# Gọi hàm preload
preload_wav2lip_checkpoint()

# Preload XTTS Vietnamese model
print("⏳ Preloading XTTS Vietnamese model...")
load_model("vi")



if __name__ == "__main__":
    gui.launch()
