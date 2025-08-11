from flask import Flask, render_template, request, redirect, url_for, send_file
import ffmpeg
import whisper
#from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os
from deep_translator import GoogleTranslator


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
import os

def extract_audio(video_path, audio_output):
    print(f"Extracting audio from: {video_path}")
    print(f"Output audio path: {audio_output}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    ffmpeg.input(video_path).output(audio_output, format='mp3', acodec='mp3', ar='16000').run(overwrite_output=True)

def extract_audio(video_path, audio_output):
    """Extract audio from video."""
    ffmpeg.input(video_path).output(audio_output, format='mp3', acodec='mp3', ar='16000').run(overwrite_output=True)

def transcribe_audio(audio_path):
    """Transcribe audio to text using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# def translate_text(text, target_language):
#     """Translate text into chosen language."""
#     translator = Translator()
#     translated_text = translator.translate(text, dest=target_language)
#     return translated_text.text

def translate_text(text, target_language):
    translated = GoogleTranslator(source='auto', target=target_language).translate(text)
    return translated

def summarize_text(text):
    """Summarize text using LSA Summarizer."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Generate 3 summary sentences
    return " ".join(str(sentence) for sentence in summary)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["video"]
        target_language = request.form["language"]
        summarize = request.form.get("summarize")

        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        audio_path = file_path.rsplit(".", 1)[0] + ".mp3"

        file.save(file_path)
        extract_audio(file_path, audio_path)
        transcribed_text = transcribe_audio(audio_path)
        translated_text = translate_text(transcribed_text, target_language)

        if summarize:
            translated_text = summarize_text(translated_text)

        return render_template("index.html", text=translated_text)

    return render_template("index.html", text="")

if __name__ == "__main__":
    app.run(debug=True)


