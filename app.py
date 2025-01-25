from flask import Flask, request, jsonify
import os
import whisper
from pydub import AudioSegment
import difflib
from sentence_transformers import SentenceTransformer, util
import ffmpeg
import sys
import time
from nltk.tokenize import sent_tokenize
import nltk
from pathlib import Path
import re

nltk.download('punkt_tab')

app = Flask(__name__)
# Paths for saving uploaded files
UPLOAD_FOLDER = 'videos'
AUDIO_FOLDER = 'audio'
TRANSCRIPTION_FOLDER = 'transcriptions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)


BASE_DIR = Path(__file__).resolve().parent  # Automatically detects the current directory
SSL_CERTIFICATE_SERVER_PEM_FILE_PATH = BASE_DIR / "ssl-keys" / "certificate.pem"
SSL_CERTIFICATE_PRIVATE_KEY_PEM_FILE_PATH = BASE_DIR / "ssl-keys" / "private_key.pem"

# Load models
whisper_model = whisper.load_model("base")  # Can use "small", "medium", or "large"
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # For similarity scoring
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Predefined questions and answers
PREDEFINED_QA = [
    {"question": "What is Lamba function?", "answer": "A lambda function is a concise way to represent a method using an expression, primarily used to implement functional interfaces"},
    {"question": "Why do you want this job?", "answer": "I am passionate about this role because it aligns with my skills and career goals."},
]

PREDEFINED_QA_FULL= "A lambda function is a concise way to represent a method using an expression, primarily used to implement functional interfaces. I am passionate about this role because it aligns with my skills and career goals."


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/upload', methods=['POST'])
def upload_video():
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # Extract audio
    audio_path = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(video.filename)[0]}.mp3")
    extract_audio(video_path, audio_path)

    # Transcribe audio
    transcription_path = os.path.join(TRANSCRIPTION_FOLDER, f"{os.path.splitext(video.filename)[0]}.txt")
    transcription = transcribe_audio(audio_path)
    with open(transcription_path, 'w') as f:
        f.write(transcription)

    # Score transcription
    scores = evaluate_responses(transcription)

    return jsonify({"scores": scores, "transcription": transcription})

def extract_audio(video_path, audio_path):
    """Extracts audio from video using FFmpeg."""
    
    ffmpeg.input(video_path).output(audio_path, ac=1).run(overwrite_output=True)

def transcribe_audio(audio_path):
    start_time = time.time()
    """Transcribes audio using Whisper."""
    result = whisper_model.transcribe(audio_path)
    end_time = time.time()
    print(f"Audio transcription completed in {end_time - start_time:.2f} seconds")
    return result['text']

def evaluate_responses(transcription):
    """Evaluates transcriptions against predefined answers and generates scores."""
    
    #candidate_responses = transcription.split(".")  # Assuming transcription has questions separated by newlines
    candidate_responses = sent_tokenize(transcription)
    scores = []
    
    # Preprocess before encoding
    preprocessed_predefined = preprocess_text(PREDEFINED_QA_FULL)
    preprocessed_transcription = preprocess_text(transcription)
    
    for i, qa in enumerate(PREDEFINED_QA):
        predefined_answer = qa['answer']
        if i < len(candidate_responses):
            candidate_answer = candidate_responses[i]
        else:
            candidate_answer = ""

        # Compute similarity
        predefined_embedding = embedding_model.encode(predefined_answer, convert_to_tensor=True)
        candidate_embedding = embedding_model.encode(candidate_answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(predefined_embedding, candidate_embedding).item()

        # Generate score (scale similarity to percentage)
        scores.append({"question": qa["question"], "score": round(similarity * 100, 2)})
        
    start_time = time.time()
    predefined_embedding = embedding_model.encode(preprocessed_predefined, convert_to_tensor=True)
    candidate_embedding = embedding_model.encode(preprocessed_transcription, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(predefined_embedding, candidate_embedding).item()
    scores.append({"Score calculated on whole text": round(similarity * 100, 2)})
    end_time = time.time()
    print(f"Scores calculation completed in {end_time - start_time:.2f} seconds")
    return scores

if len(sys.argv)>1:
    app_port_no = int(sys.argv[1])
else:
    app_port_no = 3000
    
if __name__ == '__main__':
    #app.run(debug=False, host="0.0.0.0", port=app_port_no, use_reloader=False)
    app.run(debug=False, host="0.0.0.0", port=app_port_no, use_reloader=False,ssl_context=(SSL_CERTIFICATE_SERVER_PEM_FILE_PATH, SSL_CERTIFICATE_PRIVATE_KEY_PEM_FILE_PATH))
