import uuid
import json
import os
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger
from openai import OpenAI
import io
import pickle
import shutil
import datetime
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter
import struct
import wave
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize speaker recognition model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

# Database paths and configuration
DB_DIR = "voice_database"
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.pkl")
MEMORIES_FILE = os.path.join(DB_DIR, "memories.json")
CHAT_SESSIONS_FILE = os.path.join(DB_DIR, "chat_sessions.json")
BACKUP_DIR = os.path.join(DB_DIR, "backups")

# Create database directories
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Audio processing parameters
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # Frame duration in milliseconds


def create_backup():
    """Create a timestamped backup of the database files"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create backup files with timestamp
        if os.path.exists(EMBEDDINGS_FILE):
            backup_file = os.path.join(BACKUP_DIR, f"embeddings_{timestamp}.pkl")
            shutil.copy2(EMBEDDINGS_FILE, backup_file)

        if os.path.exists(MEMORIES_FILE):
            backup_file = os.path.join(BACKUP_DIR, f"memories_{timestamp}.json")
            shutil.copy2(MEMORIES_FILE, backup_file)

        if os.path.exists(CHAT_SESSIONS_FILE):
            backup_file = os.path.join(BACKUP_DIR, f"chat_sessions_{timestamp}.json")
            shutil.copy2(CHAT_SESSIONS_FILE, backup_file)

        print(f"Created database backup with timestamp {timestamp}")

        # Clean up old backups (keep only last 5)
        cleanup_old_backups()
    except Exception as e:
        print(f"Error creating backup: {e}")


def cleanup_old_backups():
    """Remove old backups, keeping only the 5 most recent ones for each file type"""
    try:
        # Get and sort backup files by type and timestamp
        embeddings_backups = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.startswith("embeddings_")]
        )
        memories_backups = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.startswith("memories_")]
        )
        sessions_backups = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.startswith("chat_sessions_")]
        )

        # Remove old backups, keeping only the 5 most recent
        for backup_list in [embeddings_backups, memories_backups, sessions_backups]:
            if len(backup_list) > 5:
                for old_backup in backup_list[:-5]:
                    os.remove(os.path.join(BACKUP_DIR, old_backup))

    except Exception as e:
        print(f"Error cleaning up old backups: {e}")


# Audio preprocessing functions
def apply_bandpass_filter(audio_data, lowcut=300, highcut=3400, fs=16000, order=5):
    """Apply bandpass filter to focus on human voice frequencies"""
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, audio_data)
    except Exception as e:
        print(f"Error applying bandpass filter: {e}")
        return audio_data


def detect_voice_activity(audio_data, sample_rate=16000, energy_threshold=0.01):
    """
    Simple energy-based voice activity detection
    Since webrtcvad installation has issues, we'll use a simple energy-based detector
    """
    try:
        # Convert byte data to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
        else:
            audio_array = audio_data

        # Calculate short-time energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Calculate RMS energy
        energy = librosa.feature.rms(
            y=audio_array, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Determine if frames have voice based on energy threshold
        has_voice = energy > energy_threshold

        return has_voice
    except Exception as e:
        print(f"Error in voice activity detection: {e}")
        return np.array([True])  # Assume it's all speech if detection fails


def apply_noise_reduction(audio_data, sample_rate=16000, stationary=True):
    """Apply noise reduction to audio signal"""
    try:
        # Split audio into sections with and without speech based on VAD
        if len(audio_data) > sample_rate:  # At least 1 second of audio
            # Use the first 0.5 seconds for noise profile
            noise_sample = audio_data[: int(sample_rate * 0.5)]
            # Apply noise reduction
            return nr.reduce_noise(
                y=audio_data,
                y_noise=noise_sample,
                sr=sample_rate,
                stationary=stationary,
            )
        return audio_data
    except Exception as e:
        print(f"Error applying noise reduction: {e}")
        return audio_data


def transcribe_whisper(audio_recording):
    """Transcribe speech using OpenAI Whisper API with input validation"""
    try:
        if audio_recording is None or len(audio_recording) < 1000:
            print("Audio too short or empty for transcription")
            return "I couldn't understand what was said."

        audio_file = io.BytesIO(audio_recording)
        audio_file.name = (
            "audio.wav"  # Whisper requires a filename with a valid extension
        )

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            # language = ""  # specify Language explicitly
        )
        print(f"openai transcription: {transcription.text}")
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error during transcription. Please try again."


# Advanced voice isolation and enhancement functions
def enhance_voice_clarity(audio_data, sample_rate=16000):
    """Enhance voice clarity using spectral gating and equalization"""
    try:
        if audio_data.size == 0:
            return audio_data

        # Apply a mild high-shelf filter to enhance clarity
        # Simple equalization to boost frequencies in the 1-3 kHz range
        fft_data = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)

        # Create a simple eq curve that boosts 1-3 kHz range
        eq_curve = np.ones_like(freqs, dtype=float)

        # Find indices for 1kHz and 3kHz
        idx_1k = np.argmin(np.abs(freqs - 1000))
        idx_3k = np.argmin(np.abs(freqs - 3000))

        # Create a gentle boost (1.2x) in the 1-3kHz range
        eq_curve[idx_1k:idx_3k] = 1.2

        # Apply EQ
        fft_data *= eq_curve

        # Convert back to time domain
        enhanced_audio = np.fft.irfft(fft_data)

        # Normalize
        if np.max(np.abs(enhanced_audio)) > 0:
            enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95

        return enhanced_audio
    except Exception as e:
        print(f"Error enhancing voice clarity: {e}")
        return audio_data


def separate_main_voice(audio_bytes):
    """
    Isolate the main voice from background noise or other voices
    using a combination of techniques
    """
    try:
        # Convert WAV bytes to numpy array
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())

        # Convert byte frames to numpy array
        audio_data = np.frombuffer(frames, dtype=np.int16)

        # If stereo, convert to mono by averaging channels
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)

        # Convert to float for processing
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Apply bandpass filter focused on human voice range
        filtered_audio = apply_bandpass_filter(
            audio_float, lowcut=200, highcut=4000, fs=sample_rate
        )

        # Apply noise reduction with more aggressive settings
        denoised_audio = apply_noise_reduction(
            filtered_audio, sample_rate, stationary=False
        )

        # Enhance voice clarity
        enhanced_audio = enhance_voice_clarity(denoised_audio, sample_rate)

        # Convert back to int16
        processed_audio = (enhanced_audio * 32768.0).astype(np.int16)

        # Convert back to bytes
        out_buffer = io.BytesIO()
        with wave.open(out_buffer, "wb") as out_wav:
            out_wav.setnchannels(1)  # Always mono output
            out_wav.setsampwidth(2)  # 16-bit
            out_wav.setframerate(sample_rate)
            out_wav.writeframes(processed_audio.tobytes())

        return out_buffer.getvalue()
    except Exception as e:
        print(f"Error in voice separation: {e}")
        return audio_bytes  # Return original if processing fails


# Upgrade the main audio preprocessing pipeline to use all our techniques
def preprocess_audio(audio_bytes):
    """
    Enhanced audio preprocessing pipeline for voice isolation with performance optimization.
    Works with both WAV and raw PCM data.
    """
    try:
        start_time = datetime.datetime.now()

        # Check audio format and content
        if audio_bytes is None or len(audio_bytes) < 1000:
            print("Audio too short for preprocessing")
            return audio_bytes

        # Determine if we have WAV or raw PCM data
        is_wav = audio_bytes.startswith(b"RIFF")

        # Extract audio data for processing
        if is_wav:
            try:
                with io.BytesIO(audio_bytes) as wav_io:
                    with wave.open(wav_io, "rb") as wav_file:
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        frames = wav_file.readframes(wav_file.getnframes())

                # Convert to numpy array
                if channels > 1:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.reshape(-1, channels)
                    audio_data = np.mean(audio_data, axis=1).astype(np.int16)
                else:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
            except Exception as e:
                print(f"Error parsing WAV data: {e}")
                return audio_bytes
        else:
            # Assume raw PCM at 16kHz, 16-bit, mono
            try:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                sample_rate = 16000  # Assume 16kHz
                print(f"Processing raw PCM data: {len(audio_data)} samples")
            except Exception as e:
                print(f"Error parsing PCM data: {e}")
                return audio_bytes

        # Convert to float for processing
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Check for voice activity
        try:
            voice_activity = detect_voice_activity(audio_float, sample_rate)
            has_voice = np.mean(voice_activity) > 0.1

            if not has_voice:
                print("No significant voice activity detected")
                return audio_bytes
        except Exception as e:
            print(f"VAD check failed: {e}")
            # Continue with processing even if VAD check fails

        # Process audio to isolate voice
        try:
            # Apply bandpass filter focused on human voice range (optimize for speech)
            filtered_audio = apply_bandpass_filter(
                audio_float, lowcut=85, highcut=4000, fs=sample_rate
            )

            # Apply noise reduction with appropriate settings
            denoised_audio = apply_noise_reduction(
                filtered_audio, sample_rate, stationary=False
            )

            # Enhance voice clarity
            enhanced_audio = enhance_voice_clarity(denoised_audio, sample_rate)

            # Convert back to int16
            processed_audio = (enhanced_audio * 32768.0).astype(np.int16)

            # Convert back to WAV bytes
            out_buffer = io.BytesIO()
            with wave.open(out_buffer, "wb") as out_wav:
                out_wav.setnchannels(1)  # Always mono output
                out_wav.setsampwidth(2)  # 16-bit
                out_wav.setframerate(sample_rate)
                out_wav.writeframes(processed_audio.tobytes())

            processed_bytes = out_buffer.getvalue()

            # Measure processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Audio preprocessing completed in {processing_time:.3f} seconds")
            print(
                f"Original: {len(audio_bytes)} bytes → Processed: {len(processed_bytes)} bytes"
            )

            return processed_bytes
        except Exception as e:
            print(f"Error in voice processing: {e}")
            # If processing fails, create a basic WAV file with the original data
            if not is_wav:
                try:
                    out_buffer = io.BytesIO()
                    with wave.open(out_buffer, "wb") as out_wav:
                        out_wav.setnchannels(1)  # Mono
                        out_wav.setsampwidth(2)  # 16-bit
                        out_wav.setframerate(sample_rate)
                        out_wav.writeframes(audio_data.tobytes())
                    return out_buffer.getvalue()
                except:
                    pass
            return audio_bytes

    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        return audio_bytes  # Return original if processing fails


def init_database():
    """Initialize or load the database with improved error handling"""
    embeddings = {}
    memories = {}
    chat_sessions = {}

    # Load embeddings from pickle file
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                loaded_embeddings = pickle.load(f)
                # Validate and convert embeddings to proper tensor format
                for speaker_id, embedding in loaded_embeddings.items():
                    if isinstance(embedding, torch.Tensor):
                        # Ensure embedding is 2D and on CPU
                        if embedding.dim() == 3:
                            embedding = embedding.squeeze(1)
                        embedding = embedding.cpu()
                        embeddings[speaker_id] = embedding
            print(f"Loaded {len(embeddings)} speaker embeddings from database")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            # Create a backup of the corrupted file
            if os.path.exists(EMBEDDINGS_FILE):
                corrupted_file = f"{EMBEDDINGS_FILE}.corrupted"
                shutil.copy2(EMBEDDINGS_FILE, corrupted_file)
                print(f"Backed up corrupted embeddings file to {corrupted_file}")
            embeddings = {}

    # Load memories from JSON file with schema validation
    if os.path.exists(MEMORIES_FILE):
        try:
            with open(MEMORIES_FILE, "r") as f:
                loaded_memories = json.load(f)

                # Validate and convert memory format
                for speaker_id, speaker_memories in loaded_memories.items():
                    valid_memories = []
                    for memory in speaker_memories:
                        # Convert string memories to dict format
                        if isinstance(memory, str):
                            valid_memories.append(
                                {
                                    "text": memory,
                                    "sentiment": "neutral",
                                    "emotion": "neutral",
                                    "timestamp": str(uuid.uuid4()),
                                }
                            )
                        # Validate dict memories
                        elif isinstance(memory, dict) and "text" in memory:
                            # Ensure all required fields exist
                            memory_entry = {
                                "text": memory["text"],
                                "sentiment": memory.get("sentiment", "neutral"),
                                "emotion": memory.get("emotion", "neutral"),
                                "timestamp": memory.get("timestamp", str(uuid.uuid4())),
                            }
                            valid_memories.append(memory_entry)

                    if valid_memories:
                        memories[speaker_id] = valid_memories

            print(f"Loaded and validated memories for {len(memories)} speakers")
        except Exception as e:
            print(f"Error loading memories: {e}")
            # Create a backup of the corrupted file
            if os.path.exists(MEMORIES_FILE):
                corrupted_file = f"{MEMORIES_FILE}.corrupted"
                shutil.copy2(MEMORIES_FILE, corrupted_file)
                print(f"Backed up corrupted memories file to {corrupted_file}")
            memories = {}

    # Load chat sessions from JSON file
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r") as f:
                chat_sessions = json.load(f)
            print(f"Loaded {len(chat_sessions)} chat sessions from database")
        except Exception as e:
            print(f"Error loading chat sessions: {e}")
            # Create a backup of the corrupted file
            if os.path.exists(CHAT_SESSIONS_FILE):
                corrupted_file = f"{CHAT_SESSIONS_FILE}.corrupted"
                shutil.copy2(CHAT_SESSIONS_FILE, corrupted_file)
                print(f"Backed up corrupted chat sessions file to {corrupted_file}")
            chat_sessions = {}

    return embeddings, memories, chat_sessions


def save_database(embeddings, memories, chat_sessions):
    """Save the database to files with improved error handling and atomic writes"""
    # Create a backup before saving
    create_backup()

    # Save embeddings to pickle file
    try:
        # Ensure all embeddings are properly formatted before saving
        embeddings_to_save = {}
        for speaker_id, embedding in embeddings.items():
            if isinstance(embedding, torch.Tensor):
                # Ensure embedding is 2D and on CPU
                if embedding.dim() == 3:
                    embedding = embedding.squeeze(1)
                embedding = embedding.cpu()
                embeddings_to_save[speaker_id] = embedding

        # Use atomic write pattern
        temp_file = f"{EMBEDDINGS_FILE}.tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(embeddings_to_save, f)

        # Replace the old file with the new one
        shutil.move(temp_file, EMBEDDINGS_FILE)
        print(f"Saved {len(embeddings_to_save)} speaker embeddings to database")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        if os.path.exists(f"{EMBEDDINGS_FILE}.tmp"):
            os.remove(f"{EMBEDDINGS_FILE}.tmp")

    # Save memories to JSON file
    try:
        # Use atomic write pattern
        temp_file = f"{MEMORIES_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(memories, f, indent=2)

        # Replace the old file with the new one
        shutil.move(temp_file, MEMORIES_FILE)
        print(f"Saved memories for {len(memories)} speakers")
    except Exception as e:
        print(f"Error saving memories: {e}")
        if os.path.exists(f"{MEMORIES_FILE}.tmp"):
            os.remove(f"{MEMORIES_FILE}.tmp")

    # Save chat sessions to JSON file
    try:
        # Use atomic write pattern
        temp_file = f"{CHAT_SESSIONS_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(chat_sessions, f, indent=2)

        # Replace the old file with the new one
        shutil.move(temp_file, CHAT_SESSIONS_FILE)
        print(f"Saved {len(chat_sessions)} chat sessions")
    except Exception as e:
        print(f"Error saving chat sessions: {e}")
        if os.path.exists(f"{CHAT_SESSIONS_FILE}.tmp"):
            os.remove(f"{CHAT_SESSIONS_FILE}.tmp")


# Initialize database
voice_embeddings, voice_memories, chat_sessions = init_database()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

sessions = {}


def extract_embedding(audio_data):
    """Extract speaker embedding from audio data"""
    try:
        # Basic validation
        if len(audio_data) < 1000:
            print(f"Audio data too small: {len(audio_data)} bytes")
            return None

        # Preprocess audio for voice isolation and noise reduction
        processed_audio = preprocess_audio(audio_data)

        # Convert bytes to tensor
        audio_buffer = io.BytesIO(processed_audio)
        audio_buffer.seek(0)

        try:
            signal, fs = torchaudio.load(audio_buffer, format="wav")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

        # Resample if needed
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)

        # Basic validation
        if signal.shape[0] == 0 or signal.shape[1] == 0 or torch.all(signal == 0):
            print("Invalid signal detected")
            return None

        # Extract and return embedding
        embedding = verification.encode_batch(signal)
        if embedding.dim() == 3:
            embedding = embedding.squeeze(1)
        return embedding.cpu()
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def find_speaker(embedding, threshold=0.5):
    """Find if speaker exists in database"""
    if embedding is None:
        return None

    best_score = threshold
    best_speaker = None

    print(f"Checking against {len(voice_embeddings)} stored speakers")

    for speaker_id, stored_embedding in voice_embeddings.items():
        try:
            # Ensure stored embedding is 2D and on CPU
            if isinstance(stored_embedding, torch.Tensor):
                if stored_embedding.dim() == 3:
                    stored_embedding = stored_embedding.squeeze(1)
                stored_embedding = stored_embedding.cpu()

                # Compute cosine similarity between embeddings
                similarity = torch.nn.functional.cosine_similarity(
                    embedding, stored_embedding, dim=1
                )
                similarity_score = float(
                    similarity[0]
                )  # Get the first (and only) score
                print(f"Speaker {speaker_id} similarity: {similarity_score:.3f}")

                # Find the highest match above threshold
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_speaker = speaker_id
                    print(
                        f"New best match: {speaker_id} with score {similarity_score:.3f}"
                    )
        except Exception as e:
            print(f"Error comparing embeddings for speaker {speaker_id}: {e}")

    return best_speaker


def analyze_sentiment(text):
    """Simple sentiment analysis using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze sentiment. Respond with format: sentiment=positive|negative|neutral;emotion=happy|sad|angry|etc",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=30,
        )
        result = response.choices[0].message.content.strip().lower()

        # Parse result
        sentiment = "neutral"
        emotion = "neutral"

        if "sentiment=" in result:
            sentiment_part = result.split("sentiment=")[1].split(";")[0].strip()
            if sentiment_part in ["positive", "negative", "neutral"]:
                sentiment = sentiment_part

        if "emotion=" in result:
            emotion_part = result.split("emotion=")[1].split(";")[0].strip()
            if emotion_part and emotion_part != "neutral":
                emotion = emotion_part

        return {"sentiment": sentiment, "emotion": emotion}
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {"sentiment": "neutral", "emotion": "neutral"}


def summarize_memories(memories, speaker_id):
    """Summarize memories for efficient storage"""
    if not memories or len(memories) < 5:  # Only summarize if enough memories
        return memories

    try:
        # Join memory texts
        if isinstance(memories[0], dict):
            memory_texts = [m.get("text", "") for m in memories]
        else:
            memory_texts = memories

        all_memories = " ".join(memory_texts)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize these conversation memories preserving personal details, preferences, and key points.",
                },
                {"role": "user", "content": all_memories},
            ],
            max_tokens=300,
        )

        summary = response.choices[0].message.content.strip()

        # Keep latest memories and add summary
        if isinstance(memories[0], dict):
            summarized = [
                {
                    "text": f"MEMORY SUMMARY: {summary}",
                    "sentiment": "neutral",
                    "emotion": "neutral",
                }
            ]
            summarized.extend(memories[-2:])  # Keep last 2 memories
        else:
            summarized = [f"MEMORY SUMMARY: {summary}"] + memories[-2:]

        print(f"Summarized memories: {len(memories)} → {len(summarized)}")
        return summarized
    except Exception as e:
        print(f"Error summarizing: {e}")
        return memories


def extract_personal_info(memories):
    """Extract name and preferences from memories"""
    name = None
    order = None

    # Process each memory
    for memory in memories:
        # Handle string or dict
        text = memory.get("text", "") if isinstance(memory, dict) else memory
        text_lower = text.lower()

        # Extract name
        if "my name is" in text_lower:
            try:
                name_part = text_lower.split("my name is")[1].strip()
                name_end = min(
                    pos
                    for pos in [
                        name_part.find(".") if "." in name_part else len(name_part),
                        name_part.find(",") if "," in name_part else len(name_part),
                        name_part.find(" and ")
                        if " and " in name_part
                        else len(name_part),
                    ]
                    if pos > 0
                )
                name = name_part[:name_end].strip().title()
            except:
                pass

        # Extract order preferences
        order_indicators = [
            "i would like",
            "i want",
            "i'll have",
            "i will have",
            "order",
        ]
        for indicator in order_indicators:
            if indicator in text_lower:
                try:
                    order_part = text_lower.split(indicator, 1)[1]
                    order_end = min(
                        pos
                        for pos in [
                            order_part.find(".")
                            if "." in order_part
                            else len(order_part),
                            order_part.find("?")
                            if "?" in order_part
                            else len(order_part),
                        ]
                        if pos > 0
                    )
                    order = order_part[:order_end].strip()
                except:
                    pass
                break

    return name, order


# def transcribe_preview(session):
#     if session["audio_buffer"] is not None:
#         text = transcribe_whisper(session["audio_buffer"])
#         # send transcription
#         ws = session.get("websocket")
#         if ws:
#             message = {
#                 "event": "recognizing",
#                 "text": text,
#                 "language": session["language"]
#             }
#             ws.send(json.dumps(message))


@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """
    Open a new voice input session and start continuous recognition.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - language
          properties:
            language:
              type: string
              description: Language code for speech recognition (e.g., en-US)
    responses:
      200:
        description: Session created successfully
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Unique identifier for the voice recognition session
      400:
        description: Language parameter missing
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    session_id = str(uuid.uuid4())

    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    language = body["language"]

    # Store session in both active sessions and chat sessions database
    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": language,
        "websocket": None,
        "speaker_id": None,
    }

    # Initialize chat session in database if it doesn't exist
    if chat_session_id not in chat_sessions:
        chat_sessions[chat_session_id] = {
            "sessions": [],
            "speaker_id": None,
            "language": language,
        }

    # Add this session to the chat session's list
    chat_sessions[chat_session_id]["sessions"].append(session_id)

    # Save updated database
    save_database(voice_embeddings, voice_memories, chat_sessions)

    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk received successfully
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()  # raw binary data from the POST body

    # Add validation to check if this is valid WAV data
    if audio_data and len(audio_data) >= 12:
        # Check for RIFF header
        has_riff_header = audio_data.startswith(b"RIFF")
        if has_riff_header:
            # Process for voice activity detection
            try:
                with io.BytesIO(audio_data) as wav_io:
                    with wave.open(wav_io, "rb") as wav_file:
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        frames = wav_file.readframes(wav_file.getnframes())

                # Convert to numpy array for VAD
                if channels > 1:
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    audio_array = audio_array.reshape(-1, channels)
                    audio_array = np.mean(audio_array, axis=1).astype(np.int16)
                else:
                    audio_array = np.frombuffer(frames, dtype=np.int16)

                # Convert to float for processing
                audio_float = audio_array.astype(np.float32) / 32768.0

                voice_activity = detect_voice_activity(audio_float, sample_rate)
                has_voice = (
                    np.mean(voice_activity) > 0.2
                )  # Consider as speech if more than 20% is detected as voice

                if has_voice:
                    if sessions[session_id]["audio_buffer"] is not None:
                        sessions[session_id]["audio_buffer"] = (
                            sessions[session_id]["audio_buffer"] + audio_data
                        )
                    else:
                        sessions[session_id]["audio_buffer"] = audio_data

                    print(
                        f"Added audio chunk with voice activity detected: {len(audio_data)} bytes"
                    )
                else:
                    print("Skipped audio chunk: no significant voice activity")
                return jsonify({"status": "audio_chunk_received"})
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
        else:
            print("Audio chunk missing RIFF header - may be raw PCM data")

    # Fallback - just add the data to the buffer without validation
    # This handles non-WAV formatted audio data
    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] = (
            sessions[session_id]["audio_buffer"] + audio_data
        )
    else:
        sessions[session_id]["audio_buffer"] = audio_data

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """Close the session and process audio"""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    if sessions[session_id]["audio_buffer"] is not None:
        start_time = datetime.datetime.now()

        audio_buffer = sessions[session_id]["audio_buffer"]

        # Check if the audio buffer has a valid WAV header
        is_valid_wav = audio_buffer.startswith(b"RIFF") if audio_buffer else False

        if not is_valid_wav:
            print(
                "Audio buffer doesn't have a valid WAV header - attempting to convert raw audio"
            )
            # Try to convert raw PCM to WAV format
            try:
                # Assuming 16kHz, 16-bit, mono audio (common for speech)
                pcm_data = np.frombuffer(audio_buffer, dtype=np.int16)

                # Create a WAV header and file
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    wav_file.writeframes(pcm_data.tobytes())

                # Replace the buffer with properly formatted WAV data
                sessions[session_id]["audio_buffer"] = wav_buffer.getvalue()
                print(f"Converted {len(audio_buffer)} bytes of raw audio to WAV format")
            except Exception as e:
                print(f"Error converting raw audio to WAV: {e}")

        # Process speaker identification
        # Apply full preprocessing pipeline to the accumulated audio buffer
        try:
            processed_audio = preprocess_audio(sessions[session_id]["audio_buffer"])
            sessions[session_id]["processed_audio"] = (
                processed_audio  # Store processed audio
            )

            embedding = extract_embedding(processed_audio)
            if embedding is None:
                return jsonify({"error": "Failed to process audio"}), 500

            # Identify speaker
            speaker_id = find_speaker(embedding)
            if speaker_id is None:
                speaker_id = str(uuid.uuid4())
                voice_embeddings[speaker_id] = embedding
                voice_memories[speaker_id] = []

            # Update session info
            sessions[session_id]["speaker_id"] = speaker_id
            chat_sessions[chat_session_id]["speaker_id"] = speaker_id

            # Get transcription
            text = transcribe_whisper(sessions[session_id]["processed_audio"])
            if not text:
                text = "I couldn't understand what was said."

            # Analyze sentiment and add to memory in one step
            sentiment_data = analyze_sentiment(text)

            # Create memory entry
            memory_entry = {
                "text": text,
                "sentiment": sentiment_data["sentiment"],
                "emotion": sentiment_data["emotion"],
                "timestamp": str(uuid.uuid4()),
            }

            # Update memories (convert old format if needed)
            memories = voice_memories.get(speaker_id, [])
            updated_memories = []

            for memory in memories:
                if isinstance(memory, str):
                    updated_memories.append(
                        {"text": memory, "sentiment": "neutral", "emotion": "neutral"}
                    )
                else:
                    updated_memories.append(memory)

            updated_memories.append(memory_entry)

            # Summarize if needed
            if len(updated_memories) > 5:
                updated_memories = summarize_memories(updated_memories, speaker_id)

            voice_memories[speaker_id] = updated_memories

            # Extract personal info
            memory_texts = [
                m.get("text", "") if isinstance(m, dict) else m
                for m in updated_memories
            ]
            name, order = extract_personal_info(updated_memories)

            # Create memory context
            memory_context = "Previous conversation: " + " ".join(memory_texts)

            # Add personalized prefix
            personalized_prefix = ""
            if name or order or sentiment_data["emotion"] != "neutral":
                personalized_prefix = "USER INFORMATION: "
                if name:
                    personalized_prefix += f"Name: {name}. "
                if order:
                    personalized_prefix += f"Order preference: {order}. "
                if sentiment_data["emotion"] != "neutral":
                    personalized_prefix += f"Mood: {sentiment_data['emotion']}. "

                memory_context = f"{personalized_prefix}\n\n{memory_context}"

            # Calculate total processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Total audio processing completed in {processing_time:.3f} seconds")

            # Save database
            save_database(voice_embeddings, voice_memories, chat_sessions)

            # Send transcription with processing time info
            ws = sessions[session_id].get("websocket")
            if ws:
                ws.send(
                    json.dumps(
                        {
                            "event": "recognized",
                            "text": text,
                            "language": sessions[session_id]["language"],
                            "speaker_id": speaker_id,
                            "memory_context": memory_context,
                            "sentiment": sentiment_data["sentiment"],
                            "emotion": sentiment_data["emotion"],
                            "processing_time": f"{processing_time:.3f}",
                            "voice_isolation": "enabled",
                        }
                    )
                )
        except Exception as e:
            print(f"Error processing audio: {e}")
            return jsonify({"error": f"Error processing audio: {e}"}), 500

    # Cleanup
    sessions.pop(session_id, None)
    if (
        chat_session_id in chat_sessions
        and session_id in chat_sessions[chat_session_id]["sessions"]
    ):
        chat_sessions[chat_session_id]["sessions"].remove(session_id)
        save_database(voice_embeddings, voice_memories, chat_sessions)

    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """
    WebSocket endpoint for clients to receive STT results.

    This WebSocket allows clients to connect and receive speech-to-text (STT) results
    in real time. The connection is maintained until the client disconnects. If the
    session ID is invalid, an error message is sent, and the connection is closed.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the chat session.
      - name: session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the speech session.
    responses:
      400:
        description: Session not found.
      101:
        description: WebSocket connection established.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    # Store the websocket reference in the session
    sessions[session_id]["websocket"] = ws

    # Keep the socket open to send events
    # Typically we'd read messages from the client in a loop if needed
    while True:
        # If the client closes the socket, an exception is thrown or `ws.receive()` returns None
        msg = ws.receive()
        if msg is None:
            break


@app.route("/chats/<chat_session_id>/set-memories", methods=["POST"])
def set_memories(chat_session_id):
    """Store memories from chat history"""
    chat_history = request.get_json()
    if not chat_history or not isinstance(chat_history, list) or len(chat_history) == 0:
        return jsonify({"error": "Invalid chat history"}), 400

    latest_message = chat_history[-1].get("text", "")

    # Find associated speaker
    if chat_session_id in chat_sessions:
        speaker_id = chat_sessions[chat_session_id].get("speaker_id")
        if speaker_id and speaker_id in voice_memories:
            # Analyze sentiment and create memory
            sentiment_data = analyze_sentiment(latest_message)
            memory_entry = {
                "text": latest_message,
                "sentiment": sentiment_data["sentiment"],
                "emotion": sentiment_data["emotion"],
                "timestamp": str(uuid.uuid4()),
            }

            # Update memories
            current_memories = voice_memories[speaker_id]
            updated_memories = []

            for memory in current_memories:
                if isinstance(memory, str):
                    updated_memories.append(
                        {"text": memory, "sentiment": "neutral", "emotion": "neutral"}
                    )
                else:
                    updated_memories.append(memory)

            updated_memories.append(memory_entry)

            # Summarize if needed
            if len(updated_memories) > 5:
                updated_memories = summarize_memories(updated_memories, speaker_id)

            voice_memories[speaker_id] = updated_memories
            save_database(voice_embeddings, voice_memories, chat_sessions)

            return jsonify(
                {"success": "1", "speaker_id": speaker_id, "sentiment": sentiment_data}
            )

    return jsonify({"success": "0", "error": "No associated speaker found"})


@app.route("/chats/<chat_session_id>/get-memories", methods=["GET"])
def get_memories(chat_session_id):
    """Retrieve memories for a chat session"""
    if chat_session_id not in chat_sessions:
        return jsonify({"error": "Chat session not found"}), 404

    speaker_id = chat_sessions[chat_session_id].get("speaker_id")
    if not speaker_id:
        return jsonify({"memories": "No speaker identified", "personalized": False})

    memories = voice_memories.get(speaker_id, [])
    if not memories:
        return jsonify({"memories": "No memories available", "personalized": False})

    # Extract memory info
    memory_texts = [m.get("text", "") if isinstance(m, dict) else m for m in memories]
    name, order = extract_personal_info(memories)

    # Build memory text
    memory_text = " ".join(memory_texts)

    # Get emotion info
    emotion = "neutral"
    emotions = [
        m.get("emotion") for m in memories if isinstance(m, dict) and "emotion" in m
    ]
    if emotions:
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        emotion = max(emotion_counts, key=emotion_counts.get)

    # Build personalized context
    personalized = bool(name or order or emotion != "neutral")
    personalized_context = ""

    if personalized:
        personalized_context = "USER INFORMATION: "
        if name:
            personalized_context += f"Name: {name}. "
        if order:
            personalized_context += f"Order preference: {order}. "
        if emotion != "neutral":
            personalized_context += f"Mood: {emotion}. "

    response = {
        "memories": f"{personalized_context}\n\nPrevious conversation: {memory_text}"
        if personalized
        else memory_text,
        "speaker_id": speaker_id,
        "personalized": personalized,
        "emotion": emotion,
    }

    if personalized:
        response["name"] = name
        response["previous_order"] = order
        response["personalized_context"] = personalized_context

    return jsonify(response)


if __name__ == "__main__":
    # In production, you would use a real WSGI server like gunicorn/uwsgi
    app.run(debug=True, host="0.0.0.0", port=5000)
