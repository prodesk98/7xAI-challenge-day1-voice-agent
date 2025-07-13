import asyncio
from io import BytesIO

import soundfile as sf
from dotenv import load_dotenv
import os

from groq import Groq
from elevenlabs import VoiceSettings

load_dotenv()

from elevenlabs.client import ElevenLabs

el_client = ElevenLabs(
  api_key=os.getenv("ELEVEN_LABS_API_KEY"),
)
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


def convert_audio_to_wav(audio: bytes) -> bytes:
    """
    Convert audio bytes to WAV format.
    """
    try:
        audio_stream = BytesIO(audio)
        data, samplerate = sf.read(audio_stream)
        wav_stream = BytesIO()
        sf.write(wav_stream, data, samplerate, format='WAV')
        wav_stream.seek(0)
        return wav_stream.getvalue()
    except Exception as e:
        print(f"Error in audio conversion: {e}")
        return b''

def stt(audio: bytes) -> str:
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=convert_audio_to_wav(audio),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        return transcription.text
    except Exception as e:
        print(f"Error in speech-to-text conversion: {e}")
        return ""

def tts(text: str) -> bytes:
    """
    Convert text to speech using ElevenLabs API.
    """
    try:
        audio = el_client.text_to_speech.convert(
            voice_id=os.getenv("ELEVEN_LABS_VOICE_ID"),
            model_id="eleven_flash_v2_5",
            output_format="mp3_22050_32",
            text=text,
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            )
        )
        return audio
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return b''


async def atts(text: str) -> bytes:
    """
    Asynchronous text to speech conversion.
    """
    return await asyncio.to_thread(tts, text)

async def astt(audio: bytes) -> str:
    """
    Asynchronous speech to text conversion.
    """
    return await asyncio.to_thread(stt, audio)


async def process(audio: bytes) -> bytes:
    # chunks = await asyncio.to_thread(split_audio_into_chunks, audio)
    # for chunk in chunks:
    #     # Enqueue the chunk for processing
    #     circular_buffer.append(chunk)
    # # Simulate processing by returning the last chunk as bytes
    # if circular_buffer:
    #     last_chunk = circular_buffer[-1]
    #     audio_stream = BytesIO()
    #     sf.write(audio_stream, last_chunk, SAMPLE_RATE, format='WAV')
    #     return audio_stream.getvalue()
    # return None
    transcription = await astt(audio)
    if transcription == "":
        print("No transcription available.")
        return b""
    print(transcription)
    return await atts(transcription)
