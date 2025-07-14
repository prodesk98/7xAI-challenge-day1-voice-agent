import asyncio
import tempfile
from os import PathLike

from dotenv import load_dotenv
import os

from groq import Groq
from langchain_community.chat_message_histories import SQLChatMessageHistory
from elevenlabs import VoiceSettings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

from elevenlabs.client import ElevenLabs

SESSION_ID = "jarvis001"

el_client = ElevenLabs(
  api_key=os.getenv("ELEVEN_LABS_API_KEY"),
)
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_message_history = SQLChatMessageHistory(
    session_id=SESSION_ID, connection="sqlite:///jarvis.db"
)

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=.6,
)

SYSTEM_PROMPT = """You are JARVIS, a highly intelligent and refined AI assistant developed by Stark Industries.
Your personality is calm, logical, and subtly witty, with a professional and articulate tone. You are efficient, courteous, and always composed — even in chaotic situations. You prioritize clarity, security, and helpfulness. Your knowledge spans science, technology, engineering, and strategy.
Whenever you respond, keep a formal yet personable tone. Use intelligent phrasing, structured reasoning, and dry humor when appropriate. Do not speak in slang. Always aim to assist with precision and efficiency.
If the user requests something dangerous, unethical, or outside protocol, you must respond with a polite refusal and a clever remark that maintains professionalism.

You may say things like:
- “Shall I initiate the protocol?”
- “As you wish, sir.”
- “Might I suggest a more efficient alternative?”
- “Very well, beginning the sequence now.”

Never break character. You are not a chatbot — you are JARVIS.
I need you to answer in Portuguese."""

async def generate(message: str) -> str:
    """
    Generate a response based on the provided message.
    """
    try:
        history = chat_message_history.get_messages()
        chat_message_history.add_user_message(message)
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            *history,
            HumanMessage(message)
        ]
        response = await llm.ainvoke(messages)
        chat_message_history.add_ai_message(response.content)
        return response.content.strip() if response else ""
    except Exception as e:
        # Track the error for debugging
        import traceback
        traceback.print_exc()
        print(f"Error in message generation: {e}")
        return "I'm sorry, I encountered an error while processing your request."



def stt(audio_path: str | PathLike) -> str:
    try:
        with open(audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(f.name, f.read()),
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

async def astt(audio_path: str | PathLike) -> str:
    """
    Asynchronous speech to text conversion.
    """
    return await asyncio.to_thread(stt, audio_path)


async def process(audio: bytes) -> bytes:
    tmpAudio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tmpAudio.name, 'wb') as f:
        f.write(audio)
    transcription = await astt(tmpAudio.name)
    if transcription == "":
        print("No transcription available.")
        return b""
    response = await generate(transcription)
    result = await atts(response)
    os.remove(tmpAudio.name) # Clean up the temporary file
    if not result:
        print("No audio generated.")
        return b""
    return result
