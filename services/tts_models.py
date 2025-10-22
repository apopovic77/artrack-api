from pydantic import BaseModel, Field
from typing import Literal

# --- Pydantic Models for TTS Configuration ---
# Moved here to prevent circular imports

class OpenAITTSConfig(BaseModel):
    model: str = Field("tts-1", description="The model to use, e.g., 'tts-1' or 'tts-1-hd'")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Field("nova", description="The voice to use for the audio")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="The speaking rate, from 0.25 to 4.0")
    output_format: Literal["mp3", "opus", "aac", "flac"] = Field("mp3", description="The output format for the audio")

class ElevenLabsTTSConfig(BaseModel):
    model_id: str = Field("eleven_multilingual_v2", description="The model to use, e.g., 'eleven_multilingual_v2'")
    voice_id: str = Field("JBFqnCBsd6RMkjVDRZzb", description="The ID of the voice to use")
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Voice stability, from 0.0 to 1.0")
    clarity: float = Field(0.75, ge=0.0, le=1.0, description="Voice clarity/similarity, from 0.0 to 1.0")

class TTSContent(BaseModel):
    text: str
    language: str = "de-DE"
    voice: str = "nova"
    pitch: float = 0.0
    speed: float = 1.0
    stability: float | None = None
    clarity: float | None = None
    style: str | None = None # Add back for robust validation

class TTSConfig(BaseModel):
    provider: str = "openai"
    output_format: str = "mp3"
    openai: OpenAITTSConfig | None = None
    elevenlabs: ElevenLabsTTSConfig | None = None
    # For dialog generation
    voice_mapping: dict[str, str] | None = None
    dialog_mode: bool = False
    ai_gender: Literal['male', 'female'] = 'female'
    narrator_gender: Literal['male', 'female'] | None = None
    analysis_user_hint: str | None = None
    # Optional: provide a full production plan to skip AI analysis
    analysis_override: dict | None = None
    generate_title_image: bool = False
    add_sfx: bool = False
    add_music: bool = False
    manual_music_storage_id: str | None = None
    # Music mixing behavior
    music_loop: bool = True
    music_stop_at_end: bool = False
    analyze_only: bool = False

class SpeechRequest(BaseModel):
    id: str
    type: str = "speech_request"
    timestamp: str
    content: TTSContent
    config: TTSConfig
    save_options: dict | None = {"is_public": False, "owner_email": None}
    collection_id: str | None = None # New field for dynamic collection name
