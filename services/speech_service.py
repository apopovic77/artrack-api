import uuid
from pathlib import Path
from sqlalchemy.orm import Session
import google.generativeai as genai
from fastapi import HTTPException

import tts_service
from tts_models import SpeechRequest, OpenAITTSConfig, ElevenLabsTTSConfig
from artrack.storage_domain import save_file_and_record

class SpeechGenerator:
    def __init__(self, request: SpeechRequest, api_key: str, db_session: Session, image_gen_func):
        self.request = request
        self.api_key = api_key
        self.db = db_session
        self.temp_dir = Path(f"/tmp/speech_gen_{uuid.uuid4()}")
        self.temp_dir.mkdir()
        self.generate_image_endpoint = image_gen_func

    async def generate(self):
        audio_bytes = await self._generate_speech()
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS generation failed, no audio data produced.")

        saved_obj = await self._save_audio(audio_bytes)
        
        generated_image_obj = None
        if self.request.config.generate_title_image:
            generated_image_obj = await self.generate_title_image()
            
        return saved_obj, None, generated_image_obj

    async def _generate_speech(self):
        provider = self.request.config.provider
        content = self.request.content
        
        if provider == "openai":
            config = self.request.config.openai or OpenAITTSConfig()
            config.voice = content.voice
            config.speed = content.speed
            config.output_format = self.request.config.output_format
            return await tts_service.generate_openai_tts(content.text, config)

        elif provider == "gemini":
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechAsyncClient()
            synthesis_input = texttospeech.SynthesisInput(text=content.text)
            voice = texttospeech.VoiceSelectionParams(language_code=content.language, name=content.voice)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=content.speed, pitch=content.pitch)
            response = await client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            return response.audio_content

        elif provider == "elevenlabs":
            config = self.request.config.elevenlabs or ElevenLabsTTSConfig()
            config.voice_id = content.voice
            if content.stability is not None:
                config.stability = content.stability
            if content.clarity is not None:
                config.clarity = content.clarity
            return await tts_service.generate_elevenlabs_tts(content.text, config)
        
        return None

    async def generate_title_image(self):
        print("--- Image Gen: Requesting title image prompt from Gemini...")
        
        image_prompt_request = (
            "Based on the following text, create a short, visually descriptive prompt for a text-to-image AI. "
            "The prompt should capture the essence and mood of the text in a single scene. "
            "Describe the scene, characters, and atmosphere. Return only the prompt text."
            f"\n\nTEXT:\n{self.request.content.text}"
        )
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        image_prompt_response = await model.generate_content_async(image_prompt_request)
        image_prompt = image_prompt_response.text.strip()
        
        print(f"--- Image Gen: Received prompt: '{image_prompt}'")
        
        from main import ImageGenRequest
        # Ensure image uses same collection as audio; default to ai_hörbuch for dialog
        image_collection = getattr(self.request, 'collection_id', None) or "ai_hörbuch"
        return await self.generate_image_endpoint(
            ImageGenRequest(
                prompt=image_prompt,
                link_id=self.request.id,
                owner_user_id=None,
                collection_id=image_collection
            ),
            self.api_key, 
            self.db
        )

    async def _save_audio(self, audio_bytes: bytes):
        save_opts = self.request.save_options or {}
        owner_email = save_opts.get("owner_email")
        target_owner_id = None
        if owner_email:
            # This logic would need the User model to be available here
            pass

        # Determine collection to save into: prefer request.collection_id, fallback to default
        collection_id = self.request.collection_id or "ai_hörbuch"

        filename = f"tts_{self.request.id}_{self.request.config.provider}.{self.request.config.output_format}"

        # Use link_id from save_options if provided, otherwise fallback to request.id
        link_id = save_opts.get("link_id") or self.request.id

        # Store spoken text in title (first 200 chars) and description (full text)
        spoken_text = self.request.content.text or ""
        title = spoken_text[:200] if spoken_text else None
        description = spoken_text if spoken_text else None

        return await save_file_and_record(
            db=self.db, owner_user_id=target_owner_id, data=audio_bytes,
            original_filename=filename, context="tts-generation",
            is_public=save_opts.get("is_public", False),
            collection_id=collection_id,
            link_id=link_id,
            title=title,
            description=description
        )
