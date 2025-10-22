import json
import uuid
import httpx
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session
import google.generativeai as genai

import tts_service
import audio_sourcing_service
from tts_models import SpeechRequest, OpenAITTSConfig
from main import ImageGenRequest, generate_image_endpoint # Allow internal calls

# Load prompts at startup
PROMPTS = {}
try:
    with open("prompts.json", "r") as f:
        PROMPTS = json.load(f)
except FileNotFoundError:
    print("--- CRITICAL: prompts.json not found. Dialog features will fail.")

class AudioDramaGenerator:
    def __init__(self, request: SpeechRequest, api_key: str, db_session: Session):
        self.request = request
        self.api_key = api_key
        self.db = db_session
        self.temp_dir = Path(f"/tmp/audio_drama_{uuid.uuid4()}")
        self.temp_dir.mkdir()
        self.production_plan = {}
        self.generated_image_obj = None

    async def generate(self):
        # 1. Analyze script
        self.production_plan = await self._analyze_script()
        dialog_segments = self.production_plan.get("dialog", [])
        sfx_cues = self.production_plan.get("sfx", [])
        music_cues = self.production_plan.get("music", [])

        # 2. Source all audio assets
        dialog_chunks_with_paths = await self._generate_dialog_chunks(dialog_segments)
        
        music_path = None
        if self.request.config.add_music and music_cues:
            music_path = await self._source_music(music_cues)
            
        sfx_paths = []
        if self.request.config.add_sfx and sfx_cues:
            sfx_paths = await self._source_sfx(sfx_cues)

        # 3. Mix the final audio track
        final_audio_bytes = tts_service.mix_final_audio(
            dialog_chunks=dialog_chunks_with_paths,
            music_path=music_path,
            sfx_tracks=sfx_paths,
            output_format=self.request.config.output_format
        )

        # 4. (Optional) Generate title image
        if self.request.config.generate_title_image:
            self.generated_image_obj = await self._generate_title_image()

        return final_audio_bytes

    async def _analyze_script(self):
        print("--- Audio Drama: Analyzing script with Gemini...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = await model.generate_content_async(analysis_prompt)
        
        try:
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse production plan from Gemini. Raw response: {response.text}. Error: {e}")

    async def _generate_dialog_chunks(self, dialog_segments):
        dialog_chunks = []
        voice_mapping = self.request.config.voice_mapping or {}
        default_voices = {
            "male": "onyx", "female": "nova", "ai": "onyx" if self.request.config.ai_gender == 'male' else "nova", "narrator": "shimmer"
        }
        silence_chunk_path = tts_service.create_silence_chunk(1000, self.request.config.output_format, self.temp_dir)

        for i, segment in enumerate(dialog_segments):
            speaker = segment.get("speaker", f"Person {i+1}")
            text = segment.get("text", "")
            gender = segment.get("gender", "female")
            voice = voice_mapping.get(speaker, default_voices.get(gender, "nova"))
            
            config = OpenAITTSConfig(voice=voice, speed=self.request.content.speed, output_format=self.request.config.output_format)
            audio_bytes = await tts_service.generate_openai_tts(text, config)
            
            chunk_path = self.temp_dir / f"segment_{i}.{self.request.config.output_format}"
            chunk_path.write_bytes(audio_bytes)
            dialog_chunks.append({'path': chunk_path})
            
            if i < len(dialog_segments) - 1:
                dialog_chunks.append({'path': silence_chunk_path})
        
        return dialog_chunks

    async def _source_music(self, music_cues):
        music_url = await audio_sourcing_service.find_music_on_freesound(music_cues[0]['description'])
        if music_url:
            async with httpx.AsyncClient() as client:
                r = await client.get(music_url, follow_redirects=True)
                music_path = self.temp_dir / "music.mp3"
                music_path.write_bytes(r.content)
                return music_path
        return None

    async def _source_sfx(self, sfx_cues):
        # This is a placeholder for a more complex implementation
        # that would download multiple SFX and time them correctly.
        return []

    async def _generate_title_image(self):
        print("--- Audio Drama: Requesting title image prompt from Gemini...")
        image_prompt_request = PROMPTS.get("image_prompt_generation", "").format(dialog_text=self.request.content.text)
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        image_prompt_response = await model.generate_content_async(image_prompt_request)
        image_prompt = image_prompt_response.text.strip()
        
        print(f"--- Audio Drama: Received image prompt: '{image_prompt}'")
        
        return await generate_image_endpoint(
            ImageGenRequest(prompt=image_prompt, link_id=self.request.id, owner_user_id=None),
            self.api_key, 
            self.db
        )
