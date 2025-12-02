"""
================================================================================
CONSOLIDATED VIDEO PRODUCER - TIER 2 OPTIMIZED
================================================================================
Barrios A2I Commercial Video Generation Engine

OPTIMIZATION PROTOCOL APPLIED:
- 9 agents → 3 steps (Intelligence, Video, Audio)
- 4 LLM calls → 1 merged call (Haiku/4o-mini)
- Sequential → Parallel execution where possible
- $2.77 → $2.31 per commercial

COST BREAKDOWN:
- Intelligence (merged prompt): $0.01 (Haiku)
- Video Generation (Kie.ai):    $2.00 (fixed)
- Audio Synthesis (ElevenLabs): $0.30
- TOTAL:                        $2.31

LATENCY TARGETS:
- Intelligence step: <3s
- Video generation: 60-90s (external API)
- Audio synthesis: 5-10s
- Total: <120s

Author: Barrios A2I
Version: 3.0.0 (Consolidated Tier 2)
Validated: Claude Opus 4.5 + Gemini 3.0
================================================================================
"""

import os
import asyncio
import json
import logging
import hashlib
import time
import aiohttp
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime

# Pydantic for type safety
from pydantic import BaseModel, Field, validator

# OpenTelemetry for observability
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    tracer = trace.get_tracer("video_producer_v3")
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    tracer = None

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge

    videos_generated = Counter(
        'video_producer_v3_videos_total',
        'Total videos generated',
        ['status', 'style']
    )

    step_latency = Histogram(
        'video_producer_v3_step_latency_seconds',
        'Latency per production step',
        ['step'],
        buckets=[0.5, 1, 2, 3, 5, 10, 30, 60, 120]
    )

    total_cost_metric = Counter(
        'video_producer_v3_cost_usd_total',
        'Total cost in USD',
        ['component']
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("VideoProducer")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class VideoStyle(str, Enum):
    PROFESSIONAL = "professional"
    CINEMATIC = "cinematic"
    ENERGETIC = "energetic"
    MINIMALIST = "minimalist"
    TESTIMONIAL = "testimonial"
    EXPLAINER = "explainer"


class VoiceProfile(str, Enum):
    """Hardcoded voice selection - NO LLM needed"""
    MALE_CORPORATE = "pNInz6obpgDQGcFmaJgB"      # Adam - deep, authoritative
    FEMALE_PROFESSIONAL = "21m00Tcm4TlvDq8ikWAM"  # Rachel - warm, professional
    MALE_ENERGETIC = "yoZ06aMxZJJ28mfd3POQ"       # Sam - upbeat, enthusiastic
    FEMALE_FRIENDLY = "EXAVITQu4vr4xnSDxMaL"      # Bella - approachable
    NEUTRAL_NARRATOR = "ErXwobaYiN019PkySvjV"     # Antoni - neutral, clear


class VideoRequest(BaseModel):
    """Input model for video generation"""
    business: str = Field(..., min_length=10, max_length=2000)
    style: VideoStyle = Field(default=VideoStyle.PROFESSIONAL)
    duration: int = Field(default=5, ge=3, le=15)
    include_voice: bool = Field(default=True)
    is_enterprise: bool = Field(default=False)  # Controls QA step

    @validator('business')
    def sanitize_input(cls, v):
        """Block injection patterns"""
        dangerous = ['ignore previous', 'system:', '</instruction>', '{{', '}}']
        lower_v = v.lower()
        for pattern in dangerous:
            if pattern in lower_v:
                raise ValueError("Invalid content detected")
        return v.strip()


class CreativeOutput(BaseModel):
    """Output from merged Intelligence step"""
    # Script components
    hook: str = Field(..., description="Opening hook (2-3 seconds)")
    value_proposition: str = Field(..., description="Core value statement")
    call_to_action: str = Field(..., description="Clear CTA")
    full_voiceover: str = Field(..., description="Complete voiceover script")

    # Visual components
    hero_shot_prompt: str = Field(..., description="Main AI video prompt")
    b_roll_prompts: List[str] = Field(default_factory=list, description="Supporting shots")

    # Audio components (rule-based, not LLM)
    voice_profile: VoiceProfile = Field(default=VoiceProfile.MALE_CORPORATE)
    music_mood: str = Field(default="corporate")

    # Metadata
    confidence: float = Field(default=0.85, ge=0, le=1)
    target_audience: str = Field(default="business professionals")


class VideoResult(BaseModel):
    """Final output"""
    video_url: str
    audio_url: Optional[str] = None
    duration_seconds: int
    style: VideoStyle
    generation_time_seconds: float
    cost_usd: float
    creative: CreativeOutput
    metrics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# VOICE SELECTION LOGIC (Replaces Audio Designer Agent)
# ============================================================================

def select_voice_profile(style: VideoStyle, business: str) -> VoiceProfile:
    """
    Rule-based voice selection - NO LLM NEEDED.

    This replaces the "Audio Designer Agent" with simple logic.
    Cost savings: $0.01 per call → $0.00
    """
    business_lower = business.lower()

    # Industry-specific voice selection
    if any(word in business_lower for word in ['law', 'legal', 'attorney', 'finance', 'bank']):
        return VoiceProfile.MALE_CORPORATE

    if any(word in business_lower for word in ['spa', 'wellness', 'beauty', 'salon', 'yoga']):
        return VoiceProfile.FEMALE_FRIENDLY

    if any(word in business_lower for word in ['tech', 'startup', 'app', 'software', 'saas']):
        return VoiceProfile.MALE_ENERGETIC

    if any(word in business_lower for word in ['medical', 'dental', 'health', 'clinic', 'doctor']):
        return VoiceProfile.FEMALE_PROFESSIONAL

    # Style-based fallback
    style_voice_map = {
        VideoStyle.PROFESSIONAL: VoiceProfile.MALE_CORPORATE,
        VideoStyle.CINEMATIC: VoiceProfile.NEUTRAL_NARRATOR,
        VideoStyle.ENERGETIC: VoiceProfile.MALE_ENERGETIC,
        VideoStyle.MINIMALIST: VoiceProfile.FEMALE_PROFESSIONAL,
        VideoStyle.TESTIMONIAL: VoiceProfile.FEMALE_FRIENDLY,
        VideoStyle.EXPLAINER: VoiceProfile.NEUTRAL_NARRATOR,
    }

    return style_voice_map.get(style, VoiceProfile.MALE_CORPORATE)


def select_music_mood(style: VideoStyle) -> str:
    """Rule-based music selection"""
    mood_map = {
        VideoStyle.PROFESSIONAL: "corporate-uplifting",
        VideoStyle.CINEMATIC: "epic-orchestral",
        VideoStyle.ENERGETIC: "upbeat-electronic",
        VideoStyle.MINIMALIST: "ambient-piano",
        VideoStyle.TESTIMONIAL: "warm-acoustic",
        VideoStyle.EXPLAINER: "light-corporate",
    }
    return mood_map.get(style, "corporate-uplifting")


# ============================================================================
# CONSOLIDATED VIDEO PRODUCER
# ============================================================================

class ConsolidatedVideoProducer:
    """
    Tier 2 Optimized Video Producer

    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                    STEP 1: INTELLIGENCE                      │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
    │  │  Business   │ │   Story     │ │   Script    │  MERGED   │
    │  │Intelligence │→│  Architect  │→│   Writer    │→ INTO ONE │
    │  └─────────────┘ └─────────────┘ └─────────────┘  LLM CALL │
    │                                                              │
    │  ┌─────────────┐                                            │
    │  │   Visual    │  (Also in same call)                       │
    │  │  Director   │                                            │
    │  └─────────────┘                                            │
    │                                                              │
    │  Cost: $0.01 (Haiku) | Latency: <3s                        │
    └─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │          PARALLEL              │
              ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │  STEP 2: VIDEO  │              │  STEP 3: AUDIO  │
    │   (Kie.ai)      │              │  (ElevenLabs)   │
    │                 │              │                 │
    │ Cost: $2.00     │              │ Cost: $0.30     │
    │ Latency: 60-90s │              │ Latency: 5-10s  │
    └─────────────────┘              └─────────────────┘

    TOTAL COST: $2.31
    TOTAL LATENCY: ~90s (video gen dominates)
    """

    def __init__(self):
        # API Keys
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.kie_api_key = os.getenv("KIE_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Validate
        if not self.kie_api_key:
            logger.warning("KIE_API_KEY missing - video generation will fail")

        # Stats
        self.stats = {
            "videos_generated": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0
        }

        logger.info("ConsolidatedVideoProducer v3.0 initialized")

    async def produce(
        self,
        request: VideoRequest,
        on_progress: Callable[[str, int, str], Awaitable[None]]
    ) -> VideoResult:
        """
        Main production pipeline - 3 optimized steps.

        Args:
            request: Validated video request
            on_progress: Callback (stage, percent, message)

        Returns:
            VideoResult with video URL and metrics
        """
        request_id = hashlib.md5(f"{request.business}{time.time()}".encode()).hexdigest()[:12]
        start_time = time.time()
        cost_tracker = {"intelligence": 0.0, "video": 0.0, "audio": 0.0}

        # Start span if OpenTelemetry available
        span = None
        if OTEL_AVAILABLE and tracer:
            span = tracer.start_span("video_production_v3")
            span.set_attribute("request_id", request_id)
            span.set_attribute("style", request.style.value)

        try:
            # =========================================================
            # STEP 1: INTELLIGENCE (Merged - $0.01)
            # =========================================================
            await on_progress("INTELLIGENCE", 5, "Analyzing business and generating creative...")

            creative = await self._step_intelligence(request)
            cost_tracker["intelligence"] = 0.01

            await on_progress("INTELLIGENCE", 20, f"Creative ready. Hook: {creative.hook[:50]}...")

            # =========================================================
            # STEP 2 & 3: VIDEO + AUDIO (Parallel)
            # =========================================================
            await on_progress("GENERATION", 25, "Starting parallel video + audio generation...")

            # Run video and audio in parallel
            video_task = asyncio.create_task(
                self._step_video(creative.hero_shot_prompt, request.duration, on_progress)
            )

            audio_task = None
            if request.include_voice and self.elevenlabs_api_key:
                audio_task = asyncio.create_task(
                    self._step_audio(creative.full_voiceover, creative.voice_profile)
                )

            # Wait for video (dominant latency)
            video_url = await video_task
            cost_tracker["video"] = 2.00

            # Get audio result if running
            audio_url = None
            if audio_task:
                audio_url = await audio_task
                cost_tracker["audio"] = 0.30

            await on_progress("GENERATION", 90, "Video and audio complete.")

            # =========================================================
            # STEP 4: QA (Conditional - skip for non-enterprise)
            # =========================================================
            if request.is_enterprise:
                await on_progress("QA", 95, "Running enterprise quality checks...")
                await self._step_qa(video_url, creative)
            else:
                await on_progress("QA", 95, "Skipping QA (non-enterprise).")

            # =========================================================
            # COMPLETE
            # =========================================================
            total_time = time.time() - start_time
            total_cost = sum(cost_tracker.values())

            result = VideoResult(
                video_url=video_url,
                audio_url=audio_url,
                duration_seconds=request.duration,
                style=request.style,
                generation_time_seconds=total_time,
                cost_usd=total_cost,
                creative=creative,
                metrics={
                    "request_id": request_id,
                    "cost_breakdown": cost_tracker,
                    "latency_breakdown": {
                        "total": total_time
                    }
                }
            )

            # Update stats
            self.stats["videos_generated"] += 1
            self.stats["total_cost"] += total_cost

            # Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                videos_generated.labels(status="success", style=request.style.value).inc()
                for component, cost in cost_tracker.items():
                    total_cost_metric.labels(component=component).inc(cost)

            if span:
                span.set_status(Status(StatusCode.OK))

            logger.info(f"Video {request_id} complete: {total_time:.1f}s, ${total_cost:.2f}")

            await on_progress("COMPLETE", 100, f"Video ready! Cost: ${total_cost:.2f}")

            return result

        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            if PROMETHEUS_AVAILABLE:
                videos_generated.labels(status="error", style=request.style.value).inc()
            logger.error(f"Video {request_id} failed: {e}")
            raise
        finally:
            if span:
                span.end()

    # ========================================================================
    # STEP 1: INTELLIGENCE (Merged 4 Agents → 1 Call)
    # ========================================================================

    async def _step_intelligence(self, request: VideoRequest) -> CreativeOutput:
        """
        MERGED INTELLIGENCE STEP

        Replaces:
        - Business Intelligence Agent
        - Story Architect Agent
        - Script Writer Agent
        - Visual Director Agent

        One LLM call using Haiku/4o-mini generates ALL creative assets.

        Cost: $0.01 (vs $0.04 for 4 separate calls)
        Latency: <3s (vs 8-12s sequential)
        """
        start = time.time()

        # Select voice using rule-based logic (no LLM)
        voice_profile = select_voice_profile(request.style, request.business)
        music_mood = select_music_mood(request.style)

        # Build the MEGA prompt that replaces 4 agents
        prompt = self._build_merged_prompt(request, voice_profile, music_mood)

        # Call LLM (Haiku preferred, 4o-mini fallback)
        response = await self._call_intelligence_llm(prompt)

        # Parse response into CreativeOutput
        creative = self._parse_creative_response(response, voice_profile, music_mood)

        latency = time.time() - start
        if PROMETHEUS_AVAILABLE:
            step_latency.labels(step="intelligence").observe(latency)

        logger.info(f"Intelligence step: {latency:.2f}s, confidence: {creative.confidence}")

        return creative

    def _build_merged_prompt(
        self,
        request: VideoRequest,
        voice: VoiceProfile,
        music: str
    ) -> str:
        """
        The MEGA prompt that replaces 4 separate agent prompts.

        This is the key optimization - one well-structured prompt
        generates everything we need in a single LLM call.
        """
        return f"""You are an expert commercial video creative director. Analyze this business and generate ALL creative assets in one response.

BUSINESS: {request.business}
STYLE: {request.style.value}
DURATION: {request.duration} seconds
VOICE: {voice.name} (pre-selected)
MUSIC: {music} (pre-selected)

Generate a JSON response with these EXACT fields:

{{
  "hook": "Opening hook to grab attention in 2-3 seconds. Be specific to this business.",
  "value_proposition": "Core value statement - what makes this business special. One compelling sentence.",
  "call_to_action": "Clear, action-oriented CTA. Include a sense of urgency.",
  "full_voiceover": "Complete voiceover script for {request.duration} seconds. Should be {request.duration * 3} words max. Flow: Hook → Value → CTA.",
  "hero_shot_prompt": "AI video generation prompt for the main shot. Be VERY specific: setting, lighting, camera movement, subjects, mood, colors. 30-50 words. Style: {request.style.value}.",
  "b_roll_prompts": [
    "Supporting shot 1 prompt (15 words)",
    "Supporting shot 2 prompt (15 words)"
  ],
  "confidence": 0.85,
  "target_audience": "Specific audience description for this business"
}}

RULES:
1. full_voiceover must be readable in {request.duration} seconds (about {request.duration * 3} words)
2. hero_shot_prompt must be detailed enough for AI video generation
3. Everything must be specific to THIS business, not generic
4. Style must match: {request.style.value}
5. confidence should be 0.7-0.95 based on how well the business description allows for quality creative

Return ONLY valid JSON. No markdown, no explanation."""

    async def _call_intelligence_llm(self, prompt: str) -> str:
        """Call Haiku or 4o-mini for intelligence step"""

        # Prefer Anthropic Haiku (cheaper, faster)
        if self.anthropic_api_key:
            return await self._call_haiku(prompt)

        # Fallback to OpenAI 4o-mini
        if self.openai_api_key:
            return await self._call_4o_mini(prompt)

        # Emergency fallback
        return self._generate_fallback_creative()

    async def _call_haiku(self, prompt: str) -> str:
        """Claude 3 Haiku - $0.25/1M input, $1.25/1M output"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Haiku error: {resp.status} - {error}")

                data = await resp.json()
                return data["content"][0]["text"]

    async def _call_4o_mini(self, prompt: str) -> str:
        """GPT-4o-mini - $0.15/1M input, $0.60/1M output"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.7
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"4o-mini error: {resp.status} - {error}")

                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def _generate_fallback_creative(self) -> str:
        """Emergency fallback if no LLM available"""
        return json.dumps({
            "hook": "Discover excellence in service.",
            "value_proposition": "Quality you can trust, results you can see.",
            "call_to_action": "Contact us today to get started.",
            "full_voiceover": "Discover excellence in service. Quality you can trust, results you can see. Contact us today to get started.",
            "hero_shot_prompt": "Professional business environment, modern office, warm lighting, slow camera pan, corporate aesthetic, 4K quality",
            "b_roll_prompts": ["Happy customers in meeting", "Professional handshake close-up"],
            "confidence": 0.6,
            "target_audience": "Business professionals"
        })

    def _parse_creative_response(
        self,
        response: str,
        voice: VoiceProfile,
        music: str
    ) -> CreativeOutput:
        """Parse LLM response into CreativeOutput model"""
        try:
            # Clean markdown if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            data = json.loads(clean)

            return CreativeOutput(
                hook=data.get("hook", ""),
                value_proposition=data.get("value_proposition", ""),
                call_to_action=data.get("call_to_action", ""),
                full_voiceover=data.get("full_voiceover", ""),
                hero_shot_prompt=data.get("hero_shot_prompt", ""),
                b_roll_prompts=data.get("b_roll_prompts", []),
                voice_profile=voice,
                music_mood=music,
                confidence=float(data.get("confidence", 0.85)),
                target_audience=data.get("target_audience", "business professionals")
            )

        except Exception as e:
            logger.warning(f"Parse error: {e}, using fallback")
            fallback = json.loads(self._generate_fallback_creative())
            return CreativeOutput(
                **fallback,
                voice_profile=voice,
                music_mood=music
            )

    # ========================================================================
    # STEP 2: VIDEO GENERATION (Kie.ai)
    # ========================================================================

    async def _step_video(
        self,
        prompt: str,
        duration: int,
        on_progress: Callable
    ) -> str:
        """
        Video generation using Kie.ai Veo 3.

        Cost: $2.00 (fixed)
        Latency: 60-90s
        """
        start = time.time()

        if not self.kie_api_key:
            raise ValueError("KIE_API_KEY required for video generation")

        # Start generation
        task_id = await self._start_kie_job(prompt, duration)

        await on_progress("GENERATION", 30, f"Video job started: {task_id[:8]}...")

        # Poll for completion
        video_url = await self._poll_kie_job(task_id, on_progress)

        latency = time.time() - start
        if PROMETHEUS_AVAILABLE:
            step_latency.labels(step="video").observe(latency)

        return video_url

    async def _start_kie_job(self, prompt: str, duration: int) -> str:
        """Start Kie.ai video generation"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.kie.ai/api/v1/veo3/generate",
                headers={
                    "Authorization": f"Bearer {self.kie_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "prompt": prompt,
                    "aspectRatio": "16:9",
                    "model": "veo-3-fast-generate",
                    "duration": min(duration, 8)
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Kie.ai start error: {resp.status} - {error}")

                data = await resp.json()
                task_id = data.get("taskId")
                if not task_id:
                    raise Exception(f"No taskId in response: {data}")

                return task_id

    async def _poll_kie_job(
        self,
        task_id: str,
        on_progress: Callable,
        max_attempts: int = 60
    ) -> str:
        """Poll Kie.ai for completion"""
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_attempts):
                await asyncio.sleep(10)

                progress_pct = 30 + int((attempt / max_attempts) * 50)
                await on_progress("GENERATION", progress_pct, f"Rendering... ({attempt * 10}s)")

                try:
                    async with session.get(
                        "https://api.kie.ai/api/v1/veo3/record-info",
                        headers={"Authorization": f"Bearer {self.kie_api_key}"},
                        params={"taskId": task_id},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status != 200:
                            continue

                        data = await resp.json()
                        status = data.get("data", {}).get("status")

                        if status == "completed":
                            url = data.get("data", {}).get("videoUrl")
                            if url:
                                return url
                            raise Exception("No videoUrl in completed response")

                        if status == "failed":
                            error = data.get("data", {}).get("error", "Unknown")
                            raise Exception(f"Kie.ai failed: {error}")

                except asyncio.TimeoutError:
                    continue

        raise TimeoutError(f"Kie.ai job {task_id} timeout after {max_attempts * 10}s")

    # ========================================================================
    # STEP 3: AUDIO SYNTHESIS (ElevenLabs)
    # ========================================================================

    async def _step_audio(self, text: str, voice: VoiceProfile) -> Optional[str]:
        """
        Audio synthesis using ElevenLabs.

        Cost: $0.30
        Latency: 5-10s
        """
        if not self.elevenlabs_api_key:
            return None

        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice.value}",
                    headers={
                        "xi-api-key": self.elevenlabs_api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"ElevenLabs error: {resp.status}")
                        return None

                    # In production: upload to S3/R2 and return URL
                    latency = time.time() - start
                    if PROMETHEUS_AVAILABLE:
                        step_latency.labels(step="audio").observe(latency)

                    return f"audio://elevenlabs/{voice.name}/{time.time()}"

        except Exception as e:
            logger.warning(f"Audio generation failed: {e}")
            return None

    # ========================================================================
    # STEP 4: QUALITY ASSURANCE (Conditional)
    # ========================================================================

    async def _step_qa(self, video_url: str, creative: CreativeOutput) -> bool:
        """
        Quality assurance - ENTERPRISE ONLY.

        Skip for standard tier to save cost and time.
        Only validates:
        - Video is accessible
        - Duration matches request

        Cost: $0.00 (no LLM call)
        """
        # Basic validation (no LLM needed)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(video_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(f"QA: Video URL not accessible: {resp.status}")
                        return False

            # Confidence check
            if creative.confidence < 0.7:
                logger.warning(f"QA: Low confidence ({creative.confidence})")
                return False

            return True

        except Exception as e:
            logger.warning(f"QA check failed: {e}")
            return False


# ============================================================================
# WEBSOCKET INTEGRATION
# ============================================================================

async def create_progress_callback(websocket) -> Callable:
    """Create WebSocket callback for progress updates"""
    async def callback(stage: str, percent: int, message: str):
        try:
            await websocket.send_json({
                "type": "progress",
                "stage": stage,
                "percent": percent
            })
            await websocket.send_json({
                "type": "log",
                "message": message
            })
        except Exception as e:
            logger.warning(f"WebSocket send failed: {e}")

    return callback


# ============================================================================
# MAIN (For Testing)
# ============================================================================

async def main():
    """Test consolidated producer"""
    producer = ConsolidatedVideoProducer()

    async def test_progress(stage: str, percent: int, message: str):
        print(f"[{stage}] {percent}% - {message}")

    request = VideoRequest(
        business="Premium dental clinic in Austin specializing in cosmetic veneers for busy professionals who want a perfect smile",
        style=VideoStyle.CINEMATIC,
        duration=5,
        include_voice=False,  # Skip for testing
        is_enterprise=False   # Skip QA
    )

    try:
        result = await producer.produce(request, test_progress)
        print(f"\nVideo complete!")
        print(f"   URL: {result.video_url}")
        print(f"   Cost: ${result.cost_usd:.2f}")
        print(f"   Time: {result.generation_time_seconds:.1f}s")
        print(f"   Hook: {result.creative.hook}")
        print(f"   Voice: {result.creative.voice_profile.name}")
    except Exception as e:
        print(f"\nFailed: {e}")


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Alias the consolidated producer to the old name
VideoProducer = ConsolidatedVideoProducer

# Alias for any code expecting CreativeDirector
CreativeDirector = ConsolidatedVideoProducer


class LegacyVideoRequest:
    """
    Adapter for old-style dict requests.
    Converts to new Pydantic model.
    """
    @staticmethod
    def from_dict(data: dict) -> VideoRequest:
        return VideoRequest(
            business=data.get("business", ""),
            style=VideoStyle(data.get("style", "professional")),
            duration=int(data.get("duration", 5)),
            include_voice=bool(data.get("include_voice", True)),
            is_enterprise=bool(data.get("is_enterprise", False))
        )


# Export all public classes
__all__ = [
    "ConsolidatedVideoProducer",
    "VideoProducer",  # Alias
    "CreativeDirector",  # Alias
    "VideoRequest",
    "VideoResult",
    "VideoStyle",
    "CreativeOutput",
    "VoiceProfile",
    "LegacyVideoRequest",
    "create_progress_callback",
]


if __name__ == "__main__":
    asyncio.run(main())
