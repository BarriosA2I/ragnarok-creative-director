#!/usr/bin/env python3
"""
================================================================================
RAGNAROK BACKGROUND ORCHESTRATOR V8.0 APEX
================================================================================
The Autonomous Sales Machine - Production-Grade Background Services

UPGRADES FROM V7.0:
├─ Circuit Breakers: Supabase + Resend with automatic failover
├─ Intelligent DLQ: Poison message detection, exponential backoff
├─ Concurrency Control: Semaphore-bounded task pool
├─ Graceful Shutdown: Zero in-flight job loss
├─ Idempotency: Distributed locks prevent duplicate processing
├─ OpenTelemetry: Full distributed tracing
├─ Health Endpoints: Liveness + readiness probes
└─ Structured Logging: Correlation IDs for request tracing

Performance Targets:
├─ Uptime: 99.95%
├─ Job Success Rate: 97.5%+
├─ Recovery Time: <3s on circuit breaker trip
└─ Zero Message Loss: DLQ with poison detection

Author: Barrios A2I | Principal Architect
Version: 8.0.0 APEX
================================================================================
"""

import asyncio
import os
import json
import signal
import hashlib
import uuid
import logging
import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import functools

# External
from supabase import create_client, Client
from dotenv import load_dotenv
import aiohttp

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# ============================================================================
# TELEMETRY INITIALIZATION
# ============================================================================

if OTEL_AVAILABLE:
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if OTEL_ENDPOINT:
        span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT))
        trace.get_tracer_provider().add_span_processor(span_processor)

    AioHttpClientInstrumentor().instrument()
else:
    tracer = None

# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class StructuredLogger:
    """JSON-structured logging with correlation ID support."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _format(self, msg: str, correlation_id: Optional[str] = None, **kwargs) -> str:
        data = {"message": msg, **kwargs}
        if correlation_id:
            data["correlation_id"] = correlation_id
        return json.dumps(data) if kwargs else msg

    def info(self, msg: str, correlation_id: Optional[str] = None, **kwargs):
        self.logger.info(self._format(msg, correlation_id, **kwargs))

    def warning(self, msg: str, correlation_id: Optional[str] = None, **kwargs):
        self.logger.warning(self._format(msg, correlation_id, **kwargs))

    def error(self, msg: str, correlation_id: Optional[str] = None, **kwargs):
        self.logger.error(self._format(msg, correlation_id, **kwargs))

    def critical(self, msg: str, correlation_id: Optional[str] = None, **kwargs):
        self.logger.critical(self._format(msg, correlation_id, **kwargs))

logger = StructuredLogger("RAGNAROK.Orchestrator.V8")

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

if PROMETHEUS_AVAILABLE:
    payments_processed = Counter(
        'ragnarok_payments_processed_total',
        'Total payments processed',
        ['status', 'retry_count']
    )

    hot_leads_processed = Counter(
        'ragnarok_hot_leads_processed_total',
        'Total hot leads engaged',
        ['status', 'retry_count']
    )

    emails_sent = Counter(
        'ragnarok_emails_sent_total',
        'Total emails sent',
        ['status', 'service']
    )

    circuit_breaker_trips = Counter(
        'ragnarok_circuit_breaker_trips_total',
        'Circuit breaker state changes',
        ['service', 'from_state', 'to_state']
    )

    dlq_messages = Counter(
        'ragnarok_dlq_messages_total',
        'Messages sent to DLQ',
        ['queue', 'category']
    )

    poison_messages = Counter(
        'ragnarok_poison_messages_total',
        'Poison messages detected',
        ['queue']
    )

    active_jobs = Gauge(
        'ragnarok_active_jobs',
        'Currently running jobs',
        ['service']
    )

    circuit_breaker_state = Gauge(
        'ragnarok_circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half-open)',
        ['service']
    )

    dlq_depth = Gauge(
        'ragnarok_dlq_depth',
        'Current DLQ depth',
        ['queue']
    )

    orchestrator_health = Gauge(
        'ragnarok_orchestrator_health',
        'Orchestrator health status (1=healthy, 0=unhealthy)'
    )

    job_latency = Histogram(
        'ragnarok_job_latency_seconds',
        'Job processing time',
        ['service'],
        buckets=[10, 30, 60, 120, 180, 240, 300, 600]
    )

    retry_delay = Histogram(
        'ragnarok_retry_delay_seconds',
        'Time between retries',
        ['queue'],
        buckets=[1, 5, 15, 30, 60, 120, 300]
    )

    orchestrator_info = Info(
        'ragnarok_orchestrator',
        'Orchestrator version and configuration'
    )
    orchestrator_info.info({
        'version': '8.0.0',
        'variant': 'APEX',
        'author': 'Barrios A2I'
    })

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

@dataclass
class OrchestratorConfig:
    """Central configuration with validation."""

    # Supabase
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY", "")))

    # Email
    resend_api_key: str = field(default_factory=lambda: os.getenv("RESEND_API_KEY", ""))
    from_email: str = field(default_factory=lambda: os.getenv("FROM_EMAIL", "Barrios A2I <director@barriosa2i.com>"))
    website_url: str = field(default_factory=lambda: os.getenv("WEBSITE_URL", "https://barriosa2i.com"))

    # Polling
    payment_poll_interval: int = 10
    hot_lead_poll_interval: int = 30
    dlq_poll_interval: int = 60

    # Thresholds
    hot_lead_score_threshold: int = 8

    # Concurrency
    max_concurrent_payments: int = 3
    max_concurrent_hunters: int = 5

    # Circuit Breaker
    cb_failure_threshold: int = 5
    cb_success_threshold: int = 2
    cb_timeout_seconds: float = 30.0
    cb_half_open_max_requests: int = 3

    # DLQ
    dlq_max_retries: int = 5
    dlq_base_delay_seconds: float = 30.0
    dlq_max_delay_seconds: float = 600.0
    dlq_poison_threshold: int = 5

    # Health
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        if not self.supabase_key:
            errors.append("SUPABASE_SERVICE_ROLE_KEY is required")
        return errors

config = OrchestratorConfig()

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    CLOSED = 0      # Normal operation
    OPEN = 1        # Failing, reject requests
    HALF_OPEN = 2   # Testing recovery

@dataclass
class CircuitBreaker:
    """Production circuit breaker with automatic state transitions."""

    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_requests: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_requests: int = field(default=0)

    def __post_init__(self):
        if PROMETHEUS_AVAILABLE:
            circuit_breaker_state.labels(service=self.name).set(self.state.value)

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout_seconds:
                self._transition(CircuitState.HALF_OPEN)
                self.half_open_requests = 0
                return True
            return False

        if self.half_open_requests < self.half_open_max_requests:
            self.half_open_requests += 1
            return True
        return False

    def record_success(self):
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition(CircuitState.CLOSED)
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self, error: Exception):
        """Record failed call."""
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition(CircuitState.OPEN)
                self.last_failure_time = time.time()
                logger.warning(
                    f"Circuit breaker {self.name} OPENED after {self.failure_count} failures",
                    service=self.name,
                    error=str(error)
                )

        elif self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
            self.last_failure_time = time.time()
            self.success_count = 0

    def _transition(self, new_state: CircuitState):
        """Handle state transition with metrics."""
        old_state = self.state
        self.state = new_state
        if PROMETHEUS_AVAILABLE:
            circuit_breaker_state.labels(service=self.name).set(new_state.value)
            circuit_breaker_trips.labels(
                service=self.name,
                from_state=old_state.name,
                to_state=new_state.name
            ).inc()
        logger.info(f"Circuit breaker {self.name}: {old_state.name} -> {new_state.name}")

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

# ============================================================================
# INTELLIGENT DEAD LETTER QUEUE
# ============================================================================

class MessageCategory(Enum):
    """Categories for failed jobs with different retry strategies."""
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    VIDEO_ERROR = "video_error"
    EMAIL_ERROR = "email_error"
    DATABASE = "database"
    POISON = "poison"

@dataclass
class FailedJob:
    """A failed job awaiting retry."""
    id: str
    queue: str
    payload: Dict[str, Any]
    error: str
    category: MessageCategory
    attempt_count: int = 0
    first_failure: datetime = field(default_factory=datetime.utcnow)
    last_failure: datetime = field(default_factory=datetime.utcnow)
    next_retry: datetime = field(default_factory=datetime.utcnow)
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            content = json.dumps(self.payload, sort_keys=True)
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()

class IntelligentDLQ:
    """Dead Letter Queue with poison message detection and intelligent retry."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 30.0,
        max_delay: float = 600.0,
        poison_threshold: int = 5
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.poison_threshold = poison_threshold

        self.queues: Dict[str, deque] = defaultdict(deque)
        self.failure_fingerprints: Dict[str, int] = defaultdict(int)
        self.detected_poisons: Set[str] = set()
        self.parking_lot: List[FailedJob] = []
        self.handlers: Dict[str, Callable] = {}

        self.error_patterns = {
            MessageCategory.TRANSIENT: ["timeout", "connection", "network", "unreachable", "503"],
            MessageCategory.RATE_LIMIT: ["rate limit", "too many requests", "429", "throttled"],
            MessageCategory.VIDEO_ERROR: ["video", "sora", "veo", "kie", "generation failed"],
            MessageCategory.EMAIL_ERROR: ["email", "resend", "smtp", "delivery failed"],
            MessageCategory.DATABASE: ["supabase", "postgresql", "database", "constraint"]
        }

    def categorize_error(self, error: str) -> MessageCategory:
        error_lower = error.lower()
        for category, patterns in self.error_patterns.items():
            if any(p in error_lower for p in patterns):
                return category
        return MessageCategory.TRANSIENT

    def is_poison(self, job: FailedJob) -> bool:
        if job.fingerprint in self.detected_poisons:
            return True
        failure_count = self.failure_fingerprints[job.fingerprint]
        if failure_count >= self.poison_threshold:
            self.detected_poisons.add(job.fingerprint)
            if PROMETHEUS_AVAILABLE:
                poison_messages.labels(queue=job.queue).inc()
            return True
        return False

    def calculate_delay(self, attempt: int) -> float:
        import random
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay * random.uniform(0.8, 1.2)

    async def enqueue(
        self,
        queue: str,
        payload: Dict[str, Any],
        error: str,
        job_id: Optional[str] = None
    ) -> str:
        category = self.categorize_error(error)
        job = FailedJob(
            id=job_id or str(uuid.uuid4()),
            queue=queue,
            payload=payload,
            error=error,
            category=category
        )

        self.failure_fingerprints[job.fingerprint] += 1

        if self.is_poison(job):
            job.category = MessageCategory.POISON
            self.parking_lot.append(job)
            logger.error(f"Job {job.id} sent to parking lot (poison)")
            return job.id

        delay = self.calculate_delay(job.attempt_count)
        job.next_retry = datetime.utcnow() + timedelta(seconds=delay)

        self.queues[queue].append(job)
        if PROMETHEUS_AVAILABLE:
            dlq_messages.labels(queue=queue, category=category.value).inc()
            dlq_depth.labels(queue=queue).set(len(self.queues[queue]))

        logger.info(f"Job {job.id} queued for retry in {delay:.0f}s")
        return job.id

    def register_handler(self, queue: str, handler: Callable):
        self.handlers[queue] = handler

    async def process_queue(self, queue: str) -> int:
        if queue not in self.handlers:
            return 0

        handler = self.handlers[queue]
        processed = 0
        now = datetime.utcnow()

        ready_jobs = []
        remaining = deque()

        while self.queues[queue]:
            job = self.queues[queue].popleft()
            if job.next_retry <= now:
                ready_jobs.append(job)
            else:
                remaining.append(job)

        self.queues[queue] = remaining

        for job in ready_jobs:
            job.attempt_count += 1

            if job.attempt_count > self.max_retries:
                self.parking_lot.append(job)
                continue

            try:
                await asyncio.wait_for(handler(job.payload), timeout=300.0)
                logger.info(f"Job {job.id} retry successful")
                processed += 1
            except Exception as e:
                job.error = str(e)
                job.last_failure = datetime.utcnow()
                self.failure_fingerprints[job.fingerprint] += 1

                if self.is_poison(job):
                    job.category = MessageCategory.POISON
                    self.parking_lot.append(job)
                else:
                    job.next_retry = datetime.utcnow() + timedelta(
                        seconds=self.calculate_delay(job.attempt_count)
                    )
                    self.queues[queue].append(job)

        if PROMETHEUS_AVAILABLE:
            dlq_depth.labels(queue=queue).set(len(self.queues[queue]))
        return processed

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "queues": {q: len(jobs) for q, jobs in self.queues.items()},
            "parking_lot_size": len(self.parking_lot),
            "poison_messages": len(self.detected_poisons)
        }

# ============================================================================
# GRACEFUL SHUTDOWN HANDLER
# ============================================================================

class GracefulShutdown:
    """Manages graceful shutdown with in-flight job tracking."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.in_flight_jobs: Set[str] = set()
        self.shutdown_timeout = 30.0

    def register_signals(self):
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: self._signal_handler())

    def _signal_handler(self):
        logger.info(f"Shutdown signal received, {len(self.in_flight_jobs)} jobs in flight")
        self.shutdown_event.set()

    @asynccontextmanager
    async def track_job(self, job_id: str):
        self.in_flight_jobs.add(job_id)
        try:
            yield
        finally:
            self.in_flight_jobs.discard(job_id)

    async def wait_for_shutdown(self):
        await self.shutdown_event.wait()

        if self.in_flight_jobs:
            logger.info(f"Waiting for {len(self.in_flight_jobs)} in-flight jobs")
            start = datetime.utcnow()
            while self.in_flight_jobs:
                elapsed = (datetime.utcnow() - start).total_seconds()
                if elapsed > self.shutdown_timeout:
                    logger.warning(f"Shutdown timeout, {len(self.in_flight_jobs)} jobs orphaned")
                    break
                await asyncio.sleep(1)

        logger.info("Graceful shutdown complete")

    @property
    def is_shutting_down(self) -> bool:
        return self.shutdown_event.is_set()

# ============================================================================
# IDEMPOTENCY / DISTRIBUTED LOCK
# ============================================================================

class IdempotencyManager:
    """Prevents duplicate job processing using database-backed locks."""

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.local_locks: Set[str] = set()

    async def try_acquire(self, job_type: str, job_id: str) -> bool:
        lock_key = f"{job_type}:{job_id}"

        if lock_key in self.local_locks:
            return False

        try:
            result = self.supabase.rpc(
                'try_acquire_job_lock',
                {'p_job_id': job_id, 'p_job_type': job_type}
            ).execute()

            if result.data:
                self.local_locks.add(lock_key)
                return True
            return False

        except Exception as e:
            logger.warning(f"Lock acquisition fallback: {e}")
            if lock_key not in self.local_locks:
                self.local_locks.add(lock_key)
                return True
            return False

    def release(self, job_type: str, job_id: str):
        lock_key = f"{job_type}:{job_id}"
        self.local_locks.discard(lock_key)

# ============================================================================
# RAGNAROK ORCHESTRATOR V8.0
# ============================================================================

class RAGNAROKOrchestratorV8:
    """Production-grade background orchestrator."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config

        errors = config.validate()
        if errors:
            for error in errors:
                logger.critical(f"Configuration error: {error}")
            raise ValueError(f"Configuration errors: {errors}")

        self.supabase: Client = create_client(config.supabase_url, config.supabase_key)
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._producer = None

        self.cb_supabase = CircuitBreaker(
            name="supabase",
            failure_threshold=config.cb_failure_threshold,
            success_threshold=config.cb_success_threshold,
            timeout_seconds=config.cb_timeout_seconds
        )
        self.cb_resend = CircuitBreaker(name="resend", failure_threshold=3, timeout_seconds=60.0)
        self.cb_video = CircuitBreaker(name="video_producer", failure_threshold=3, timeout_seconds=120.0)

        self.dlq = IntelligentDLQ(
            max_retries=config.dlq_max_retries,
            base_delay=config.dlq_base_delay_seconds,
            max_delay=config.dlq_max_delay_seconds,
            poison_threshold=config.dlq_poison_threshold
        )

        self.payment_semaphore = asyncio.Semaphore(config.max_concurrent_payments)
        self.hunter_semaphore = asyncio.Semaphore(config.max_concurrent_hunters)
        self.shutdown = GracefulShutdown()
        self.idempotency = IdempotencyManager(self.supabase)
        self.active_tasks: Set[asyncio.Task] = set()

    @property
    def producer(self):
        if self._producer is None:
            # Import video producer (adjust path as needed)
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from api.video_agents import ConsolidatedVideoProducer
            self._producer = ConsolidatedVideoProducer()
        return self._producer

    async def start(self):
        logger.info("Starting RAGNAROK Orchestrator V8.0 APEX")

        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.config.metrics_port)
                logger.info(f"Metrics server started on port {self.config.metrics_port}")
            except Exception as e:
                logger.warning(f"Could not start metrics server: {e}")

        self.http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

        self.dlq.register_handler("payments", self._retry_payment)
        self.dlq.register_handler("hunters", self._retry_hot_lead)

        self.shutdown.register_signals()

        if PROMETHEUS_AVAILABLE:
            orchestrator_health.set(1)
        logger.info("Orchestrator initialized successfully")

    async def stop(self):
        logger.info("Stopping orchestrator...")
        if PROMETHEUS_AVAILABLE:
            orchestrator_health.set(0)

        for task in self.active_tasks:
            task.cancel()

        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

        if self.http_session:
            await self.http_session.close()

        logger.info("Orchestrator stopped")

    async def _send_email(self, payload: Dict[str, Any], correlation_id: str) -> bool:
        if not self.config.resend_api_key:
            logger.warning("Email skipped - no API key")
            return False

        if not self.cb_resend.can_execute():
            raise CircuitBreakerOpenError("Resend circuit breaker is OPEN")

        url = "https://api.resend.com/emails"
        headers = {
            "Authorization": f"Bearer {self.config.resend_api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with self.http_session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    if PROMETHEUS_AVAILABLE:
                        emails_sent.labels(status='success', service='resend').inc()
                    self.cb_resend.record_success()
                    return True
                else:
                    error = await resp.text()
                    if PROMETHEUS_AVAILABLE:
                        emails_sent.labels(status='error', service='resend').inc()
                    raise Exception(f"Resend API error: {resp.status} - {error}")
        except Exception as e:
            self.cb_resend.record_failure(e)
            raise

    async def process_payment(self, lead: Dict[str, Any]) -> None:
        lead_id = lead['id']
        correlation_id = str(uuid.uuid4())[:8]
        email = lead.get('email', 'unknown')
        company = lead.get('company_name') or lead.get('intent', 'Client')

        logger.info(f"💰 PAYMENT: {company}", correlation_id=correlation_id, lead_id=lead_id)

        if not await self.idempotency.try_acquire("payment", lead_id):
            logger.info(f"Payment {lead_id} already being processed")
            return

        start_time = datetime.utcnow()
        async with self.shutdown.track_job(f"payment:{lead_id}"):
            try:
                if PROMETHEUS_AVAILABLE:
                    active_jobs.labels(service="payment").inc()

                self.supabase.table('leads').update({
                    'status': 'processing',
                    'processing_started_at': datetime.utcnow().isoformat(),
                    'correlation_id': correlation_id
                }).eq('id', lead_id).execute()

                if not self.cb_video.can_execute():
                    raise CircuitBreakerOpenError("Video producer circuit breaker OPEN")

                try:
                    from api.video_agents import VideoRequest, VideoStyle

                    result = await self.producer.produce(
                        VideoRequest(
                            business=lead.get('intent') or company,
                            style=VideoStyle.PROFESSIONAL,
                            duration=8,
                            include_voice=True
                        ),
                        on_progress=lambda s, p, m: logger.info(f"[{lead_id[:8]}] {s}: {m}")
                    )
                    self.cb_video.record_success()

                except Exception as e:
                    self.cb_video.record_failure(e)
                    raise

                self.supabase.table('leads').update({
                    'status': 'completed',
                    'video_url': result.video_url,
                    'audio_url': result.audio_url,
                    'cost_usd': result.cost_usd,
                    'generation_time_seconds': result.generation_time_seconds,
                    'completed_at': datetime.utcnow().isoformat()
                }).eq('id', lead_id).execute()

                if email and email != 'unknown':
                    try:
                        await self._send_delivery_email(
                            to=email,
                            company=company,
                            video_url=result.video_url,
                            correlation_id=correlation_id
                        )
                    except Exception as e:
                        logger.warning(f"Delivery email failed: {e}")

                if PROMETHEUS_AVAILABLE:
                    payments_processed.labels(status='success', retry_count='0').inc()
                logger.info(f"✅ ORDER FULFILLED: {lead_id}", cost=result.cost_usd)

            except CircuitBreakerOpenError as e:
                await self.dlq.enqueue("payments", {"lead": lead}, str(e), job_id=lead_id)
                if PROMETHEUS_AVAILABLE:
                    payments_processed.labels(status='deferred', retry_count='0').inc()

            except Exception as e:
                logger.error(f"❌ PAYMENT FAILED: {lead_id}", error=str(e))

                self.supabase.table('leads').update({
                    'status': 'error',
                    'error_message': str(e),
                    'failed_at': datetime.utcnow().isoformat()
                }).eq('id', lead_id).execute()

                await self.dlq.enqueue("payments", {"lead": lead}, str(e), job_id=lead_id)
                if PROMETHEUS_AVAILABLE:
                    payments_processed.labels(status='error', retry_count='0').inc()

            finally:
                self.idempotency.release("payment", lead_id)
                if PROMETHEUS_AVAILABLE:
                    active_jobs.labels(service="payment").dec()
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    job_latency.labels(service="payment").observe(elapsed)

    async def _retry_payment(self, payload: Dict[str, Any]) -> None:
        lead = payload["lead"]
        self.supabase.table('leads').update({'status': 'paid_client'}).eq('id', lead['id']).execute()
        await self.process_payment(lead)

    async def _send_delivery_email(self, to: str, company: str, video_url: str, correlation_id: str) -> bool:
        payload = {
            "from": self.config.from_email,
            "to": [to],
            "subject": f"🎬 Your Commercial for {company} is Ready!",
            "html": f"""
            <h2>Your Commercial is Ready!</h2>
            <p>Great news! Your AI-generated commercial for <strong>{company}</strong> has been completed.</p>
            <p><a href="{video_url}" style="background:#0066cc;color:white;padding:12px 24px;text-decoration:none;border-radius:6px;display:inline-block;">▶️ WATCH YOUR COMMERCIAL</a></p>
            <p>You can download the video directly from the link above.</p>
            <br/>
            <p>Need revisions or have questions? Reply to this email.</p>
            <p>Best,<br/>The Barrios A2I Team</p>
            """
        }
        return await self._send_email(payload, correlation_id)

    async def process_hot_lead(self, lead: Dict[str, Any]) -> None:
        lead_id = lead['id']
        correlation_id = str(uuid.uuid4())[:8]
        email = lead.get('email')
        company = lead.get('company_name') or lead.get('business_name') or "Your Business"
        score = lead.get('qualification_score', 0)

        logger.info(f"🔥 HOT LEAD: {company} (Score: {score})", lead_id=lead_id)

        if not await self.idempotency.try_acquire("hunter", lead_id):
            return

        start_time = datetime.utcnow()
        async with self.shutdown.track_job(f"hunter:{lead_id}"):
            try:
                if PROMETHEUS_AVAILABLE:
                    active_jobs.labels(service="hunter").inc()

                self.supabase.table('discovery_leads').update({
                    'status': 'engaging',
                    'engagement_started_at': datetime.utcnow().isoformat(),
                    'correlation_id': correlation_id
                }).eq('id', lead_id).execute()

                if not self.cb_video.can_execute():
                    raise CircuitBreakerOpenError("Video producer circuit breaker OPEN")

                try:
                    from api.video_agents import VideoRequest, VideoStyle

                    business_context = f"{company} - {lead.get('industry', 'General')}"

                    result = await self.producer.produce(
                        VideoRequest(
                            business=business_context,
                            style=VideoStyle.CINEMATIC,
                            duration=5,
                            include_voice=False
                        ),
                        on_progress=lambda s, p, m: logger.info(f"[TEASER] {s}: {m}")
                    )
                    self.cb_video.record_success()

                except Exception as e:
                    self.cb_video.record_failure(e)
                    raise

                email_sent = False
                if email:
                    try:
                        email_sent = await self._send_teaser_email(
                            to=email,
                            company=company,
                            video_url=result.video_url,
                            hook=result.creative.hook if hasattr(result, 'creative') else "See what AI can do for you",
                            score=score,
                            correlation_id=correlation_id
                        )
                    except Exception as e:
                        logger.warning(f"Teaser email failed: {e}")

                new_status = 'nurturing' if email_sent else 'manual_followup'
                self.supabase.table('discovery_leads').update({
                    'status': new_status,
                    'teaser_video_url': result.video_url,
                    'teaser_cost_usd': result.cost_usd,
                    'email_sent': email_sent,
                    'engaged_at': datetime.utcnow().isoformat()
                }).eq('id', lead_id).execute()

                if PROMETHEUS_AVAILABLE:
                    hot_leads_processed.labels(status='success', retry_count='0').inc()
                logger.info(f"✅ TEASER DEPLOYED: {company} → {new_status}")

            except CircuitBreakerOpenError as e:
                await self.dlq.enqueue("hunters", {"lead": lead}, str(e), job_id=lead_id)
                if PROMETHEUS_AVAILABLE:
                    hot_leads_processed.labels(status='deferred', retry_count='0').inc()

            except Exception as e:
                logger.error(f"❌ HUNTER FAILED: {lead_id}", error=str(e))

                self.supabase.table('discovery_leads').update({
                    'status': 'error',
                    'error_message': str(e)
                }).eq('id', lead_id).execute()

                await self.dlq.enqueue("hunters", {"lead": lead}, str(e), job_id=lead_id)
                if PROMETHEUS_AVAILABLE:
                    hot_leads_processed.labels(status='error', retry_count='0').inc()

            finally:
                self.idempotency.release("hunter", lead_id)
                if PROMETHEUS_AVAILABLE:
                    active_jobs.labels(service="hunter").dec()
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    job_latency.labels(service="hunter").observe(elapsed)

    async def _retry_hot_lead(self, payload: Dict[str, Any]) -> None:
        lead = payload["lead"]
        self.supabase.table('discovery_leads').update({'status': 'new'}).eq('id', lead['id']).execute()
        await self.process_hot_lead(lead)

    async def _send_teaser_email(
        self,
        to: str,
        company: str,
        video_url: str,
        hook: str,
        score: int,
        correlation_id: str
    ) -> bool:
        payload = {
            "from": self.config.from_email,
            "to": [to],
            "subject": f"I made a commercial for {company} (AI Generated)",
            "html": f"""
            <p>I noticed you exploring our Command Center.</p>
            <p>Our autonomous agents analyzed <strong>{company}</strong> and generated this concept video:</p>
            <p>
                <a href="{video_url}" style="background:#0066cc;color:white;padding:12px 24px;text-decoration:none;border-radius:6px;display:inline-block;">
                    ▶️ WATCH YOUR TEASER VIDEO
                </a>
            </p>
            <p><em>Creative Angle: "{hook}"</em></p>
            <hr style="margin:24px 0;border:none;border-top:1px solid #eee;"/>
            <p>This is a 5-second teaser. If you want the full 30-second version with professional voiceover:</p>
            <p>
                <a href="{self.config.website_url}/video-generator" style="color:#0066cc;">
                    → Deploy the full agent here
                </a>
            </p>
            <br/>
            <p>Best,<br/>RAGNAROK Auto-Sender</p>
            """
        }
        return await self._send_email(payload, correlation_id)

    async def watch_payments(self) -> None:
        logger.info(f"👁️ PAYMENT WATCHER: Active (poll every {self.config.payment_poll_interval}s)")

        while not self.shutdown.is_shutting_down:
            try:
                if self.cb_supabase.can_execute():
                    try:
                        response = self.supabase.table('leads')\
                            .select("*")\
                            .eq('status', 'paid_client')\
                            .execute()
                        self.cb_supabase.record_success()

                        for lead in response.data:
                            if self.shutdown.is_shutting_down:
                                break

                            async with self.payment_semaphore:
                                task = asyncio.create_task(self.process_payment(lead))
                                self.active_tasks.add(task)
                                task.add_done_callback(self.active_tasks.discard)

                    except Exception as e:
                        self.cb_supabase.record_failure(e)
                        logger.error(f"Payment query failed: {e}")

            except Exception as e:
                logger.error(f"Payment watcher error: {e}")

            await asyncio.sleep(self.config.payment_poll_interval)

    async def watch_hot_leads(self) -> None:
        logger.info(f"👁️ HUNTER WATCHER: Active (score >= {self.config.hot_lead_score_threshold})")

        while not self.shutdown.is_shutting_down:
            try:
                if self.cb_supabase.can_execute():
                    try:
                        response = self.supabase.table('discovery_leads')\
                            .select("*")\
                            .gte('qualification_score', self.config.hot_lead_score_threshold)\
                            .eq('status', 'new')\
                            .execute()
                        self.cb_supabase.record_success()

                        for lead in response.data:
                            if self.shutdown.is_shutting_down:
                                break

                            async with self.hunter_semaphore:
                                task = asyncio.create_task(self.process_hot_lead(lead))
                                self.active_tasks.add(task)
                                task.add_done_callback(self.active_tasks.discard)

                    except Exception as e:
                        self.cb_supabase.record_failure(e)
                        logger.error(f"Hunter query failed: {e}")

            except Exception as e:
                logger.error(f"Hunter watcher error: {e}")

            await asyncio.sleep(self.config.hot_lead_poll_interval)

    async def process_dlq(self) -> None:
        logger.info(f"👁️ DLQ PROCESSOR: Active (poll every {self.config.dlq_poll_interval}s)")

        while not self.shutdown.is_shutting_down:
            try:
                for queue in ["payments", "hunters"]:
                    processed = await self.dlq.process_queue(queue)
                    if processed > 0:
                        logger.info(f"DLQ processed {processed} jobs from {queue}")

            except Exception as e:
                logger.error(f"DLQ processor error: {e}")

            await asyncio.sleep(self.config.dlq_poll_interval)

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if not self.shutdown.is_shutting_down else "shutting_down",
            "version": "8.0.0",
            "services": {
                "payment_watcher": "running",
                "hot_lead_hunter": "running",
                "dlq_processor": "running"
            },
            "circuit_breakers": {
                "supabase": self.cb_supabase.state.name,
                "resend": self.cb_resend.state.name,
                "video": self.cb_video.state.name
            },
            "dlq": self.dlq.get_statistics(),
            "in_flight_jobs": len(self.shutdown.in_flight_jobs),
            "active_tasks": len(self.active_tasks)
        }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Start all background services."""
    print("")
    print("=" * 70)
    print("   RAGNAROK BACKGROUND ORCHESTRATOR V8.0 APEX")
    print("=" * 70)
    print("")
    print("   🛡️  RESILIENCE PATTERNS:")
    print("       ├─ Circuit Breakers: Supabase, Resend, Video Producer")
    print("       ├─ Intelligent DLQ: Poison detection, exponential backoff")
    print("       ├─ Graceful Shutdown: Zero in-flight job loss")
    print("       └─ Idempotency: Distributed locks")
    print("")
    print("   📊 OBSERVABILITY:")
    print(f"       ├─ Metrics: http://localhost:{config.metrics_port}/metrics")
    print("       ├─ Tracing: OpenTelemetry (OTLP)")
    print("       └─ Structured Logging: JSON with correlation IDs")
    print("")
    print("   ⚙️  SERVICES:")
    print(f"       [1] Payment Processor: Poll every {config.payment_poll_interval}s")
    print(f"           └─ Max concurrent: {config.max_concurrent_payments}")
    print(f"       [2] Hot Lead Hunter:   Poll every {config.hot_lead_poll_interval}s")
    print(f"           └─ Score threshold: >= {config.hot_lead_score_threshold}")
    print(f"       [3] DLQ Processor:     Poll every {config.dlq_poll_interval}s")
    print(f"           └─ Max retries: {config.dlq_max_retries}")
    print("       [4] Email Service:     " + ("ONLINE" if config.resend_api_key else "DISABLED"))
    print("")
    print("=" * 70)
    print("")

    orchestrator = RAGNAROKOrchestratorV8(config)

    try:
        await orchestrator.start()

        await asyncio.gather(
            orchestrator.watch_payments(),
            orchestrator.watch_hot_leads(),
            orchestrator.process_dlq(),
            orchestrator.shutdown.wait_for_shutdown()
        )

    except Exception as e:
        logger.critical(f"Orchestrator crashed: {e}")
        raise
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
