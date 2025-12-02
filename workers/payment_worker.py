"""
================================================================================
PAYMENT WORKER
================================================================================
Background worker for processing payments and triggering video generation.
Listens to Supabase webhooks for paid leads.

Usage:
  python workers/payment_worker.py

Environment:
  SUPABASE_URL: Supabase project URL
  SUPABASE_KEY: Supabase service role key
  STRIPE_SECRET_KEY: Stripe secret key
  STRIPE_WEBHOOK_SECRET: Stripe webhook signing secret
================================================================================
"""

import os
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("PaymentWorker")


async def process_payment_event(event: dict):
    """Process a payment event from Stripe"""
    event_type = event.get("type")

    if event_type == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        customer_email = session.get("customer_email")
        amount = session.get("amount_total", 0) / 100  # Convert cents to dollars

        logger.info(f"Payment received: ${amount} from {customer_email}")

        # TODO: Trigger video generation for paid customer
        # await trigger_video_generation(customer_email, session)

    elif event_type == "payment_intent.succeeded":
        intent = event.get("data", {}).get("object", {})
        amount = intent.get("amount", 0) / 100
        logger.info(f"Payment intent succeeded: ${amount}")


async def listen_for_events():
    """Main event loop for payment processing"""
    logger.info("Payment worker starting...")
    logger.info(f"Supabase URL: {os.getenv('SUPABASE_URL', 'NOT SET')}")
    logger.info(f"Stripe configured: {bool(os.getenv('STRIPE_SECRET_KEY'))}")

    # Keep worker alive
    while True:
        try:
            # In production, this would:
            # 1. Listen to Supabase realtime for new paid leads
            # 2. Process Stripe webhooks
            # 3. Trigger video generation

            await asyncio.sleep(60)  # Check every minute
            logger.debug("Payment worker heartbeat")

        except Exception as e:
            logger.error(f"Worker error: {e}")
            await asyncio.sleep(5)


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("RAGNAROK PAYMENT WORKER")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Run the async event loop
    asyncio.run(listen_for_events())


if __name__ == "__main__":
    main()
