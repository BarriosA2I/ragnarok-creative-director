"""
================================================================================
VIDEO CLEANUP WORKER
================================================================================
Removes video files older than MAX_AGE_DAYS to manage persistent disk space.
Runs as a Render cron job daily at 3 AM UTC.

Usage:
  python workers/cleanup_old_videos.py

Environment:
  VIDEO_OUTPUT_DIR: Path to video storage (default: /var/data/videos)
  MAX_AGE_DAYS: Days to keep videos (default: 7)
  DRY_RUN: If "true", only logs what would be deleted
================================================================================
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("VideoCleanup")


def get_file_age_days(file_path: Path) -> float:
    """Get file age in days"""
    mtime = file_path.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / (24 * 60 * 60)


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def cleanup_videos(
    video_dir: str,
    max_age_days: int = 7,
    dry_run: bool = False
) -> dict:
    """
    Remove video files older than max_age_days.

    Args:
        video_dir: Directory containing video files
        max_age_days: Maximum age in days before deletion
        dry_run: If True, only log what would be deleted

    Returns:
        Stats dict with counts and sizes
    """
    video_path = Path(video_dir)

    if not video_path.exists():
        logger.warning(f"Video directory does not exist: {video_dir}")
        return {"status": "skipped", "reason": "directory_not_found"}

    # Video file extensions to clean
    video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.mp3', '.wav'}

    stats = {
        "total_files": 0,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "kept_files": 0,
        "kept_bytes": 0,
        "errors": []
    }

    logger.info(f"Scanning {video_dir} for files older than {max_age_days} days...")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE DELETION'}")

    for file_path in video_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Only process video/audio files
        if file_path.suffix.lower() not in video_extensions:
            continue

        stats["total_files"] += 1
        file_size = file_path.stat().st_size
        file_age = get_file_age_days(file_path)

        if file_age > max_age_days:
            # File is old, delete it
            stats["deleted_files"] += 1
            stats["deleted_bytes"] += file_size

            logger.info(
                f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'}: "
                f"{file_path.name} ({format_size(file_size)}, {file_age:.1f} days old)"
            )

            if not dry_run:
                try:
                    file_path.unlink()
                except Exception as e:
                    stats["errors"].append(f"{file_path.name}: {str(e)}")
                    logger.error(f"Failed to delete {file_path}: {e}")
        else:
            # File is recent, keep it
            stats["kept_files"] += 1
            stats["kept_bytes"] += file_size

    # Log summary
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files scanned:  {stats['total_files']}")
    logger.info(f"Files deleted:        {stats['deleted_files']} ({format_size(stats['deleted_bytes'])})")
    logger.info(f"Files kept:           {stats['kept_files']} ({format_size(stats['kept_bytes'])})")

    if stats["errors"]:
        logger.warning(f"Errors encountered:   {len(stats['errors'])}")
        for error in stats["errors"]:
            logger.warning(f"  - {error}")

    logger.info("=" * 60)

    return stats


def cleanup_logs(
    logs_dir: str,
    max_age_days: int = 30,
    dry_run: bool = False
) -> dict:
    """
    Remove log files older than max_age_days.
    """
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        return {"status": "skipped", "reason": "directory_not_found"}

    log_extensions = {'.log', '.txt', '.json'}
    stats = {"deleted": 0, "kept": 0}

    for file_path in logs_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in log_extensions:
            continue

        file_age = get_file_age_days(file_path)

        if file_age > max_age_days:
            stats["deleted"] += 1
            if not dry_run:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete log {file_path}: {e}")
        else:
            stats["kept"] += 1

    logger.info(f"Log cleanup: deleted {stats['deleted']}, kept {stats['kept']}")
    return stats


def main():
    """Main entry point for cleanup worker"""
    # Configuration from environment
    video_dir = os.getenv("VIDEO_OUTPUT_DIR", "/var/data/videos")
    logs_dir = os.getenv("LOGS_DIR", "/var/data/logs")
    max_age_days = int(os.getenv("MAX_AGE_DAYS", "7"))
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    logger.info("=" * 60)
    logger.info("RAGNAROK VIDEO CLEANUP WORKER")
    logger.info("=" * 60)
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Logs directory:  {logs_dir}")
    logger.info(f"Max age (days):  {max_age_days}")
    logger.info(f"Dry run:         {dry_run}")
    logger.info(f"Started at:      {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Clean videos
    video_stats = cleanup_videos(video_dir, max_age_days, dry_run)

    # Clean logs (keep for 30 days)
    log_stats = cleanup_logs(logs_dir, max_age_days=30, dry_run=dry_run)

    # Report disk usage
    try:
        import shutil
        total, used, free = shutil.disk_usage("/var/data")
        logger.info(f"Disk usage: {format_size(used)} used / {format_size(total)} total ({format_size(free)} free)")
    except Exception as e:
        logger.warning(f"Could not get disk usage: {e}")

    logger.info(f"Cleanup completed at {datetime.now().isoformat()}")

    # Exit with error if any deletions failed
    if video_stats.get("errors"):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
