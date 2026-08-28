import asyncio
import logging

from pymongo.errors import PyMongoError

from services.evaluation_history_service import (
    cleanup_expired_evaluations,
    cleanup_orphaned_evaluation_artifacts,
)
from services.individual_analysis_history_service import protected_individual_result_ids
from utils.result_files import cleanup_old_results

logger = logging.getLogger(__name__)
RESULT_CLEANUP_INTERVAL_SECONDS = 60


async def cleanup_temporary_results_once() -> None:
    active_result_ids = await cleanup_expired_evaluations()
    individual_result_ids = await protected_individual_result_ids()
    protected_result_ids = active_result_ids | individual_result_ids
    await asyncio.to_thread(
        cleanup_old_results,
        protected_result_ids=protected_result_ids,
    )
    await asyncio.to_thread(
        cleanup_orphaned_evaluation_artifacts,
        active_result_ids,
    )


async def run_result_cleanup_loop() -> None:
    while True:
        try:
            await asyncio.sleep(RESULT_CLEANUP_INTERVAL_SECONDS)
            await cleanup_temporary_results_once()
        except asyncio.CancelledError:
            raise
        except (OSError, PyMongoError, ValueError):
            logger.exception("temporary_result_cleanup_failed")
