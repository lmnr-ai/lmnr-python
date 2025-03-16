import asyncio
import logging
import time

logger = logging.getLogger(__name__)

INJECT_PLACEHOLDER = """
() => {
    const BATCH_SIZE = 1000;  // Maximum events to store in memory
    
    window.lmnrRrwebEventsBatch = [];
    
    // Utility function to compress individual event data
    async function compressEventData(data) {
        const jsonString = JSON.stringify(data);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const compressedStream = blob.stream().pipeThrough(new CompressionStream('gzip'));
        const compressedResponse = new Response(compressedStream);
        const compressedData = await compressedResponse.arrayBuffer();
        return Array.from(new Uint8Array(compressedData));
    }
    
    window.lmnrGetAndClearEvents = () => {
        const events = window.lmnrRrwebEventsBatch;
        window.lmnrRrwebEventsBatch = [];
        return events;
    };

    // Add heartbeat events
    setInterval(async () => {
        const heartbeat = {
            type: 6,
            data: await compressEventData({ source: 'heartbeat' }),
            timestamp: Date.now()
        };
        
        window.lmnrRrwebEventsBatch.push(heartbeat);
        
        // Prevent memory issues by limiting batch size
        if (window.lmnrRrwebEventsBatch.length > BATCH_SIZE) {
            window.lmnrRrwebEventsBatch = window.lmnrRrwebEventsBatch.slice(-BATCH_SIZE);
        }
    }, 1000);

    window.lmnrRrweb.record({
        async emit(event) {
            // Compress the data field
            const compressedEvent = {
                ...event,
                data: await compressEventData(event.data)
            };
            window.lmnrRrwebEventsBatch.push(compressedEvent);
        }
    });
}
"""


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def retry_sync(func, retries=5, delay=0.5, error_message="Operation failed"):
    """Utility function for retry logic in synchronous operations"""
    for attempt in range(retries):
        try:
            result = func()
            if result:  # If function returns truthy value, consider it successful
                return result
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message} after all retries")
                return None
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message}: {e}")
                return None
        time.sleep(delay)
    return None


async def retry_async(func, retries=5, delay=0.5, error_message="Operation failed"):
    """Utility function for retry logic in asynchronous operations"""
    for attempt in range(retries):
        try:
            result = await func()
            if result:  # If function returns truthy value, consider it successful
                return result
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message} after all retries")
                return None
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message}: {e}")
                return None
        await asyncio.sleep(delay)
    return None
