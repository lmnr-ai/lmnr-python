import asyncio
import orjson
import os
import threading
import time

from weakref import WeakKeyDictionary

from opentelemetry import trace

from lmnr.sdk.decorators import observe
from lmnr.sdk.browser.utils import retry_async
from lmnr.sdk.browser.background_send_events import (
    get_background_loop,
    track_async_send,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import MaskInputOptions

logger = get_default_logger(__name__)

OLD_BUFFER_TIMEOUT = 60
CDP_OPERATION_TIMEOUT_SECONDS = 10

# CDP ContextId is int
frame_to_isolated_context_id: dict[str, int] = {}

# Store locks per event loop to avoid pytest-asyncio issues
_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()
_locks_lock = threading.Lock()
_fallback_lock = asyncio.Lock()


def get_lock() -> asyncio.Lock:
    """Get or create a lock for the current event loop.

    This ensures each event loop gets its own lock instance, avoiding
    cross-event-loop binding issues that occur in pytest-asyncio when
    tests run in different event loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running
        logger.warning("No event loop running, using fallback lock")
        return _fallback_lock

    with _locks_lock:
        if loop not in _locks:
            _locks[loop] = asyncio.Lock()
        return _locks[loop]


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "recorder", "record.umd.min.cjs"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

with open(os.path.join(current_dir, "inject_script.js"), "r") as f:
    INJECT_SCRIPT_CONTENT = f.read()


async def should_skip_page(cdp_session):
    """Checks if the page url is an error page or an empty page.
    This function returns True in case of any error in our code, because
    it is safer to not record events than to try to inject the recorder
    into something that is already broken.
    """
    cdp_client = cdp_session.cdp_client

    try:
        # Get the current page URL
        result = await asyncio.wait_for(
            cdp_client.send.Runtime.evaluate(
                {
                    "expression": "window.location.href",
                    "returnByValue": True,
                },
                session_id=cdp_session.session_id,
            ),
            timeout=CDP_OPERATION_TIMEOUT_SECONDS,
        )

        url = result.get("result", {}).get("value", "")

        # Comprehensive list of browser error URLs
        error_url_patterns = [
            "about:blank",
            # Chrome error pages
            "chrome-error://",
            "chrome://network-error/",
            "chrome://network-errors/",
            # Chrome crash and debugging pages
            "chrome://crash/",
            "chrome://crashdump/",
            "chrome://kill/",
            "chrome://hang/",
            "chrome://shorthang/",
            "chrome://gpuclean/",
            "chrome://gpucrash/",
            "chrome://gpuhang/",
            "chrome://memory-exhaust/",
            "chrome://memory-pressure-critical/",
            "chrome://memory-pressure-moderate/",
            "chrome://inducebrowsercrashforrealz/",
            "chrome://inducebrowserdcheckforrealz/",
            "chrome://inducebrowserheapcorruption/",
            "chrome://heapcorruptioncrash/",
            "chrome://badcastcrash/",
            "chrome://ppapiflashcrash/",
            "chrome://ppapiflashhang/",
            "chrome://quit/",
            "chrome://restart/",
            # Firefox error pages
            "about:neterror",
            "about:certerror",
            "about:blocked",
            # Firefox crash and debugging pages
            "about:crashcontent",
            "about:crashparent",
            "about:crashes",
            "about:tabcrashed",
            # Edge error pages (similar to Chrome)
            "edge-error://",
            "edge://crash/",
            "edge://kill/",
            "edge://hang/",
            # Safari/WebKit error indicators (data URLs with error content)
            "webkit-error://",
        ]

        # Check if current URL matches any error pattern
        if any(url.startswith(pattern) for pattern in error_url_patterns):
            logger.debug(f"Detected browser error page from URL: {url}")
            return True

        # Additional check for data URLs that might contain error pages
        if url.startswith("data:") and any(
            error_term in url.lower()
            for error_term in ["error", "crash", "failed", "unavailable", "not found"]
        ):
            logger.debug(f"Detected error page from data URL: {url[:100]}...")
            return True

        return False

    except asyncio.TimeoutError:
        logger.debug("Timeout error when checking if error page")
        return True
    except Exception as e:
        logger.debug(f"Error during checking if error page: {e}")
        return True


def get_mask_input_setting() -> MaskInputOptions:
    """Get the mask_input setting from session recording configuration."""
    try:
        config = TracerWrapper.get_session_recording_options()
        return config.get(
            "mask_input_options",
            MaskInputOptions(
                textarea=False,
                text=False,
                number=False,
                select=False,
                email=False,
                tel=False,
            ),
        )
    except (AttributeError, Exception):
        # Fallback to default configuration if TracerWrapper is not initialized
        return MaskInputOptions(
            textarea=False,
            text=False,
            number=False,
            select=False,
            email=False,
            tel=False,
        )


# browser_use.browser.session.CDPSession (browser-use >= 0.6.0)
async def get_isolated_context_id(cdp_session) -> int | None:
    async with get_lock():
        tree = {}
        try:
            tree = await asyncio.wait_for(
                cdp_session.cdp_client.send.Page.getFrameTree(
                    session_id=cdp_session.session_id
                ),
                timeout=CDP_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.debug("Timeout error when getting frame tree")
            return None
        except Exception as e:
            logger.debug(f"Failed to get frame tree: {e}")
            return None
        frame = tree.get("frameTree", {}).get("frame", {})
        frame_id = frame.get("id")
        loader_id = frame.get("loaderId")

        if frame_id is None or loader_id is None:
            logger.debug("Failed to get frame id or loader id")
            return None
        key = f"{frame_id}_{loader_id}"

        if key in frame_to_isolated_context_id:
            return frame_to_isolated_context_id[key]

        try:
            result = await asyncio.wait_for(
                cdp_session.cdp_client.send.Page.createIsolatedWorld(
                    {
                        "frameId": frame_id,
                        "worldName": "laminar-isolated-context",
                    },
                    session_id=cdp_session.session_id,
                ),
                timeout=CDP_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.debug("Timeout error when getting isolated context id")
            return None
        except Exception as e:
            logger.debug(f"Failed to get isolated context id: {e}")
            return None
        isolated_context_id = result["executionContextId"]
        frame_to_isolated_context_id[key] = isolated_context_id
        return isolated_context_id


# browser_use.browser.session.CDPSession (browser-use >= 0.6.0)
async def inject_session_recorder(cdp_session) -> int | None:
    """Injects the session recorder base as well as the recorder itself.
    Returns the isolated context id if successful.
    """
    isolated_context_id = None
    cdp_client = cdp_session.cdp_client
    try:
        should_skip = True
        try:
            should_skip = await should_skip_page(cdp_session)
        except Exception as e:
            logger.debug(f"Failed to check if error page: {e}")

        if should_skip:
            logger.debug("Empty page detected, skipping session recorder injection")
            return

        isolated_context_id = await get_isolated_context_id(cdp_session)
        try:
            is_loaded = await is_recorder_present(cdp_session, isolated_context_id)
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if is_loaded:
            return

        if isolated_context_id is None:
            logger.debug("Failed to get isolated context id")
            return

        async def load_session_recorder():
            try:
                await asyncio.wait_for(
                    cdp_client.send.Runtime.evaluate(
                        {
                            "expression": f"({RRWEB_CONTENT})()",
                            "contextId": isolated_context_id,
                        },
                        session_id=cdp_session.session_id,
                    ),
                    timeout=CDP_OPERATION_TIMEOUT_SECONDS,
                )
                return True
            except asyncio.TimeoutError:
                logger.debug("Timeout error when loading session recorder base")
                return False
            except Exception as e:
                logger.debug(f"Failed to load session recorder base: {e}")
                return False

        if not await retry_async(
            load_session_recorder,
            retries=3,
            delay=1,
            error_message="Failed to load session recorder",
        ):
            return

        try:
            await asyncio.wait_for(
                cdp_client.send.Runtime.evaluate(
                    {
                        "expression": f"({INJECT_SCRIPT_CONTENT})({orjson.dumps(get_mask_input_setting()).decode('utf-8')}, true)",
                        "contextId": isolated_context_id,
                    },
                    session_id=cdp_session.session_id,
                ),
                timeout=CDP_OPERATION_TIMEOUT_SECONDS,
            )
            return isolated_context_id
        except asyncio.TimeoutError:
            logger.debug("Timeout error when injecting session recorder")
        except Exception as e:
            logger.debug(f"Failed to inject recorder: {e}")

    except Exception as e:
        logger.debug(f"Error during session recorder injection: {e}")


# browser_use.browser.session.CDPSession (browser-use >= 0.6.0)
@observe(name="cdp_use.session", ignore_input=True, ignore_output=True)
async def start_recording_events(
    cdp_session,
    lmnr_session_id: str,
    client: AsyncLaminarClient,
):
    cdp_client = cdp_session.cdp_client

    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    isolated_context_id = await inject_session_recorder(cdp_session)
    if isolated_context_id is None:
        logger.debug("Failed to inject session recorder, not registering bindings")
        return

    # Get the background loop for async sends (independent of CDP's loop)
    background_loop = get_background_loop()

    # Buffer for reassembling chunks
    chunk_buffers = {}

    async def send_events_from_browser(chunk: dict):
        try:
            # Handle chunked data
            batch_id = chunk["batchId"]
            chunk_index = chunk["chunkIndex"]
            total_chunks = chunk["totalChunks"]
            data = chunk["data"]

            # Initialize buffer for this batch if needed
            if batch_id not in chunk_buffers:
                chunk_buffers[batch_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "timestamp": time.time(),
                }

            # Store chunk
            chunk_buffers[batch_id]["chunks"][chunk_index] = data

            # Check if we have all chunks
            if len(chunk_buffers[batch_id]["chunks"]) == total_chunks:
                # Reassemble the full message
                full_data = ""
                for i in range(total_chunks):
                    full_data += chunk_buffers[batch_id]["chunks"][i]

                # Parse the JSON
                events = orjson.loads(full_data)

                # Send to server in background loop (independent of CDP's loop)
                if events and len(events) > 0:
                    future = asyncio.run_coroutine_threadsafe(
                        client._browser_events.send(lmnr_session_id, trace_id, events),
                        background_loop,
                    )
                    track_async_send(future)

                # Clean up buffer
                del chunk_buffers[batch_id]

            # Clean up old incomplete buffers
            current_time = time.time()
            to_delete = []
            for bid, buffer in chunk_buffers.items():
                if current_time - buffer["timestamp"] > OLD_BUFFER_TIMEOUT:
                    to_delete.append(bid)
            for bid in to_delete:
                logger.debug(f"Cleaning up incomplete chunk buffer: {bid}")
                del chunk_buffers[bid]

        except Exception as e:
            logger.debug(f"Could not send events: {e}")

    # cdp_use.cdp.runtime.events.BindingCalledEvent
    async def send_events_callback(event, cdp_session_id: str | None = None):
        if event["name"] != "lmnrSendEvents":
            return
        if event["executionContextId"] != isolated_context_id:
            return
        asyncio.create_task(send_events_from_browser(orjson.loads(event["payload"])))

    await cdp_client.send.Runtime.addBinding(
        {
            "name": "lmnrSendEvents",
            "executionContextId": isolated_context_id,
        },
        session_id=cdp_session.session_id,
    )
    cdp_client.register.Runtime.bindingCalled(send_events_callback)

    await enable_target_discovery(cdp_session)
    register_on_target_created(cdp_session, lmnr_session_id, client)


# browser_use.browser.session.CDPSession (browser-use >= 0.6.0)
async def enable_target_discovery(cdp_session):
    cdp_client = cdp_session.cdp_client
    await cdp_client.send.Target.setDiscoverTargets(
        {
            "discover": True,
        },
        session_id=cdp_session.session_id,
    )


# browser_use.browser.session.CDPSession (browser-use >= 0.6.0)
def register_on_target_created(
    cdp_session, lmnr_session_id: str, client: AsyncLaminarClient
):
    # cdp_use.cdp.target.events.TargetCreatedEvent
    def on_target_created(event, cdp_session_id: str | None = None):
        target_info = event["targetInfo"]
        if target_info["type"] == "page":
            asyncio.create_task(inject_session_recorder(cdp_session=cdp_session))

    try:
        cdp_session.cdp_client.register.Target.targetCreated(on_target_created)
    except Exception as e:
        logger.debug(f"Failed to register on target created: {e}")


# browser_use.browser.session.CDPSession (browser-use >= 0.6.0)
async def is_recorder_present(
    cdp_session, isolated_context_id: int | None = None
) -> bool:
    # This function returns True on any error, because it is safer to not record
    # events than to try to inject the recorder into a broken context.
    cdp_client = cdp_session.cdp_client
    if isolated_context_id is None:
        isolated_context_id = await get_isolated_context_id(cdp_session)
    if isolated_context_id is None:
        logger.debug("Failed to get isolated context id")
        return True

    try:
        result = await asyncio.wait_for(
            cdp_client.send.Runtime.evaluate(
                {
                    "expression": "typeof window.lmnrRrweb !== 'undefined'",
                    "contextId": isolated_context_id,
                },
                session_id=cdp_session.session_id,
            ),
            timeout=CDP_OPERATION_TIMEOUT_SECONDS,
        )
        if result and "result" in result and "value" in result["result"]:
            return result["result"]["value"]
        return False
    except asyncio.TimeoutError:
        logger.debug("Timeout error when checking if session recorder is present")
        return True
    except Exception:
        logger.debug("Exception when checking if session recorder is present")
        return True


async def take_full_snapshot(cdp_session):
    cdp_client = cdp_session.cdp_client
    isolated_context_id = await get_isolated_context_id(cdp_session)
    if isolated_context_id is None:
        logger.debug("Failed to get isolated context id")
        return False

    if await should_skip_page(cdp_session):
        logger.debug("Skipping full snapshot")
        return False

    try:
        result = await asyncio.wait_for(
            cdp_client.send.Runtime.evaluate(
                {
                    "expression": """(() => {
    if (window.lmnrRrweb) {
        try {
            window.lmnrRrweb.record.takeFullSnapshot();
            return true;
        } catch (e) {
            console.error("Error taking full snapshot:", e);
            return false;
        }
    }
    return false;
})()""",
                    "contextId": isolated_context_id,
                },
                session_id=cdp_session.session_id,
            ),
            timeout=CDP_OPERATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.debug("Timeout error when taking full snapshot")
        return False
    except Exception as e:
        logger.debug(f"Error when taking full snapshot: {e}")
        return False
    if result and "result" in result and "value" in result["result"]:
        return result["result"]["value"]
    return False
