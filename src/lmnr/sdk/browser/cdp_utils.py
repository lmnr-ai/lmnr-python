import asyncio
import orjson
import os
import time

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
lock = asyncio.Lock()

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "recorder", "record.umd.min.cjs"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

INJECT_PLACEHOLDER = """
(maskInputOptions) => {
    const BATCH_TIMEOUT = 2000; // Send events after 2 seconds
    const MAX_WORKER_PROMISES = 50; // Max concurrent worker promises
    const HEARTBEAT_INTERVAL = 2000;
    const CHUNK_SIZE = 256 * 1024; // 256KB chunks
    const CHUNK_SEND_DELAY = 100; // 100ms delay between chunks

    window.lmnrRrwebEventsBatch = [];
    window.lmnrChunkQueue = [];
    window.lmnrChunkSequence = 0;
    window.lmnrCurrentBatchId = null;

    // Create a Web Worker for heavy JSON processing with chunked processing
    const createCompressionWorker = () => {
        const workerCode = `
            self.onmessage = async function(e) {
                const { jsonString, buffer, id, useBuffer } = e.data;
                try {
                    let uint8Array;

                    if (useBuffer && buffer) {
                        // Use transferred ArrayBuffer (no copying needed!)
                        uint8Array = new Uint8Array(buffer);
                    } else {
                        // Convert JSON string to bytes
                        const textEncoder = new TextEncoder();
                        uint8Array = textEncoder.encode(jsonString);
                    }

                    const compressionStream = new CompressionStream('gzip');
                    const writer = compressionStream.writable.getWriter();
                    const reader = compressionStream.readable.getReader();

                    writer.write(uint8Array);
                    writer.close();

                    const chunks = [];
                    let totalLength = 0;

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        chunks.push(value);
                        totalLength += value.length;
                    }

                    const compressedData = new Uint8Array(totalLength);
                    let offset = 0;
                    for (const chunk of chunks) {
                        compressedData.set(chunk, offset);
                        offset += chunk.length;
                    }

                    self.postMessage({ id, success: true, data: compressedData });
                } catch (error) {
                    self.postMessage({ id, success: false, error: error.message });
                }
            };
        `;

        const blob = new Blob([workerCode], { type: 'application/javascript' });
        return new Worker(URL.createObjectURL(blob));
    };

    let compressionWorker = null;
    let workerPromises = new Map();
    let workerId = 0;
    let workerSupported = null; // null = unknown, true = supported, false = blocked by CSP

    // Test if workers are supported (not blocked by CSP)
    function testWorkerSupport() {
        if (workerSupported !== null) {
            return workerSupported;
        }
        
        try {
            const testWorker = createCompressionWorker();
            testWorker.terminate();
            workerSupported = true;
            return true;
        } catch (error) {
            console.warn('Web Workers blocked by CSP, will use main thread compression:', error);
            workerSupported = false;
            return false;
        }
    }

    // Cleanup function for worker
    const cleanupWorker = () => {
        if (compressionWorker) {
            compressionWorker.terminate();
            compressionWorker = null;
        }
        workerPromises.clear();
        workerId = 0;
    };

    // Clean up stale promises to prevent memory leaks
    const cleanupStalePromises = () => {
        if (workerPromises.size > MAX_WORKER_PROMISES) {
            const toDelete = [];
            for (const [id, promise] of workerPromises) {
                if (toDelete.length >= workerPromises.size - MAX_WORKER_PROMISES) break;
                toDelete.push(id);
                promise.reject(new Error('Promise cleaned up due to memory pressure'));
            }
            toDelete.forEach(id => workerPromises.delete(id));
        }
    };

    // Non-blocking JSON.stringify using chunked processing
    function stringifyNonBlocking(obj, chunkSize = 10000) {
        return new Promise((resolve, reject) => {
            try {
                // For very large objects, we need to be more careful
                // Use requestIdleCallback if available, otherwise setTimeout
                const scheduleWork = window.requestIdleCallback ||
                    ((cb) => setTimeout(cb, 0));

                let result = '';
                let keys = [];
                let keyIndex = 0;

                // Pre-process to get all keys if it's an object
                if (typeof obj === 'object' && obj !== null && !Array.isArray(obj)) {
                    keys = Object.keys(obj);
                }

                function processChunk() {
                    try {
                        if (Array.isArray(obj) || typeof obj !== 'object' || obj === null) {
                            // For arrays and primitives, just stringify directly
                            result = JSON.stringify(obj);
                            resolve(result);
                            return;
                        }

                        // For objects, process in chunks
                        const endIndex = Math.min(keyIndex + chunkSize, keys.length);

                        if (keyIndex === 0) {
                            result = '{';
                        }

                        for (let i = keyIndex; i < endIndex; i++) {
                            const key = keys[i];
                            const value = obj[key];

                            if (i > 0) result += ',';
                            result += JSON.stringify(key) + ':' + JSON.stringify(value);
                        }

                        keyIndex = endIndex;

                        if (keyIndex >= keys.length) {
                            result += '}';
                            resolve(result);
                        } else {
                            // Schedule next chunk
                            scheduleWork(processChunk);
                        }
                    } catch (error) {
                        reject(error);
                    }
                }

                processChunk();
            } catch (error) {
                reject(error);
            }
        });
    }

    // Fast compression for small objects (main thread)
    async function compressSmallObject(data) {
        const jsonString = JSON.stringify(data);
        const textEncoder = new TextEncoder();
        const uint8Array = textEncoder.encode(jsonString);

        const compressionStream = new CompressionStream('gzip');
        const writer = compressionStream.writable.getWriter();
        const reader = compressionStream.readable.getReader();

        writer.write(uint8Array);
        writer.close();

        const chunks = [];
        let totalLength = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            totalLength += value.length;
        }

        const compressedData = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
            compressedData.set(chunk, offset);
            offset += chunk.length;
        }

        return compressedData;
    }

    // Alternative: Use transferable objects for maximum efficiency
    async function compressLargeObjectTransferable(data) {
        try {
            // Check if workers are supported first
            if (!testWorkerSupport()) {
                return compressSmallObject(data);
            }
            
            // Clean up stale promises first
            cleanupStalePromises();
            
            // Stringify on main thread but non-blocking
            const jsonString = await stringifyNonBlocking(data);

            // Convert to ArrayBuffer (transferable)
            const encoder = new TextEncoder();
            const uint8Array = encoder.encode(jsonString);
            const buffer = uint8Array.buffer; // Use the original buffer for transfer

            return new Promise((resolve, reject) => {
                if (!compressionWorker) {
                    compressionWorker = createCompressionWorker();
                    compressionWorker.onmessage = (e) => {
                        const { id, success, data: result, error } = e.data;
                        const promise = workerPromises.get(id);
                        if (promise) {
                            workerPromises.delete(id);
                            if (success) {
                                promise.resolve(result);
                            } else {
                                promise.reject(new Error(error));
                            }
                        }
                    };
                    
                    compressionWorker.onerror = (error) => {
                        console.error('Compression worker error:', error);
                        cleanupWorker();
                        compressSmallObject(data).then(resolve, reject);
                    };
                }

                const id = ++workerId;
                workerPromises.set(id, { resolve, reject });

                // Set timeout to prevent hanging promises
                setTimeout(() => {
                    if (workerPromises.has(id)) {
                        workerPromises.delete(id);
                        reject(new Error('Compression timeout'));
                    }
                }, 10000);

                // Transfer the ArrayBuffer (no copying!)
                compressionWorker.postMessage({
                    buffer,
                    id,
                    useBuffer: true
                }, [buffer]);
            });
        } catch (error) {
            console.warn('Failed to process large object with transferable:', error);
            return compressSmallObject(data);
        }
    }

    // Worker-based compression for large objects
    async function compressLargeObject(data) {
        // Check if workers are supported first - if not, use main thread compression
        if (!testWorkerSupport()) {
            return await compressSmallObject(data);
        }
        
        try {
            // Use transferable objects for better performance
            return await compressLargeObjectTransferable(data);
        } catch (error) {
            console.warn('Transferable failed, falling back to string method:', error);
            try {
                // Fallback to string method with worker
                const jsonString = await stringifyNonBlocking(data);

                return new Promise((resolve, reject) => {
                    if (!compressionWorker) {
                        compressionWorker = createCompressionWorker();
                        compressionWorker.onmessage = (e) => {
                            const { id, success, data: result, error } = e.data;
                            const promise = workerPromises.get(id);
                            if (promise) {
                                workerPromises.delete(id);
                                if (success) {
                                    promise.resolve(result);
                                } else {
                                    promise.reject(new Error(error));
                                }
                            }
                        };
                        
                        compressionWorker.onerror = (error) => {
                            console.error('Compression worker error:', error);
                            cleanupWorker();
                        };
                    }

                    const id = ++workerId;
                    workerPromises.set(id, { resolve, reject });
                    
                    // Set timeout to prevent hanging promises
                    setTimeout(() => {
                        if (workerPromises.has(id)) {
                            workerPromises.delete(id);
                            reject(new Error('Compression timeout'));
                        }
                    }, 10000);
                    
                    compressionWorker.postMessage({ jsonString, id });
                });
            } catch (workerError) {
                console.warn('Worker creation failed, falling back to main thread compression:', workerError);
                // Final fallback: compress on main thread (may block UI but will work)
                return await compressSmallObject(data);
            }
        }
    }

    
    setInterval(cleanupWorker, 5000);
    
    function isLargeEvent(type) {
        const LARGE_EVENT_TYPES = [
            2, // FullSnapshot
        ];

        if (LARGE_EVENT_TYPES.includes(type)) {
            return true;
        }

        return false;
    }

    // Create chunks from a string with metadata
    function createChunks(str, batchId) {
        const chunks = [];
        const totalChunks = Math.ceil(str.length / CHUNK_SIZE);
        
        for (let i = 0; i < str.length; i += CHUNK_SIZE) {
            const chunk = str.slice(i, i + CHUNK_SIZE);
            chunks.push({
                batchId: batchId,
                chunkIndex: chunks.length,
                totalChunks: totalChunks,
                data: chunk,
                isFinal: chunks.length === totalChunks - 1
            });
        }
        
        return chunks;
    }

    // Send chunks with flow control
    async function sendChunks(chunks) {
        if (typeof window.lmnrSendEvents !== 'function') {
            return;
        }

        window.lmnrChunkQueue.push(...chunks);
        
        // Process queue
        while (window.lmnrChunkQueue.length > 0) {
            const chunk = window.lmnrChunkQueue.shift();
            try {
                window.lmnrSendEvents(JSON.stringify(chunk));
                // Small delay between chunks to avoid overwhelming CDP
                await new Promise(resolve => setTimeout(resolve, CHUNK_SEND_DELAY));
            } catch (error) {
                console.error('Failed to send chunk:', error);
                // On error, clear failed chunk batch from queue
                window.lmnrChunkQueue = window.lmnrChunkQueue.filter(c => c.batchId !== chunk.batchId);
                break;
            }
        }
    }

    async function sendBatchIfReady() {
        if (window.lmnrRrwebEventsBatch.length > 0 && typeof window.lmnrSendEvents === 'function') {
            const events = window.lmnrRrwebEventsBatch;
            window.lmnrRrwebEventsBatch = [];

            try {
                // Generate unique batch ID
                const batchId = `${Date.now()}_${window.lmnrChunkSequence++}`;
                window.lmnrCurrentBatchId = batchId;
                
                // Stringify the entire batch
                const batchString = JSON.stringify(events);
                
                // Check size and chunk if necessary
                if (batchString.length <= CHUNK_SIZE) {
                    // Small enough to send as single chunk
                    const chunk = {
                        batchId: batchId,
                        chunkIndex: 0,
                        totalChunks: 1,
                        data: batchString,
                        isFinal: true
                    };
                    window.lmnrSendEvents(JSON.stringify(chunk));
                } else {
                    // Need to chunk
                    const chunks = createChunks(batchString, batchId);
                    await sendChunks(chunks);
                }
            } catch (error) {
                console.error('Failed to send events:', error);
                // Clear batch to prevent memory buildup
                window.lmnrRrwebEventsBatch = [];
            }
        }
    }

    async function bufferToBase64(buffer) {
        const base64url = await new Promise(r => {
            const reader = new FileReader()
            reader.onload = () => r(reader.result)
            reader.readAsDataURL(new Blob([buffer]))
        });
        return base64url.slice(base64url.indexOf(',') + 1);
    }

    if (!window.lmnrStartedRecordingEvents) {
        setInterval(sendBatchIfReady, BATCH_TIMEOUT);

        window.lmnrRrweb.record({
            async emit(event) {
                try {
                    const isLarge = isLargeEvent(event.type);
                    const compressedResult = isLarge ?
                        await compressLargeObject(event.data) :
                        await compressSmallObject(event.data);

                    const base64Data = await bufferToBase64(compressedResult);
                    const eventToSend = {
                        ...event,
                        data: base64Data,
                    };
                    window.lmnrRrwebEventsBatch.push(eventToSend);
                } catch (error) {
                    console.warn('Failed to push event to batch', error);
                }
            },
            recordCanvas: true,
            collectFonts: true,
            recordCrossOriginIframes: true,
            maskInputOptions: {
                password: true,
                textarea: maskInputOptions.textarea || false,
                text: maskInputOptions.text || false,
                number: maskInputOptions.number || false,
                select: maskInputOptions.select || false,
                email: maskInputOptions.email || false,
                tel: maskInputOptions.tel || false,
            }
        });

        function heartbeat() {
            // Add heartbeat events
            setInterval(
                () => {
                    window.lmnrRrweb.record.addCustomEvent('heartbeat', {
                        title: document.title,
                        url: document.URL,
                    })
                },
                HEARTBEAT_INTERVAL,
            );
        }

        heartbeat();
        window.lmnrStartedRecordingEvents = true;
    }
}
"""


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
    async with lock:
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
                        "expression": f"({INJECT_PLACEHOLDER})({orjson.dumps(get_mask_input_setting()).decode('utf-8')})",
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
