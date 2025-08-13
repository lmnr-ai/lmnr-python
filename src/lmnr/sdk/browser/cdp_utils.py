import asyncio
import orjson
import logging
import os
import time

from opentelemetry import trace

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.decorators import observe
from lmnr.sdk.browser.utils import retry_sync, retry_async
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.sdk.types import MaskInputOptions

try:
    if is_package_installed("cdp-use"):
        from cdp_use.cdp.runtime.events import BindingCalledEvent
        from cdp_use.client import CDPClient
    else:
        raise ImportError(
            "Attempted to import lmnr.sdk.browser.cdp_utils, but cdp-use is not installed. "
            "Use `pip install cdp-use` to install it."
        )
except ImportError as e:
    raise ImportError(
        "Attempted to import lmnr.sdk.browser.cdp_utils, but cdp-use is not installed. "
        "Use `pip install cdp-use` to install it."
    ) from e

try:
    if is_package_installed("browser-use"):
        from browser_use.browser import BrowserSession
    else:
        raise ImportError(
            "Attempted to import lmnr.sdk.browser.cdp_utils, but browser_use is not installed. "
            "Use `pip install browser_use` to install it."
        )
except ImportError as e:
    raise ImportError(
        "Attempted to import lmnr.sdk.browser.cdp_utils, but browser_use is not installed. "
        "Use `pip install browser_use` to install it."
    ) from e

logger = logging.getLogger(__name__)

OLD_BUFFER_TIMEOUT = 60

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "recorder", "record.umd.min.cjs"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

INJECT_PLACEHOLDER = """
(mask_input_options) => {
    const BATCH_TIMEOUT = 2000; // Send events after 2 seconds
    const MAX_WORKER_PROMISES = 50; // Max concurrent worker promises
    const HEARTBEAT_INTERVAL = 1000;
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
    async function compressLargeObject(data, isLarge = true) {
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
            3, // IncrementalSnapshot
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
                await window.lmnrSendEvents(chunk);
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
                    await window.lmnrSendEvents(chunk);
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

    setInterval(sendBatchIfReady, BATCH_TIMEOUT);

    async function bufferToBase64(buffer) {
        const base64url = await new Promise(r => {
            const reader = new FileReader()
            reader.onload = () => r(reader.result)
            reader.readAsDataURL(new Blob([buffer]))
        });
        return base64url.slice(base64url.indexOf(',') + 1);
    }
 
    window.lmnrRrweb.record({
        async emit(event) {
            try {
                const isLarge = isLargeEvent(event.type);
                const compressedResult = isLarge ?
                    await compressLargeObject(event.data, true) :
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
            textarea: mask_input_options.textarea || false,
            text: mask_input_options.text || false,
            number: mask_input_options.number || false,
            select: mask_input_options.select || false,
            email: mask_input_options.email || false,
            tel: mask_input_options.tel || false,
        }
    });

    function heartbeat() {
        // Add heartbeat events
        setInterval(() => {
            window.lmnrRrweb.record.addCustomEvent('heartbeat', {
                title: document.title,
                    url: document.URL,
                })
            }, HEARTBEAT_INTERVAL
        );
    }

    heartbeat();

}
"""


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


def inject_session_recorder_sync(session: BrowserSession):
    cdp_session = asyncio.run(session.get_or_create_cdp_session())
    cdp_client = cdp_session.cdp_client
    try:
        try:
            result = asyncio.run(
                session.cdp_client.send.Runtime.callFunctionOn(
                    {
                        "functionDeclaration": """() => typeof window.lmnrRrweb !== 'undefined'""",
                    },
                )
            )
            if result and "result" in result and "value" in result["result"]:
                is_loaded = result["result"]["value"]
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            def load_session_recorder():
                try:
                    asyncio.run(
                        cdp_client.send.Runtime.callFunctionOn(
                            {
                                "functionDeclaration": RRWEB_CONTENT,
                            },
                        )
                    )
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load session recorder: {e}")
                    return False

            if not retry_sync(
                load_session_recorder,
                delay=1,
                error_message="Failed to load session recorder",
            ):
                return

            try:
                asyncio.run(
                    cdp_client.send.Runtime.callFunctionOn(
                        {
                            "functionDeclaration": INJECT_PLACEHOLDER,
                            "arguments": [{"value": get_mask_input_setting()}],
                        },
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to inject session recorder: {e}")

    except Exception as e:
        logger.error(f"Error during session recorder injection: {e}")


async def inject_session_recorder_async(session: BrowserSession):
    cdp_session = await session.get_or_create_cdp_session()
    cdp_client = cdp_session.cdp_client
    try:
        try:
            result = await cdp_client.send.Runtime.callFunctionOn(
                {
                    "functionDeclaration": """() => typeof window.lmnrRrweb !== 'undefined'""",
                },
            )
            if result and "result" in result and "value" in result["result"]:
                is_loaded = result["result"]["value"]
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            async def load_session_recorder():
                try:
                    await cdp_client.send.Runtime.callFunctionOn(
                        {
                            "functionDeclaration": RRWEB_CONTENT,
                            "awaitPromise": True,
                        },
                    )
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load session recorder: {e}")
                    return False

            if not await retry_async(
                load_session_recorder,
                delay=1,
                error_message="Failed to load session recorder",
            ):
                return

            try:
                await cdp_client.send.Runtime.callFunctionOn(
                    {
                        "functionDeclaration": INJECT_PLACEHOLDER,
                        "arguments": [{"value": get_mask_input_setting()}],
                        "awaitPromise": True,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to inject session recorder placeholder: {e}")

    except Exception as e:
        logger.error(f"Error during session recorder injection: {e}")


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
def start_recording_events_sync(
    session: BrowserSession, session_id: str, client: LaminarClient
):
    cdp_session = asyncio.run(session.get_or_create_cdp_session())
    cdp_client = cdp_session.cdp_client

    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    # Buffer for reassembling chunks
    chunk_buffers = {}

    def send_events_from_browser(chunk):
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
                full_data = "".join(
                    chunk_buffers[batch_id]["chunks"][i] for i in range(total_chunks)
                )

                # Parse the JSON
                events = orjson.loads(full_data)

                # Send to server
                if events and len(events) > 0:
                    client._browser_events.send(session_id, trace_id, events)

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

    try:

        def send_events_callback(event: BindingCalledEvent):
            send_events_from_browser(orjson.loads(event["payload"]))

        asyncio.run(
            cdp_client.send.Runtime.addBinding(
                {
                    "name": "lmnrSendEvents",
                },
            )
        )
        cdp_client.register.Runtime.bindingCalled(
            {
                "name": "lmnrSendEvents",
            },
            send_events_callback,
        )
    except Exception as e:
        logger.debug(f"Could not expose function: {e}")

    inject_session_recorder_sync(session)

    # def on_load(p):
    #     try:
    #         inject_session_recorder_sync(p)
    #     except Exception as e:
    #         logger.error(f"Error in on_load handler: {e}")

    # page.on("domcontentloaded", on_load)


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
async def start_recording_events_async(
    session: BrowserSession, session_id: str, client: AsyncLaminarClient
):
    cdp_session = await session.get_or_create_cdp_session()
    cdp_client = cdp_session.cdp_client

    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    # Buffer for reassembling chunks
    chunk_buffers = {}

    async def send_events_from_browser(chunk):
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

                # Send to server
                if events and len(events) > 0:
                    await client._browser_events.send(session_id, trace_id, events)

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

    try:

        async def send_events_callback(event: BindingCalledEvent):
            send_events_from_browser(orjson.loads(event["payload"]))

        print("adding binding")
        await cdp_client.send.Runtime.addBinding(
            {
                "name": "lmnrSendEvents",
            },
        )
        print("binding added")
        cdp_client.register.Runtime.bindingCalled(
            {
                "name": "lmnrSendEvents",
            },
            send_events_callback,
        )
        print("binding registered")
    except Exception as e:
        logger.debug(f"Could not expose function: {e}")

    print("injecting session recorder")
    await inject_session_recorder_async(session)

    # async def on_load(p):
    #     try:
    #         await inject_session_recorder_async(p)
    #     except Exception as e:
    #         logger.error(f"Error in on_load handler: {e}")

    # page.on("domcontentloaded", on_load)


def take_full_snapshot(cdp_client: CDPClient):
    result = asyncio.run(
        cdp_client.send.Runtime.callFunctionOn(
            """() => {
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
    }"""
        )
    )
    if result and "result" in result and "value" in result["result"]:
        return result["result"]["value"]
    return False


async def take_full_snapshot_async(cdp_client: CDPClient):
    result = await cdp_client.send.Runtime.callFunctionOn(
        """() => {
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
    }"""
    )
    if result and "result" in result and "value" in result["result"]:
        return result["result"]["value"]
    return False
