import logging
import os

from opentelemetry import trace

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.decorators import observe
from lmnr.sdk.browser.utils import retry_sync, retry_async
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.opentelemetry_lib.tracing.context import get_current_context

try:
    if is_package_installed("playwright"):
        from playwright.async_api import Page
        from playwright.sync_api import Page as SyncPage
    elif is_package_installed("patchright"):
        from patchright.async_api import Page
        from patchright.sync_api import Page as SyncPage
    else:
        raise ImportError(
            "Attempted to import lmnr.sdk.browser.pw_utils, but neither "
            "playwright nor patchright is installed. Use `pip install playwright` "
            "or `pip install patchright` to install one of the supported browsers."
        )
except ImportError as e:
    raise ImportError(
        "Attempted to import lmnr.sdk.browser.pw_utils, but neither "
        "playwright nor patchright is installed. Use `pip install playwright` "
        "or `pip install patchright` to install one of the supported browsers."
    ) from e

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "recorder", "record.umd.min.cjs"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

INJECT_PLACEHOLDER = """
() => {
    const BATCH_TIMEOUT = 2000; // Send events after 2 seconds
    const MAX_WORKER_PROMISES = 50; // Max concurrent worker promises
    const HEARTBEAT_INTERVAL = 1000;
    
    window.lmnrRrwebEventsBatch = [];

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
        try {
            // Use transferable objects for better performance
            return await compressLargeObjectTransferable(data);
        } catch (error) {
            console.warn('Transferable failed, falling back to string method:', error);
            // Fallback to string method
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

    async function sendBatchIfReady() {
        if (window.lmnrRrwebEventsBatch.length > 0 && typeof window.lmnrSendEvents === 'function') {
            const events = window.lmnrRrwebEventsBatch;
            window.lmnrRrwebEventsBatch = [];

            try {
                await window.lmnrSendEvents(events);
            } catch (error) {
                console.error('Failed to send events:', error);
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
        recordCrossOriginIframes: true
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


def inject_session_recorder_sync(page: SyncPage):
    try:
        try:
            is_loaded = page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            )
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            def load_session_recorder():
                try:
                    page.evaluate(RRWEB_CONTENT)
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
                page.evaluate(INJECT_PLACEHOLDER)
            except Exception as e:
                logger.debug(f"Failed to inject session recorder: {e}")

    except Exception as e:
        logger.error(f"Error during session recorder injection: {e}")


async def inject_session_recorder_async(page: Page):
    try:
        try:
            is_loaded = await page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            )
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            async def load_session_recorder():
                try:
                    await page.evaluate(RRWEB_CONTENT)
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
                await page.evaluate(INJECT_PLACEHOLDER)
            except Exception as e:
                logger.debug(f"Failed to inject session recorder placeholder: {e}")

    except Exception as e:
        logger.error(f"Error during session recorder injection: {e}")


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
def start_recording_events_sync(page: SyncPage, session_id: str, client: LaminarClient):

    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    def send_events_from_browser(events):
        try:
            if events and len(events) > 0:
                client._browser_events.send(session_id, trace_id, events)
        except Exception as e:
            logger.debug(f"Could not send events: {e}")

    try:
        page.expose_function("lmnrSendEvents", send_events_from_browser)
    except Exception as e:
        logger.debug(f"Could not expose function: {e}")

    inject_session_recorder_sync(page)

    def on_load(p):
        try:
            inject_session_recorder_sync(p)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    page.on("domcontentloaded", on_load)


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
async def start_recording_events_async(
    page: Page, session_id: str, client: AsyncLaminarClient
):
    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)
    
    async def send_events_from_browser(events):
        try:
            if events and len(events) > 0:
                await client._browser_events.send(session_id, trace_id, events)
        except Exception as e:
            logger.debug(f"Could not send events: {e}")

    try:
        await page.expose_function("lmnrSendEvents", send_events_from_browser)
    except Exception as e:
        logger.debug(f"Could not expose function: {e}")

    await inject_session_recorder_async(page)
    
    async def on_load(p):
        try:
            await inject_session_recorder_async(p)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    page.on("domcontentloaded", on_load)


def take_full_snapshot(page: Page):
    return page.evaluate(
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


async def take_full_snapshot_async(page: Page):
    return await page.evaluate(
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
