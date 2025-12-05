/**
 * The session recording inject script function.
 * This function runs in the browser context and sets up rrweb recording.
 * 
 * @param {Object} maskInputOptions - Optional recording options for masking inputs
 * @param {boolean} stringifyCallbackArgs - If true, stringify arguments when calling
 *                                          lmnrSendEvents (for raw CDP bindings)
 */
(maskInputOptions, stringifyCallbackArgs) => {
    const BATCH_TIMEOUT = 2000; // Send events after 2 seconds
    const MAX_WORKER_PROMISES = 50; // Max concurrent worker promises
    const HEARTBEAT_INTERVAL = 2000;
    const CHUNK_SIZE = 256 * 1024; // 256KB chunks
    const CHUNK_SEND_DELAY = 100; // 100ms delay between chunks

    window.lmnrRrwebEventsBatch = [];
    window.lmnrChunkQueue = [];
    window.lmnrChunkSequence = 0;
    window.lmnrCurrentBatchId = null;

    // Define a wrapper function that handles stringification based on the parameter
    const sendEvent = stringifyCallbackArgs 
        ? (chunk) => window.lmnrSendEvents(JSON.stringify(chunk))
        : (chunk) => window.lmnrSendEvents(chunk);

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
    let workerCreationInitiated = false;

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
        workerCreationInitiated = false;
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
                if (!compressionWorker && !workerCreationInitiated) {
                    workerCreationInitiated = true;
                    try {
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
                    } catch (error) {
                        workerCreationInitiated = false;
                        throw error;
                    }
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
                    if (!compressionWorker && !workerCreationInitiated) {
                        workerCreationInitiated = true;
                        try {
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
                        } catch (error) {
                            workerCreationInitiated = false;
                            throw error;
                        }
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
                await sendEvent(chunk);
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
                    await sendEvent(chunk);
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
