/**
 * The session recording inject script function.
 * This function runs in the browser context (isolated world) and sets up rrweb recording.
 *
 * Design principles:
 * - Synchronous emit: events are instantly pushed to a batch array, never lost
 * - Compress at send time: the whole batch is gzipped once via CompressionStream
 * - No Web Workers: compression runs on the main thread of the isolated world
 *   (doesn't block the main page since we're in a separate execution context)
 * - Chunking for CDP message size limits
 *
 * @param {Object} maskInputOptions - Optional recording options for masking inputs
 * @param {boolean} stringifyCallbackArgs - If true, stringify arguments when calling
 *                                          lmnrSendEvents (for raw CDP bindings)
 */
(maskInputOptions, stringifyCallbackArgs) => {
    const BATCH_TIMEOUT = 2000; // Send events every 2 seconds
    const HEARTBEAT_INTERVAL = 1000;
    const CHUNK_SIZE = 256 * 1024; // 256KB chunks for CDP message limits

    window.lmnrRrwebEventsBatch = [];
    window.lmnrChunkSequence = 0;
    window.lmnrSendInProgress = false;

    // Define a wrapper function that handles stringification based on the parameter
    const sendEvent = stringifyCallbackArgs
        ? (chunk) => window.lmnrSendEvents(JSON.stringify(chunk))
        : (chunk) => window.lmnrSendEvents(chunk);

    // Gzip compress a string using CompressionStream API (main thread, no workers)
    async function gzipCompress(str) {
        const encoder = new TextEncoder();
        const inputBytes = encoder.encode(str);

        const cs = new CompressionStream('gzip');
        const writer = cs.writable.getWriter();
        const reader = cs.readable.getReader();

        writer.write(inputBytes);
        writer.close();

        const parts = [];
        let totalLength = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            parts.push(value);
            totalLength += value.length;
        }

        const result = new Uint8Array(totalLength);
        let offset = 0;
        for (const part of parts) {
            result.set(part, offset);
            offset += part.length;
        }
        return result;
    }

    // Convert a Uint8Array to base64 string
    async function bufferToBase64(buffer) {
        const base64url = await new Promise(r => {
            const reader = new FileReader();
            reader.onload = () => r(reader.result);
            reader.readAsDataURL(new Blob([buffer]));
        });
        return base64url.slice(base64url.indexOf(',') + 1);
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

    async function sendBatchIfReady() {
        if (window.lmnrRrwebEventsBatch.length === 0) {
            return;
        }
        if (typeof window.lmnrSendEvents !== 'function') {
            return;
        }
        // Prevent overlapping sends - if previous send is still in progress,
        // events stay in the batch and will be picked up next interval
        if (window.lmnrSendInProgress) {
            return;
        }

        window.lmnrSendInProgress = true;
        const events = window.lmnrRrwebEventsBatch;
        window.lmnrRrwebEventsBatch = [];

        try {
            // Compress each event's data field individually
            const compressedEvents = [];
            for (const event of events) {
                try {
                    const dataString = JSON.stringify(event.data);
                    const compressed = await gzipCompress(dataString);
                    const base64Data = await bufferToBase64(compressed);
                    compressedEvents.push({
                        ...event,
                        data: base64Data,
                    });
                } catch (e) {
                    console.error('Failed to compress event:', e);
                }
            }

            const batchId = `${Date.now()}_${window.lmnrChunkSequence++}`;
            const batchString = JSON.stringify(compressedEvents);

            if (batchString.length <= CHUNK_SIZE) {
                sendEvent({
                    batchId: batchId,
                    chunkIndex: 0,
                    totalChunks: 1,
                    data: batchString,
                    isFinal: true
                });
            } else {
                const chunks = createChunks(batchString, batchId);
                for (const chunk of chunks) {
                    sendEvent(chunk);
                }
            }
        } catch (error) {
            console.error('Failed to send events:', error);
        } finally {
            window.lmnrSendInProgress = false;
        }
    }

    if (!window.lmnrStartedRecordingEvents) {
        setInterval(sendBatchIfReady, BATCH_TIMEOUT);

        window.lmnrRrweb.record({
            emit(event) {
                // Synchronous emit - just push to batch, no async processing.
                // Compression happens later in sendBatchIfReady.
                window.lmnrRrwebEventsBatch.push(event);
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

        // Heartbeat events to indicate the session is still alive
        setInterval(
            () => {
                window.lmnrRrweb.record.addCustomEvent('heartbeat', {
                    title: document.title,
                    url: document.URL,
                });
            },
            HEARTBEAT_INTERVAL,
        );

        window.lmnrStartedRecordingEvents = true;
    }
}
