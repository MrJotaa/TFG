import {
    apiErrorFromResponse,
    getStoredToken,
    resolveApiUrl,
} from './client.js'

function streamEventError(event) {
    const error = new Error(event.message || 'Unexpected evaluation error')
    error.status = event.status || 500
    error.details =
        event.detail && typeof event.detail === 'object' ? event.detail : null
    return error
}

export async function streamEvaluationRequest(path, payload, { onProgress } = {}) {
    const token = getStoredToken()
    const headers = {
        'Content-Type': 'application/json',
    }
    if (token) {
        headers.Authorization = `Bearer ${token}`
    }

    const response = await fetch(resolveApiUrl(path), {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
    })

    if (!response.ok) {
        throw await apiErrorFromResponse(response)
    }
    if (!response.body) {
        throw new Error('Evaluation progress stream is not available')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalResult = null

    function processLine(line) {
        if (!line.trim()) {
            return
        }

        const event = JSON.parse(line)
        if (event.type === 'progress') {
            onProgress?.({
                processed: event.processed,
                total: event.total,
                phase: event.phase,
            })
            return
        }
        if (event.type === 'error') {
            throw streamEventError(event)
        }
        if (event.type === 'result') {
            finalResult = event.data
        }
    }

    try {
        while (true) {
            const { value, done } = await reader.read()
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

            let newlineIndex = buffer.indexOf('\n')
            while (newlineIndex !== -1) {
                const line = buffer.slice(0, newlineIndex)
                buffer = buffer.slice(newlineIndex + 1)
                processLine(line)
                newlineIndex = buffer.indexOf('\n')
            }

            if (done) {
                break
            }
        }

        if (buffer.trim()) {
            processLine(buffer)
        }
    } finally {
        reader.releaseLock()
    }

    if (finalResult === null) {
        throw new Error('Evaluation stream ended without a result')
    }
    return finalResult
}
