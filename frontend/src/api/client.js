const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function apiRequest(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, options)

    if (!response.ok) {
        const error = await response.json().catch(() => ({
            detail: 'Unexpected API error',
        }))

        throw new Error(error.detail || 'Unexpected API error')
    }

    return response.json()
}