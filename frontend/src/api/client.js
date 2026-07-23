const API_BASE_URL =
    import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const AUTH_TOKEN_KEY = 'auth_token'

export function getStoredToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function setStoredToken(token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token)
}

export function removeStoredToken() {
    localStorage.removeItem(AUTH_TOKEN_KEY)
}

export function resolveApiUrl(path) {
    if (!path) {
        return null
    }

    if (/^https?:\/\//i.test(path)) {
        return path
    }

    return `${API_BASE_URL}${path}`
}

export async function apiRequest(path, options = {}) {
    const token = getStoredToken()

    const isFormData = options.body instanceof FormData

    const headers = {
        ...(options.headers || {}),
    }

    if (token) {
        headers.Authorization = `Bearer ${token}`
    }

    if (isFormData) {
        delete headers['Content-Type']
    }

    const response = await fetch(resolveApiUrl(path), {
        ...options,
        headers,
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({
            detail: 'Unexpected API error',
        }))

        const requestError = new Error(error.detail || 'Unexpected API error')
        requestError.status = response.status
        throw requestError
    }

    if (response.status === 204) {
        return null
    }

    return response.json()
}
