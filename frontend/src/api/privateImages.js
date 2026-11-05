import { apiErrorFromResponse, getStoredToken, resolveApiUrl } from './client.js'

const PRIVATE_MEDIA_PATHS = [
    /^\/datasets\/[^/]+\/images\/[^/]+\/?$/,
    /^\/results\/[^/]+\/[^/]+\/?$/,
]

export function isPrivateImageUrl(source) {
    if (!source) return false
    try {
        const url = new URL(resolveApiUrl(source))
        const apiRoot = new URL(resolveApiUrl('/'))
        const prefix = apiRoot.pathname.replace(/\/$/, '')
        if (url.origin !== apiRoot.origin || !url.pathname.startsWith(`${prefix}/`)) {
            return false
        }
        const resourcePath = url.pathname.slice(prefix.length)
        return PRIVATE_MEDIA_PATHS.some((pattern) => pattern.test(resourcePath))
    } catch {
        return false
    }
}

export async function fetchPrivateImage(source, { signal } = {}) {
    if (!isPrivateImageUrl(source)) {
        throw new Error('La URL no corresponde a una imagen privada de la API')
    }
    const token = getStoredToken()
    if (!token) {
        throw new Error('Es necesario iniciar sesión para visualizar esta imagen')
    }
    const response = await fetch(resolveApiUrl(source), {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
        signal,
    })
    if (!response.ok) {
        throw await apiErrorFromResponse(response)
    }
    const image = await response.blob()
    if (!image.size || (image.type && !image.type.startsWith('image/'))) {
        throw new Error('La API no ha devuelto una imagen válida')
    }
    return URL.createObjectURL(image)
}
