import { forwardRef, useCallback, useEffect, useRef, useState } from 'react'
import { resolveApiUrl } from '../api/client'
import { fetchPrivateImage, isPrivateImageUrl } from '../api/privateImages'

/** Displays JWT-protected API images without placing the token in their URL.
 *
 * For lazy previews the observer watches the surrounding layout element,
 * rather than an <img> which initially has no src (and may have zero size).
 * Once the authenticated blob is ready, native lazy loading must be disabled:
 * that would otherwise postpone decoding a second time.
 */
const PrivateImage = forwardRef(function PrivateImage({ src, onError, loading, ...props }, forwardedRef) {
    const elementRef = useRef(null)
    const errorHandlerRef = useRef(onError)
    const [visible, setVisible] = useState(false)
    const [loaded, setLoaded] = useState(null)
    const [loadError, setLoadError] = useState(null)
    const privateImage = isPrivateImageUrl(src)
    const lazyPrivateImage = privateImage && loading === 'lazy'

    useEffect(() => {
        errorHandlerRef.current = onError
    }, [onError])

    const attachRef = useCallback((element) => {
        elementRef.current = element
        if (typeof forwardedRef === 'function') forwardedRef(element)
        else if (forwardedRef) forwardedRef.current = element
    }, [forwardedRef])

    useEffect(() => {
        if (!lazyPrivateImage || visible || typeof IntersectionObserver === 'undefined') {
            return undefined
        }

        const element = elementRef.current
        const target = element?.parentElement ?? element
        if (!target) return undefined

        // A source-less img can occupy no space. Its parent often has the
        // dimensions of a card or a figure, making it a reliable sentinel.
        const rect = target.getBoundingClientRect()
        if (rect.width === 0 || rect.height === 0) {
            // Do not wait forever on an unobservable element.
            queueMicrotask(() => setVisible(true))
            return undefined
        }

        const observer = new IntersectionObserver((entries) => {
            if (entries.some((entry) => entry.isIntersecting)) {
                setVisible(true)
                observer.disconnect()
            }
        }, { rootMargin: '400px' })
        observer.observe(target)
        return () => observer.disconnect()
    }, [lazyPrivateImage, visible, src])

    useEffect(() => {
        if (!privateImage || !src || (lazyPrivateImage && !visible && typeof IntersectionObserver !== 'undefined')) {
            return undefined
        }

        const controller = new AbortController()
        let active = true
        let objectUrl = null

        fetchPrivateImage(src, { signal: controller.signal })
            .then((url) => {
                objectUrl = url
                if (active) {
                    setLoaded({ source: src, url })
                    setLoadError(null)
                } else {
                    URL.revokeObjectURL(url)
                }
            })
            .catch((error) => {
                if (!active || error.name === 'AbortError') return
                setLoaded({ source: src, url: null })
                setLoadError({ source: src, message: error.message })
                // Preserve an actionable error instead of silently rendering
                // an empty image (e.g. 401/403, missing file or CORS failure).
                console.error('Error cargando imagen privada:', src, error)
                errorHandlerRef.current?.(error)
            })

        return () => {
            active = false
            controller.abort()
            if (objectUrl) URL.revokeObjectURL(objectUrl)
        }
    }, [privateImage, src, lazyPrivateImage, visible])

    const imageSrc = privateImage
        ? (loaded?.source === src ? loaded.url : undefined)
        : resolveApiUrl(src)
    const currentError = loadError?.source === src ? loadError.message : null

    return (
        <img
            {...props}
            ref={attachRef}
            src={imageSrc}
            loading={privateImage ? undefined : loading}
            onError={onError}
            title={currentError ? `No se pudo cargar la imagen: ${currentError}` : props.title}
            aria-label={currentError ? `${props.alt ?? 'Imagen'}: no disponible` : props['aria-label']}
        />
    )
})

export default PrivateImage
