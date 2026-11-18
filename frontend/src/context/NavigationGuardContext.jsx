import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react'
import {
    getNavigationGuardDialogCopy,
    protectBeforeUnload,
    shouldProtectNavigation,
} from '../utils/navigationGuard'

const NavigationGuardContext = createContext(null)

const EMPTY_PROTECTION = {
    hasTemporaryWork: false,
    busy: false,
}

function NavigationGuardDialog({ request, protection, onCancel, onConfirm }) {
    const cancelButtonRef = useRef(null)
    const copy = getNavigationGuardDialogCopy({
        busy: protection.busy,
        intent: request?.intent,
    })

    useEffect(() => {
        if (!request) return undefined

        cancelButtonRef.current?.focus()

        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                event.preventDefault()
                onCancel()
            }
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [request, onCancel])

    if (!request) return null

    return (
        <div className="navigation-guard-backdrop">
            <button
                type="button"
                className="modal-backdrop-dismiss"
                tabIndex={-1}
                aria-label="Cancelar navegación"
                onClick={onCancel}
            />
            <section
                className={`navigation-guard-dialog ${copy.tone}`}
                role="alertdialog"
                aria-modal="true"
                aria-labelledby="navigation-guard-title"
                aria-describedby="navigation-guard-message"
            >
                <div className="navigation-guard-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24">
                        <path d="M12 3.5 2.8 19.2a1.2 1.2 0 0 0 1 1.8h16.4a1.2 1.2 0 0 0 1-1.8L12 3.5Z" />
                        <path d="M12 9v5M12 17.5v.1" />
                    </svg>
                </div>

                <div className="navigation-guard-copy">
                    <span className="evaluation-eyebrow">Trabajo temporal</span>
                    <h2 id="navigation-guard-title">{copy.title}</h2>
                    <p id="navigation-guard-message">{copy.message}</p>
                </div>

                <div className="navigation-guard-actions">
                    <button
                        ref={cancelButtonRef}
                        type="button"
                        className="secondary-action"
                        onClick={onCancel}
                    >
                        {copy.cancelLabel}
                    </button>
                    <button
                        type="button"
                        className={copy.tone === 'danger' ? 'danger-action' : 'primary-action'}
                        onClick={onConfirm}
                    >
                        {copy.confirmLabel}
                    </button>
                </div>
            </section>
        </div>
    )
}

export function NavigationGuardProvider({ children }) {
    const [protection, setProtection] = useState(EMPTY_PROTECTION)
    const [pendingRequest, setPendingRequest] = useState(null)

    const setNavigationProtection = useCallback((nextProtection = {}) => {
        const normalizedProtection = {
            hasTemporaryWork: Boolean(nextProtection.hasTemporaryWork),
            busy: Boolean(nextProtection.busy),
        }

        setProtection(normalizedProtection)

        if (!shouldProtectNavigation(normalizedProtection)) {
            setPendingRequest(null)
        }
    }, [])

    const clearNavigationProtection = useCallback(() => {
        setProtection(EMPTY_PROTECTION)
        setPendingRequest(null)
    }, [])

    const requestNavigation = useCallback((action, options = {}) => {
        if (typeof action !== 'function') return false

        if (!shouldProtectNavigation(protection)) {
            action()
            return true
        }

        setPendingRequest({
            action,
            intent: options.intent ?? 'navigate',
        })
        return false
    }, [protection])

    const cancelPendingNavigation = useCallback(() => {
        setPendingRequest(null)
    }, [])

    const confirmPendingNavigation = useCallback(() => {
        const action = pendingRequest?.action
        setPendingRequest(null)
        setProtection(EMPTY_PROTECTION)
        action?.()
    }, [pendingRequest])


    useEffect(() => {
        if (!shouldProtectNavigation(protection)) return undefined

        const handleBeforeUnload = (event) => protectBeforeUnload(event)
        window.addEventListener('beforeunload', handleBeforeUnload)
        return () => window.removeEventListener('beforeunload', handleBeforeUnload)
    }, [protection])

    const value = useMemo(
        () => ({
            protection,
            setNavigationProtection,
            clearNavigationProtection,
            requestNavigation,
        }),
        [
            protection,
            setNavigationProtection,
            clearNavigationProtection,
            requestNavigation,
        ],
    )

    return (
        <NavigationGuardContext.Provider value={value}>
            {children}
            <NavigationGuardDialog
                request={pendingRequest}
                protection={protection}
                onCancel={cancelPendingNavigation}
                onConfirm={confirmPendingNavigation}
            />
        </NavigationGuardContext.Provider>
    )
}

// Context hooks intentionally live beside their provider.
// eslint-disable-next-line react-refresh/only-export-components
export function useNavigationGuard() {
    const context = useContext(NavigationGuardContext)

    if (!context) {
        throw new Error('useNavigationGuard must be used inside NavigationGuardProvider')
    }

    return context
}
