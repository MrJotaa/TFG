import { createContext, useContext, useEffect, useState } from 'react'
import { getCurrentUser, loginUser, logoutUser, signupUser } from '../api/auth'
import { removeStoredToken, setStoredToken } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loadingAuth, setLoadingAuth] = useState(true)
    const [authError, setAuthError] = useState('')

    useEffect(() => {
        async function loadCurrentUser() {
            try {
                setAuthError('')
                const result = await getCurrentUser()
                setUser(result.user)
            } catch {
                removeStoredToken()
                setUser(null)
            } finally {
                setLoadingAuth(false)
            }
        }

        loadCurrentUser()
    }, [])

    async function login({ email, password }) {
        setAuthError('')

        const result = await loginUser({ email, password })

        setStoredToken(result.access_token)
        setUser(result.user)

        return result.user
    }

    async function signup({ name, email, password }) {
        setAuthError('')

        const result = await signupUser({ name, email, password })

        setStoredToken(result.access_token)
        setUser(result.user)

        return result.user
    }

    function logout() {
        void logoutUser().catch(() => {})
        removeStoredToken()
        setUser(null)
    }

    const value = {
        user,
        setUser,
        loadingAuth,
        authError,
        setAuthError,
        login,
        signup,
        logout,
        isAuthenticated: Boolean(user),
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}

// Context hooks intentionally live beside their provider.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
    const context = useContext(AuthContext)

    if (!context) {
        throw new Error('useAuth must be used inside AuthProvider')
    }

    return context
}