import { apiRequest } from './client'

export function signupUser({ name, email, password }) {
    return apiRequest('/auth/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name,
            email,
            password,
        }),
    })
}

export function loginUser({ email, password }) {
    return apiRequest('/auth/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            email,
            password,
        }),
    })
}

export function getCurrentUser() {
    return apiRequest('/auth/me')
}