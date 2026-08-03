import { apiRequest } from './client'

export function getUsers() {
    return apiRequest('/users')
}

export function createUser(userData) {
    return apiRequest('/users', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
    })
}

export function getUserById(userId) {
    return apiRequest(`/users/${userId}`)
}

export function updateUser(userId, userData) {
    return apiRequest(`/users/${userId}`, {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
    })
}

export function deleteUser(userId) {
    return apiRequest(`/users/${userId}`, {
        method: 'DELETE',
    })
}

export function deleteCurrentUser(password) {
    return apiRequest('/users/me', {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            password,
        }),
    })
}

export function updateResultRetention(resultRetentionSeconds) {
    return apiRequest('/users/me/retention', {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            result_retention_seconds: Number(resultRetentionSeconds),
        }),
    })
}
