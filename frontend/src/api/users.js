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