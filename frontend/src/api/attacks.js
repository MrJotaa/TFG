import { apiRequest } from './client'

export function getAttacks() {
    return apiRequest('/attacks')
}
