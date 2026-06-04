import { useEffect, useState } from 'react'
import { createUser, getUsers } from '../api/users'

function UserSelector({
    selectedUser,
    onUserSelect,
    showCreateForm = true,
    showExistingUsers = true,
    onUserCreated,
}) {
    const [users, setUsers] = useState([])
    const [name, setName] = useState('')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    async function loadUsers() {
        try {
            setError('')
            const data = await getUsers()
            setUsers(data)
        } catch (err) {
            setError(err.message)
        }
    }

    useEffect(() => {
        if (showExistingUsers) {
            loadUsers()
        }
    }, [showExistingUsers])

    async function handleCreateUser(event) {
        event.preventDefault()

        try {
            setLoading(true)
            setError('')

            const newUser = await createUser({
                name,
                email,
                password,
            })

            setUsers((currentUsers) => [...currentUsers, newUser])
            onUserSelect(newUser)
            onUserCreated?.(newUser)

            setName('')
            setEmail('')
            setPassword('')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="user-selector">
            {error && <p className="error">{error}</p>}

            {showCreateForm && (
                <form onSubmit={handleCreateUser} className="user-form">
                    <label>
                        Nombre
                        <input
                            type="text"
                            placeholder="Nombre"
                            value={name}
                            onChange={(event) => setName(event.target.value)}
                            required
                        />
                    </label>

                    <label>
                        Correo
                        <input
                            type="email"
                            placeholder="Correo"
                            value={email}
                            onChange={(event) => setEmail(event.target.value)}
                            required
                        />
                    </label>

                    <label>
                        Contraseña
                        <input
                            type="password"
                            placeholder="Contraseña"
                            value={password}
                            onChange={(event) => setPassword(event.target.value)}
                            required
                        />
                    </label>

                    <button
                        type="submit"
                        className="primary-action auth-submit"
                        disabled={loading}
                    >
                        {loading ? 'Creando...' : 'Crear usuario'}
                    </button>
                </form>
            )}

            {showExistingUsers && (
                <div className="existing-users">
                    {users.length === 0 ? (
                        <p>No hay usuarios registrados.</p>
                    ) : (
                        <label>
                            Usuario
                            <select
                                value={selectedUser?.id || ''}
                                onChange={(event) => {
                                    const user = users.find(
                                        (item) => item.id === event.target.value,
                                    )
                                    onUserSelect(user || null)
                                }}
                            >
                                <option value="">Selecciona un usuario</option>

                                {users.map((user) => (
                                    <option key={user.id} value={user.id}>
                                        {user.name} ({user.email})
                                    </option>
                                ))}
                            </select>
                        </label>
                    )}

                    {selectedUser && (
                        <p className="selected-user-message">
                            Usuario seleccionado:{' '}
                            <strong>{selectedUser.name}</strong>
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}

export default UserSelector