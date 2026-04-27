import { useEffect, useState } from 'react'
import { createUser, getUsers } from '../api/users'

function UserSelector({ selectedUser, onUserSelect }) {
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
        loadUsers()
    }, [])

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
        <div>
            <h2>Usuario</h2>

            {error && <p className="error">{error}</p>}

            <form onSubmit={handleCreateUser}>
                <input
                    type="text"
                    placeholder="Nombre"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    required
                />

                <input
                    type="email"
                    placeholder="Correo"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                />

                <input
                    type="password"
                    placeholder="Contraseña"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                />

                <button type="submit" disabled={loading}>
                    {loading ? 'Creando...' : 'Crear usuario'}
                </button>
            </form>

            <div>
                <h3>Usuarios existentes</h3>

                {users.length === 0 ? (
                    <p>No hay usuarios registrados.</p>
                ) : (
                    <select
                        value={selectedUser?.id || ''}
                        onChange={(event) => {
                            const user = users.find((item) => item.id === event.target.value)
                            onUserSelect(user)
                        }}
                    >
                        <option value="">Selecciona un usuario</option>

                        {users.map((user) => (
                            <option key={user.id} value={user.id}>
                                {user.name} ({user.email})
                            </option>
                        ))}
                    </select>
                )}

                {selectedUser && (
                    <p>
                        Usuario seleccionado: <strong>{selectedUser.name}</strong>
                    </p>
                )}
            </div>
        </div>
    )
}

export default UserSelector