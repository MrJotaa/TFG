import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { deleteUser, updateUser } from '../api/users'
import { getDatasetsByUser } from '../api/datasets'
import { useAuth } from '../context/AuthContext'
import { useExperiment } from '../context/ExperimentContext'

function ProfilePage() {
    const navigate = useNavigate()
    const { user, setUser, logout } = useAuth()
    const { resetExperiment } = useExperiment()

    const [name, setName] = useState(user?.name || '')
    const [email, setEmail] = useState(user?.email || '')
    const [password, setPassword] = useState('')
    const [datasets, setDatasets] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState('')

    useEffect(() => {
        async function loadDatasets() {
            if (!user) return

            try {
                setError('')
                const data = await getDatasetsByUser(user.id)
                setDatasets(data)
            } catch (err) {
                setError(err.message)
            }
        }

        loadDatasets()
    }, [user])

    async function handleUpdateProfile(event) {
        event.preventDefault()

        if (!user) return

        const updateData = {
            name,
            email,
        }

        if (password.trim()) {
            updateData.password = password
        }

        try {
            setLoading(true)
            setError('')
            setSuccess('')

            const updatedUser = await updateUser(user.id, updateData)

            setUser(updatedUser)
            setPassword('')
            setSuccess('Perfil actualizado correctamente.')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    function handleLogout() {
        logout()
        resetExperiment()
        navigate('/')
    }

    async function handleDeleteAccount() {
        if (!user) return

        const confirmed = window.confirm(
            '¿Seguro que quieres eliminar tu cuenta? También se eliminarán todos tus datasets e imágenes. Esta acción no se puede deshacer.',
        )

        if (!confirmed) return

        try {
            setLoading(true)
            setError('')

            await deleteUser(user.id)

            logout()
            resetExperiment()
            navigate('/')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    if (!user) {
        return (
            <section className="profile-page">
                <div className="empty-state">
                    <h2>No hay usuario autenticado</h2>
                    <button
                        type="button"
                        className="primary-action"
                        onClick={() => navigate('/')}
                    >
                        Ir al login
                    </button>
                </div>
            </section>
        )
    }

    return (
        <section className="profile-page">
            <div className="page-toolbar">
                <div>
                    <h2>Configuración de Usuario</h2>
                    <p>Gestiona tus datos, datasets y sesión.</p>
                </div>

                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard')}
                >
                    Volver al Dashboard
                </button>
            </div>

            {error && <p className="error">{error}</p>}
            {success && <p className="success-message">{success}</p>}

            <div className="profile-layout">
                <section className="management-card">
                    <h3>Datos del usuario</h3>

                    <form onSubmit={handleUpdateProfile} className="profile-form">
                        <label>
                            Nombre
                            <input
                                type="text"
                                value={name}
                                onChange={(event) => setName(event.target.value)}
                                required
                            />
                        </label>

                        <label>
                            Correo
                            <input
                                type="email"
                                value={email}
                                onChange={(event) => setEmail(event.target.value)}
                                required
                            />
                        </label>

                        <label>
                            Nueva contraseña
                            <input
                                type="password"
                                value={password}
                                onChange={(event) => setPassword(event.target.value)}
                                placeholder="Dejar vacío para no cambiar"
                            />
                        </label>

                        <button
                            type="submit"
                            className="primary-action"
                            disabled={loading}
                        >
                            {loading ? 'Guardando...' : 'Guardar cambios'}
                        </button>
                    </form>
                </section>

                <section className="management-card">
                    <h3>Datasets del usuario</h3>

                    <p>
                        Total de datasets:{' '}
                        <strong>{datasets.length}</strong>
                    </p>

                    {datasets.length === 0 ? (
                        <p>No tienes datasets creados.</p>
                    ) : (
                        <div className="profile-dataset-list">
                            {datasets.slice(0, 5).map((dataset) => (
                                <div
                                    key={dataset.id}
                                    className="profile-dataset-item"
                                >
                                    <strong>{dataset.name}</strong>
                                    <span>{dataset.image_count} imágenes</span>
                                </div>
                            ))}
                        </div>
                    )}

                    <button
                        type="button"
                        className="secondary-action"
                        onClick={() => navigate('/datasets')}
                    >
                        Gestionar datasets
                    </button>
                </section>

                <section className="management-card">
                    <h3>Sesión</h3>

                    <button
                        type="button"
                        className="secondary-action"
                        onClick={handleLogout}
                    >
                        Cerrar sesión
                    </button>
                </section>

                <section className="management-card danger-zone">
                    <h3>Zona peligrosa</h3>

                    <p>
                        Eliminar tu cuenta borrará también todos tus datasets e imágenes.
                    </p>

                    <button
                        type="button"
                        className="danger-action"
                        onClick={handleDeleteAccount}
                        disabled={loading}
                    >
                        Eliminar cuenta
                    </button>
                </section>
            </div>
        </section>
    )
}

export default ProfilePage