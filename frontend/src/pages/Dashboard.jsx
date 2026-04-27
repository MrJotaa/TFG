import { useEffect, useState } from 'react'
import UserSelector from '../components/UserSelector'
import DatasetUploader from '../components/DatasetUploader'
import { getDatasetsByUser } from '../api/datasets'

function Dashboard() {
    const [selectedUser, setSelectedUser] = useState(null)
    const [datasets, setDatasets] = useState([])
    const [datasetsError, setDatasetsError] = useState('')

    useEffect(() => {
        async function loadUserDatasets() {
            if (!selectedUser) {
                setDatasets([])
                return
            }

            try {
                setDatasetsError('')
                const data = await getDatasetsByUser(selectedUser.id)
                setDatasets(data)
            } catch (err) {
                setDatasetsError(err.message)
            }
        }

        loadUserDatasets()
    }, [selectedUser])

    function handleDatasetUploaded(dataset) {
        setDatasets((currentDatasets) => [...currentDatasets, dataset])
    }

    return (
        <main>
            <header>
                <h1>Adversarial AI App</h1>
                <p>
                    Plataforma para gestionar datasets, seleccionar imágenes y analizar
                    predicciones de modelos de inteligencia artificial.
                </p>
            </header>

            <section>
                <UserSelector
                    selectedUser={selectedUser}
                    onUserSelect={setSelectedUser}
                />
            </section>

            <section>
                <DatasetUploader
                    selectedUser={selectedUser}
                    onDatasetUploaded={handleDatasetUploaded}
                />
            </section>

            <section>
                <h2>Datasets</h2>

                {datasetsError && <p className="error">{datasetsError}</p>}

                {!selectedUser ? (
                    <p>Selecciona un usuario para ver sus datasets.</p>
                ) : datasets.length === 0 ? (
                    <p>Este usuario no tiene datasets.</p>
                ) : (
                    <ul>
                        {datasets.map((dataset) => (
                            <li key={dataset.id}>
                                {dataset.name} — {dataset.image_count} imágenes
                            </li>
                        ))}
                    </ul>
                )}
            </section>

            <section>
                <h2>Predicción</h2>
                <p>Zona de selección de imagen y visualización de resultados.</p>
            </section>

            <section>
                <h2>Comparativa adversaria</h2>
                <p>Zona reservada para la futura comparación con ataques adversarios.</p>
            </section>
        </main>
    )
}

export default Dashboard