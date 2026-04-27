import { useState } from 'react'
import UserSelector from '../components/UserSelector'

function Dashboard() {
    const [selectedUser, setSelectedUser] = useState(null)

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
                <h2>Datasets</h2>
                <p>Zona de subida y selección de datasets.</p>
            </section>

            <section>
                <h2>Predicción</h2>
                <p>Zona de selección de imagen y visualización de resultados.</p>
            </section>

            <section>
                <h2>Comparativa adversaria</h2>
                <p>Zona reservada para la futura comparación con ataques adversarios.</p>
            </section>
            <section>
                <h2>Debug</h2>
                <p>Aquí irán logs temporales de desarrollo.</p>
            </section>
        </main>
    )
}

export default Dashboard