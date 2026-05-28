import { useEffect, useState } from 'react'
import UserSelector from '../components/UserSelector'
import DatasetUploader from '../components/DatasetUploader'
import DatasetList from '../components/DatasetList'
import ImageGallery from '../components/ImageGallery'
import { getDatasetsByUser } from '../api/datasets'
import PredictionPanel from '../components/PredictionPanel'
import ModelSelector from '../components/ModelSelector'
import DatasetImageUploader from '../components/DatasetImageUploader'
import AttackPanel from '../components/AttackPanel'

function Dashboard() {
    const [selectedUser, setSelectedUser] = useState(null)
    const [datasets, setDatasets] = useState([])
    const [selectedDataset, setSelectedDataset] = useState(null)
    const [selectedImage, setSelectedImage] = useState(null)
    const [datasetsError, setDatasetsError] = useState('')
    const [imagesRefreshKey, setImagesRefreshKey] = useState(0)

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

    function handleImagesUploaded(uploadedImages) {
        setImagesRefreshKey((current) => current + 1)

        setDatasets((currentDatasets) =>
            currentDatasets.map((dataset) =>
                dataset.id === selectedDataset.id
                    ? {
                        ...dataset,
                        image_count: dataset.image_count + uploadedImages.length,
                    }
                    : dataset,
            ),
        )

        setSelectedDataset((currentDataset) =>
            currentDataset
                ? {
                    ...currentDataset,
                    image_count: currentDataset.image_count + uploadedImages.length,
                }
                : currentDataset,
        )
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
                <DatasetList
                    datasets={datasets}
                    selectedDataset={selectedDataset}
                    onSelectDataset={(dataset) => {
                        setSelectedDataset(dataset)
                        setSelectedImage(null)
                    }}
                />
            </section>

            <section>
                <DatasetImageUploader
                    selectedDataset={selectedDataset}
                    onImagesUploaded={handleImagesUploaded}
                />
            </section>

            <ImageGallery
                selectedDataset={selectedDataset}
                selectedImage={selectedImage}
                onSelectImage={setSelectedImage}
                refreshKey={imagesRefreshKey}
            />

            <section>
                <ModelSelector />
            </section>

            <section>
                <PredictionPanel
                    selectedDataset={selectedDataset}
                    selectedImage={selectedImage}
                />
            </section>

            <section>
                <AttackPanel
                    selectedDataset={selectedDataset}
                    selectedImage={selectedImage}
                />
            </section>
        </main>
    )
}

export default Dashboard