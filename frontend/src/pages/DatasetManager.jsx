import DatasetUploader from '../components/DatasetUploader'
import DatasetImageUploader from '../components/DatasetImageUploader'
import DatasetList from '../components/DatasetList'
import ImageGallery from '../components/ImageGallery'
import { useExperiment } from '../context/ExperimentContext'

function DatasetManager() {
    const {
        selectedUser,
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset,
        selectedImage,
        setSelectedImage,
    } = useExperiment()

    function handleDatasetUploaded(dataset) {
        setDatasets((currentDatasets) => [...currentDatasets, dataset])
    }

    function handleImagesUploaded(uploadedImages) {
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
        <section className="view-panel">
            <h2>Gestión de datasets</h2>
            <p>
                Sube datasets, añade imágenes a datasets existentes y edita etiquetas.
            </p>

            <DatasetUploader
                selectedUser={selectedUser}
                onDatasetUploaded={handleDatasetUploaded}
            />

            <DatasetList
                datasets={datasets}
                selectedDataset={selectedDataset}
                onSelectDataset={(dataset) => {
                    setSelectedDataset(dataset)
                    setSelectedImage(null)
                }}
            />

            <DatasetImageUploader
                selectedDataset={selectedDataset}
                onImagesUploaded={handleImagesUploaded}
            />

            <ImageGallery
                selectedDataset={selectedDataset}
                selectedImage={selectedImage}
                onSelectImage={setSelectedImage}
                showLabelEditor
            />
        </section>
    )
}

export default DatasetManager