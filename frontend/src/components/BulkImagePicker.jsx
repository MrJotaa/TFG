import { useRef, useState } from 'react'
import { getSelectedFilesSummary } from '../utils/datasetManagement'

function UploadIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 16V5m0 0L8 9m4-4 4 4M5 15.5V19h14v-3.5" />
        </svg>
    )
}

function BulkImagePicker({ files, onChange, disabled = false }) {
    const inputRef = useRef(null)
    const [dragging, setDragging] = useState(false)
    const summary = getSelectedFilesSummary(files)

    function updateFiles(nextFiles) {
        if (disabled) return
        onChange(Array.from(nextFiles || []))
    }

    const selectedImageNoun = summary.count === 1 ? 'imagen seleccionada' : 'imágenes seleccionadas'
    const dropzoneTitle = summary.count > 0
        ? `${summary.count} ${selectedImageNoun}`
        : 'Arrastra tus imágenes aquí'
    const dropzoneDescription = summary.count > 0
        ? `${summary.totalMegabytes.toFixed(1)} MB en total · pulsa para cambiar la selección`
        : 'o pulsa para seleccionar varios archivos de una vez'

    return (
        <div className="bulk-image-picker">
            <input
                ref={inputRef}
                className="bulk-image-native-input"
                type="file"
                accept="image/*"
                multiple
                disabled={disabled}
                onChange={(event) => updateFiles(event.target.files)}
            />

            <button
                type="button"
                className={`bulk-image-dropzone${dragging ? ' dragging' : ''}`}
                disabled={disabled}
                onClick={() => inputRef.current?.click()}
                onDragEnter={(event) => {
                    event.preventDefault()
                    if (!disabled) setDragging(true)
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                    event.preventDefault()
                    setDragging(false)
                    updateFiles(event.dataTransfer.files)
                }}
            >
                <span className="bulk-image-dropzone-icon"><UploadIcon /></span>
                <strong>{dropzoneTitle}</strong>
                <span>{dropzoneDescription}</span>
            </button>

            {summary.count > 0 && (
                <div className="bulk-image-file-summary" aria-live="polite">
                    {summary.previewNames.map((name) => (
                        <span key={name}>{name}</span>
                    ))}
                    {summary.remainingCount > 0 && (
                        <strong>+{summary.remainingCount} más</strong>
                    )}
                </div>
            )}
        </div>
    )
}

export default BulkImagePicker
