import { useEffect, useState } from 'react'

function ImageLabelEditor({ image, onSaveLabel }) {
    const [label, setLabel] = useState(image.label || '')
    const [saving, setSaving] = useState(false)

    useEffect(() => {
        setLabel(image.label || '')
    }, [image])

    async function handleSave() {
        try {
            setSaving(true)
            await onSaveLabel(image.id, label)
        } finally {
            setSaving(false)
        }
    }

    return (
        <div>
            <input
                type="text"
                placeholder="Etiqueta"
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                style={{ width: '120px' }}
            />

            <button type="button" onClick={handleSave} disabled={saving}>
                {saving ? 'Guardando...' : 'Guardar'}
            </button>
        </div>
    )
}

export default ImageLabelEditor