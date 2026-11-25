export function filterVisibleImages(images, filterImage) {
    if (!filterImage) return images
    return images.filter((image) => filterImage(image))
}

export function resolveImageSelection(selectedImage, clickedImage, selectionLocked = false) {
    if (selectionLocked) return selectedImage
    return selectedImage?.id === clickedImage.id ? null : clickedImage
}
