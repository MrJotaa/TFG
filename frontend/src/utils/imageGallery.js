export function filterVisibleImages(images, filterImage) {
    if (!filterImage) return images
    return images.filter((image) => filterImage(image))
}
