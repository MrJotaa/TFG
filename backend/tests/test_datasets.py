def test_create_dataset_with_image_returns_dataset(
    client,
    registered_user,
    image_file_factory,
):
    user_id = registered_user["user"]["id"]

    response = client.post(
        "/datasets",
        data={
            "user_id": user_id,
            "name": "My Dataset",
        },
        files=[
            image_file_factory("cat.png"),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"]
    assert data["name"] == "My Dataset"
    assert data["owner_id"] == user_id
    assert data["image_count"] == 1
    assert len(data["images"]) == 1

    image = data["images"][0]

    assert image["id"]
    assert image["filename"] == "cat.png"
    assert image["url"].startswith(f"/datasets/{data['id']}/images/")
    assert image["label"] is None


def test_create_dataset_with_invalid_user_returns_400(
    client,
    image_file_factory,
):
    response = client.post(
        "/datasets",
        data={
            "user_id": "invalid-user-id",
            "name": "Invalid Dataset",
        },
        files=[
            image_file_factory("image.png"),
        ],
    )

    assert response.status_code == 400
    assert "invalid user id" in response.json()["detail"].lower()


def test_create_dataset_with_non_existing_user_returns_400(
    client,
    image_file_factory,
):
    response = client.post(
        "/datasets",
        data={
            "user_id": "507f1f77bcf86cd799439011",
            "name": "Invalid Dataset",
        },
        files=[
            image_file_factory("image.png"),
        ],
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_create_dataset_with_invalid_file_extension_returns_400(
    client,
    registered_user,
):
    user_id = registered_user["user"]["id"]

    response = client.post(
        "/datasets",
        data={
            "user_id": user_id,
            "name": "Invalid File Dataset",
        },
        files=[
            (
                "files",
                ("notes.txt", b"not an image", "text/plain"),
            ),
        ],
    )

    assert response.status_code == 400
    assert "invalid image file" in response.json()["detail"].lower()


def test_list_datasets_returns_created_dataset(client, created_dataset):
    response = client.get("/datasets")

    assert response.status_code == 200

    datasets = response.json()

    assert len(datasets) == 1
    assert datasets[0]["id"] == created_dataset["id"]
    assert datasets[0]["name"] == created_dataset["name"]


def test_get_dataset_by_id_returns_dataset(client, created_dataset):
    response = client.get(f"/datasets/{created_dataset['id']}")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == created_dataset["id"]
    assert data["name"] == created_dataset["name"]
    assert data["image_count"] == 1


def test_get_dataset_with_invalid_id_returns_404(client):
    response = client.get("/datasets/invalid-id")

    assert response.status_code == 404
    assert "invalid dataset id" in response.json()["detail"].lower()


def test_get_dataset_with_unknown_id_returns_404(client):
    response = client.get("/datasets/507f1f77bcf86cd799439011")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_list_dataset_images_returns_images(client, created_dataset):
    response = client.get(f"/datasets/{created_dataset['id']}/images")

    assert response.status_code == 200

    images = response.json()

    assert len(images) == 1
    assert images[0]["id"] == created_dataset["images"][0]["id"]
    assert images[0]["filename"] == "image-1.png"


def test_add_images_to_dataset_returns_new_images(
    client,
    created_dataset,
    image_file_factory,
):
    response = client.post(
        f"/datasets/{created_dataset['id']}/images",
        files=[
            image_file_factory("image-2.png", color=(0, 255, 0)),
            image_file_factory("image-3.png", color=(0, 0, 255)),
        ],
    )

    assert response.status_code == 200

    added_images = response.json()

    assert len(added_images) == 2
    assert added_images[0]["filename"] == "image-2.png"
    assert added_images[1]["filename"] == "image-3.png"

    dataset_response = client.get(f"/datasets/{created_dataset['id']}")
    assert dataset_response.status_code == 200
    assert dataset_response.json()["image_count"] == 3


def test_add_image_to_unknown_dataset_returns_400(
    client,
    image_file_factory,
):
    response = client.post(
        "/datasets/507f1f77bcf86cd799439011/images",
        files=[
            image_file_factory("image.png"),
        ],
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_get_dataset_image_returns_file(client, created_dataset):
    dataset_id = created_dataset["id"]
    image_id = created_dataset["images"][0]["id"]

    response = client.get(f"/datasets/{dataset_id}/images/{image_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] in {
        "image/png",
        "application/octet-stream",
    }
    assert response.content


def test_get_unknown_dataset_image_returns_404(client, created_dataset):
    dataset_id = created_dataset["id"]

    response = client.get(f"/datasets/{dataset_id}/images/unknown-image")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_update_image_label_returns_updated_image(client, created_dataset):
    dataset_id = created_dataset["id"]
    image_id = created_dataset["images"][0]["id"]

    response = client.patch(
        f"/datasets/{dataset_id}/images/{image_id}/label",
        json={
            "label": "tabby cat",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == image_id
    assert data["label"] == "tabby cat"


def test_update_image_label_with_blank_value_sets_none(client, created_dataset):
    dataset_id = created_dataset["id"]
    image_id = created_dataset["images"][0]["id"]

    response = client.patch(
        f"/datasets/{dataset_id}/images/{image_id}/label",
        json={
            "label": "   ",
        },
    )

    assert response.status_code == 200
    assert response.json()["label"] is None


def test_delete_dataset_image_removes_image(client, created_dataset):
    dataset_id = created_dataset["id"]
    image_id = created_dataset["images"][0]["id"]

    response = client.delete(f"/datasets/{dataset_id}/images/{image_id}")

    assert response.status_code == 200
    assert response.json() == {"message": "Image deleted successfully"}

    images_response = client.get(f"/datasets/{dataset_id}/images")

    assert images_response.status_code == 200
    assert images_response.json() == []

    dataset_response = client.get(f"/datasets/{dataset_id}")

    assert dataset_response.status_code == 200
    assert dataset_response.json()["image_count"] == 0


def test_delete_unknown_dataset_image_returns_404(client, created_dataset):
    dataset_id = created_dataset["id"]

    response = client.delete(f"/datasets/{dataset_id}/images/unknown-image")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_dataset_removes_dataset(client, created_dataset):
    dataset_id = created_dataset["id"]

    response = client.delete(f"/datasets/{dataset_id}")

    assert response.status_code == 200
    assert response.json() == {"message": "Dataset deleted successfully"}

    get_response = client.get(f"/datasets/{dataset_id}")

    assert get_response.status_code == 404


def test_delete_unknown_dataset_returns_404(client):
    response = client.delete("/datasets/507f1f77bcf86cd799439011")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_user_datasets_returns_only_user_datasets(
    client,
    registered_user,
    image_file_factory,
):
    user_id = registered_user["user"]["id"]

    first_response = client.post(
        "/datasets",
        data={
            "user_id": user_id,
            "name": "Dataset A",
        },
        files=[
            image_file_factory("a.png"),
        ],
    )
    assert first_response.status_code == 200

    second_response = client.post(
        "/datasets",
        data={
            "user_id": user_id,
            "name": "Dataset B",
        },
        files=[
            image_file_factory("b.png"),
        ],
    )
    assert second_response.status_code == 200

    response = client.get(f"/users/{user_id}/datasets")

    assert response.status_code == 200

    datasets = response.json()
    dataset_names = {dataset["name"] for dataset in datasets}

    assert dataset_names == {"Dataset A", "Dataset B"}