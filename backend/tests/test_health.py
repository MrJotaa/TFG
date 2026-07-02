def test_root_returns_api_running(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "API running"}
