import pytest


from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestApi:

    def test_home(self):

        response = client.get("/")
        assert response.status_code == 200

    def test_fake_predict(self):

        response = client.get("/fake_predict/?descr='hello ca va'")
        assert response.status_code == 200
