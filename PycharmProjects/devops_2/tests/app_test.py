import json

from tornado.testing import AsyncHTTPTestCase

from src.app import make_app


class CerTesterAdd(AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def test_url(self):
        # This is where your actual test will take place.
        response = self.fetch('/add/?first=10&second=5', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {'result': 15})
        response = self.fetch('/add/?first=3&second=-9', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(data, {'result': -6})


class CerTesterSub(AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def test_url(self):
        # This is where your actual test will take place.
        response = self.fetch('/sub/?first=65&second=5', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {'result': 60})
        response = self.fetch('/sub/?first=10&second=-9', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {'result': 19})


class CerTesterMult(AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def test_url(self):
        # This is where your actual test will take place.
        response = self.fetch('/mult/?first=5&second=7', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {'result': 35})
        response = self.fetch('/mult/?first=3&second=10', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {'result': 30})


class CerTesterDiv(AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def test_url(self):
        # This is where your actual test will take place.
        response = self.fetch('/div/?first=30&second=6', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {'result': 5})
        response = self.fetch('/div/?first=7&second=0', method="GET")
        data = json.loads(response.body.decode('utf-8'))
        self.assertEqual(response.code, 200)
        self.assertEqual(data, {"error": "Division by 0"})
