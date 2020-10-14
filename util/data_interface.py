import json


class JsonDAO:

    def save(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def load(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        return data
