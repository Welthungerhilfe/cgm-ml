import requests
import json
import numpy as np

addr = 'http://localhost:5000'
test_url = addr + "/cgm/regressor"

# Call server with some random pointclouds.
point_clouds = np.random.random((3, 30000, 4)).astype("float32")
point_clouds = point_clouds.flatten()
print(point_clouds.shape)


# Do HTTP-request.
response = requests.post(test_url, data=point_clouds.tostring())
print(json.loads(response.text))
