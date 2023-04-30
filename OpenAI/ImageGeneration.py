import os
import openai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
openai.organization = os.getenv('OPENAI_ORGANIZATION')
openai.api_key = os.getenv('OPENAI_APIKEY')
openai.Model.list()

response = openai.Image.create(
  prompt="sky and mountain",
  n=1,
  size="512x512"
)
image_url = response['data'][0]['url']
# print(image_url)
im = np.array(Image.open(io.BytesIO(requests.get(image_url).content)))
plt.imshow(im)
plt.show()