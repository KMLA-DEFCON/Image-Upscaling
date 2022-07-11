import os
from os import path


os.rename("saved_images", "img")
os.rename('images', "saved_images")
os.rename("img", "images")