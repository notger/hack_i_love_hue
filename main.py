import logging
from numpy.lib.npyio import load
from src.image_manipulation import Image

logging.basicConfig(
    level='DEBUG',
    format='%(levelname)s: %(message)s'
)

image = Image(file_name='images/default-2021-12-07-205120.jpeg')
