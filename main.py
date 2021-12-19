import logging
from numpy.lib.npyio import load
from src.image_manipulation import Image

logging.basicConfig(
    level='DEBUG',
    format='%(levelname)s: %(message)s'
)

image = Image(file_name='images/test1.jpeg')
image.image.show()
image.get_tile(2, 1).show()
