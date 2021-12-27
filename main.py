import logging
from numpy.lib.npyio import load
from src.image_manipulation import Image

# Set up logging:
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
logger = logging.getLogger('main')
logger.setLevel('INFO')
logger.addHandler(console_handler)
logger.propagate = False

logging.basicConfig(
    level='INFO',
    format='%(levelname)s: %(message)s'
)

image = Image(file_name='images/test1.jpeg')
#image.image.show()
