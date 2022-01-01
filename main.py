import logging
from src.image_manipulation import Image
from src.solution_base import Solution
from src.solver_naive import naive_method

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

# ------------------------------- Choose filename here! --------------------------
# File has to be in images directory and please only type the pure file-name here,
# without the images-folder.
file_name = 'test1.jpeg'

# Do stuff.
image = Image(file_name=f'images/{file_name}')
solution = Solution(image)
solution.solve(naive_method)
solution.generate_gif()

logger.info(f'Successfully solved the image and created the output gif in the solutions-folder.')
