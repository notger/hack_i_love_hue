"""
Module to visualise a state.
"""
import logging
import numpy as np
from typing import List, Tuple
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw

# Set up logging:
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
logger = logging.getLogger('state_visualisation')
logger.setLevel('INFO')
logger.addHandler(console_handler)
logger.propagate = False


def generate_solution_gif(solution: 'Solution') -> None:
    file_name = solution.image.file_name.split('/')[-1]
    images = [_generate_state_image(step) for step in solution.steps]
    images[0].save(
        f'solutions/{file_name}.gif', 
        save_all=True,
        optimize=False, 
        append_images=images[1:], 
        loop=0,
        duration=len(images),
    )

def _generate_state_image(
    state: 'State', 
    cell_size: int = 60, 
    add_swapped_element_line: bool = True,
    swapped_element_line_colour: int = 30,
    fixed_tiles_information: List[Tuple] = None,
) -> PILImage:
    N_i, N_j, _ = state.colouring.shape
    m = np.zeros((N_i * cell_size, N_j * cell_size, 3), dtype=int)

    # Fill the colours in cell-wise:
    for i in range(N_i):
        for j in range(N_j):
            m[i*cell_size:(i+1)*cell_size-1, j*cell_size:(j+1)*cell_size-1, :] = state.colouring[i, j, :]

    im = PILImage.fromarray(np.uint8(np.swapaxes(m, 0, 1)))

    if add_swapped_element_line:
        draw = PILImageDraw.Draw(im)
        draw.line(
            _line_coordinates(state, cell_size), 
            fill=0,
            width=cell_size // 25,
        )
        
    return im

def _line_coordinates(state: 'State', cell_size: int):
    """
    Returns the line coordinates in the PIL-coordinate-system.
    """
    (x, y), (x_in, y_in) = state.swapped_elements
    left_x, right_x = (min(x, x_in), max(x, x_in))
    bottom_y, top_y = (min(y, y_in), max(y, y_in))

    coord_set_for_drawing = (left_x, bottom_y, right_x, top_y)

    logger.debug(f'Drawing a line for coordinates {coord_set_for_drawing} derived from swapped elements {state.swapped_elements}.')

    # Scale the return value to the actual image coordinates:
    return tuple(
        [a * cell_size + round(0.5 * cell_size) for a in coord_set_for_drawing]
    )
