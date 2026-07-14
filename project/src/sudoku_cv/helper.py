import numpy as np


def make_grid_compilation(images: list[np.ndarray], gap: int = 2) -> np.ndarray:
    """Lay exactly 81 single-channel images out in a 9x9 grid for visual inspection."""
    if len(images) != 81:
        raise ValueError(f"Expected 81 images, received {len(images)}")

    # Calculate dimensions based on the largest image in the list
    cell_height = max(img.shape[0] for img in images)
    cell_width = max(img.shape[1] for img in images)
    
    canvas_height = 9 * cell_height + 10 * gap
    canvas_width = 9 * cell_width + 10 * gap
    
    is_color = images[0].ndim == 3
    if is_color:
        canvas = np.full(
            (canvas_height, canvas_width, 3),
            255,
            dtype=np.uint8,
        )
    else:
        canvas = np.full(
            (canvas_height, canvas_width),
            255,
            dtype=np.uint8,
        )
    

    for index, img in enumerate(images):
        row, column = divmod(index, 9)
        y = gap + row * (cell_height + gap)
        x = gap + column * (cell_width + gap)
        
        # Center the image in its designated slot
        start_y = y + (cell_height - img.shape[0]) // 2
        start_x = x + (cell_width - img.shape[1]) // 2
        
        canvas[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]] = img
        
    return canvas