
def convert_top_left_origin_to_matplotlib(coords, fig_height):
    """
    Converts coordinates from a top-left origin system to matplotlib's bottom-left origin.

    Parameters:
    - coords: A list or tuple [x, y, width, height] where:
        * (x, y) is the top-left corner.
        * width and height are the size of the element.
    - fig: The matplotlib figure object (needed to get dimensions for scaling).

    Returns:
    - A list [x_new, y_new, width_new, height_new] in matplotlib coordinates.
    """
    # Extract the original top-left coordinates
    x, y, width, height = coords

    # Convert the y-coordinate to matplotlib's bottom-left origin
    y_new = fig_height - y - height  # Flip the y-axis

    # Return the new coordinates (x remains the same, y is flipped)
    return [x, y_new, width, height]
