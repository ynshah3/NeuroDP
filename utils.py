import numpy as np
from PIL import Image


RED = 0.2126
GREEN = 0.7152
BLUE = 0.0722


def luminance_helper(v):
    v /= 255.0
    if v <= 0.03928:
        return v / 12.92
    else:
        return pow((v + 0.055) / 1.055, 2.4)


def luminance(rgb):
    a = list(map(luminance_helper, rgb))
    return a[0] * RED + a[1] * GREEN + a[2] * BLUE


def contrast(rgb1, rgb2):
    lum1 = luminance(rgb1)
    lum2 = luminance(rgb2)
    brightest = max(lum1, lum2)
    darkest = min(lum1, lum2)
    return (brightest + 0.05) / (darkest + 0.05)


def flip_pixels(image, percentage):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate the number of pixels to flip
    total_pixels = image_array.shape[0] * image_array.shape[1]
    pixels_to_flip = int((percentage / 100) * total_pixels)

    # Randomly select pixels to flip
    flip_indices = np.random.choice(total_pixels, pixels_to_flip, replace=False)
    for index in flip_indices:
        row = index // image_array.shape[1]
        col = index % image_array.shape[1]
        # Flip pixel color across all channels
        image_array[row, col, :] = 255 - image_array[row, col, :]

    # Convert the modified array back to an image
    modified_image = Image.fromarray(image_array)

    return modified_image