import random
from PIL import Image, ImageDraw, ImageFont
import os
from utils import flip_pixels


OPERATIONS = {
    "plus": "+",
    "minus": "-",
    "multiply": "x",
    "divide": "/",
    "percent": "%"
}


def create_image(operation, percent, img_id, folder):
    """Creates and saves an image with the given numeral and colors."""
    font = ImageFont.load_default()
    image_size = (224, 224)
    font_size = 170
    image = Image.new('RGB', image_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./assets/arial.ttf", font_size)
    text_width, text_height = draw.textsize(operation, font=font)
    position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height - 30) / 2)
    draw.text(position, operation, fill=(255, 255, 255), font=font)

    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"{folder}/{img_id:08d}_{(percent / 100):.2f}.png"
    output_image = flip_pixels(image, percent)
    output_image.save(filename)


def main():
    total_images = 1000
    images_per_operation = total_images // 5
    percent = list(range(1, 41))

    id = 0
    for ctr, operation in enumerate(OPERATIONS.values()):
        folder = f"datasets/fragments/test/{ctr}"
        for p in percent:
            for j in range(5):
                img_id = id
                create_image(operation, p, img_id, folder)
                id += 1


if __name__ == "__main__":
    main()
