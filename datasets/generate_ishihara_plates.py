from PIL import Image, ImageDraw, ImageFont
import os
import random
from utils import contrast


COLORS = {
    "Red": (255, 0, 0),
    "Red-Orange": (255, 69, 0),
    "Orange": (255, 165, 0),
    "Yellow-Orange": (255, 215, 0),
    "Yellow": (255, 255, 0),
    "Yellow-Green": (173, 255, 47),
    "Green": (0, 128, 0),
    "Blue-Green": (0, 255, 127),
    "Blue": (0, 0, 255),
    "Blue-Violet": (138, 43, 226),
    "Violet": (127, 0, 255),
    "Red-Violet": (199, 21, 133)
}

THRESHOLD = 0.003


def generate_color(color, tl, th, sign):
    """Generates a random color and a slightly different shade for the foreground."""
    # bg_color = tuple(random.randint(0, 255) for _ in range(3))
    bg_color = color
    rangeR, rangeG, rangeB = list(range(255)), list(range(255)), list(range(255))
    random.shuffle(rangeR)
    random.shuffle(rangeG)
    random.shuffle(rangeB)
    idx = [0,1,2]
    random.shuffle(idx)
    ranges = [rangeR, rangeG, rangeB]
    for distortionX in ranges[idx[0]]:
        for distortionY in ranges[idx[1]]:
            for distortionZ in ranges[idx[2]]:
                fg_color = (
                    min(max(bg_color[0] + sign * distortionX, 0), 255),
                    min(max(bg_color[1] + sign * distortionY, 0), 255),
                    min(max(bg_color[2] + sign * distortionZ, 0), 255)
                )
                ct = contrast(bg_color, fg_color)
                if tl <= ct <= th:
                    return bg_color, fg_color, ct
    print(f"Couldn't find any: {color}, {tl}-{th}, {sign}")
    return None, None, None


def create_image(numeral, bg_color, fg_color, ct, img_id, folder):
    """Creates and saves an image with the given numeral and colors."""
    font = ImageFont.load_default()
    image_size = (224, 224)
    font_size = 170
    image = Image.new('RGB', image_size, color=bg_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./assets/Monas-BLBW8.ttf", font_size)
    text_width, text_height = draw.textsize(str(numeral), font=font)
    position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height - 30) / 2)
    draw.text(position, str(numeral), fill=fg_color, font=font)

    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"{folder}/{img_id:08d}_{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}_{ct:.3f}.png"
    image.save(filename)


def main():
    total_images = 1000
    images_per_numeral = total_images // 10
    contrasts = [(1.001, 1.05), (1.05, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.5)]

    for numeral in range(10):
        i = 0
        folder = f"plates/{numeral}"
        for color in COLORS.values():
            for tl, th in contrasts:
                for sign in [-1, 1]:
                    bg_color, fg_color, ct = generate_color(color, tl, th, sign)
                    if bg_color is None:
                        bg_color, fg_color, ct = generate_color(color, tl, th, sign * -1)
                    if bg_color is None:
                        continue
                    img_id = numeral * images_per_numeral + i
                    create_image(numeral, bg_color, fg_color, th, img_id, folder)
                    i += 1
                    print(f'{numeral}: {i}')


if __name__ == "__main__":
    main()
