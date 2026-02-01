from PIL import Image

img = Image.open("/Users/chensicheng/Downloads/PathRWKV/assets/banner_raw.jpg")
new_size = (60, 16)
resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
resized_img.save(
    "/Users/chensicheng/Downloads/PathRWKV/assets/banner.jpg", dpi=(300, 300)
)
