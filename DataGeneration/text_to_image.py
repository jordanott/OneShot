# text to image
from PIL import Image, ImageDraw, ImageFont

def create_img(text,name):
    img = Image.new('L', (200, 200))
    d = ImageDraw.Draw(img)
    d.text((5, 5), text,fill=255)
    img.save(name + '.png')
