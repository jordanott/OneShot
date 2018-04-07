# text to image
from PIL import Image, ImageDraw, ImageFont

def create_img(text,name):
    img = Image.new('L', (400, 400))
    d = ImageDraw.Draw(img)
    d.text((15, 15), text,fill=255)
    img.save(name + '.png')
