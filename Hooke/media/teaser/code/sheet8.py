from common import *
import scene8
ts = [0.5, 2.0, 3.6, 6.0]
imgs = [to_img(post(scene8.render(t), int(t*24), 2)).resize((960, 540)) for t in ts]
imgs.append(to_img(post(scene8.render_end(2.5), 10, 2)).resize((960, 540)))
sheet = Image.new('RGB', (1920, 540*3))
for i, im in enumerate(imgs): sheet.paste(im, ((i%2)*960, (i//2)*540))
sheet.save('build/sheet_scene8.png')
