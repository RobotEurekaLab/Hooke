import sys, importlib
from common import *
mod, part = sys.argv[1], int(sys.argv[2]); ts = [float(x) for x in sys.argv[3:]]
m = importlib.import_module(mod)
imgs = [to_img(post(m.render(t), int(t*24), part)).resize((960, 540)) for t in ts]
cols = 2; rows = (len(imgs)+1)//2
sheet = Image.new('RGB', (960*cols, 540*rows))
for i, im in enumerate(imgs): sheet.paste(im, ((i%cols)*960, (i//cols)*540))
sheet.save(f'build/sheet_{mod}.png'); print('ok')
