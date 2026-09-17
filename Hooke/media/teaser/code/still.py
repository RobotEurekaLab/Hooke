import sys, time, importlib
from common import *
mod, t, part = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
m = importlib.import_module(mod)
t0 = time.time(); fr = m.render(t); t1 = time.time()
out = post(fr, int(t*24), part, grade=None)
print('render', round(t1-t0,2), 'post', round(time.time()-t1,2))
to_img(out).save(f'build/still_{mod}_{t}.png')
