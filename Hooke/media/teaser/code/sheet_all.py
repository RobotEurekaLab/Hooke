from common import *
import importlib, scene8
shots = [("scene1",2.6),("scene2",1.4),("scene3",0.3),("scene4",0.5),("scene4",2.0),
         ("scene5",5.0),("scene6",3.6),("scene7",2.4),("scene8",5.0)]
imgs=[]
for mod,t in shots:
    m=importlib.import_module(mod)
    imgs.append(to_img(post(m.render(t), int(t*24), 1 if mod in("scene1","scene2","scene3","scene4") else 2)).resize((640,360)))
imgs.append(to_img(post(scene8.render_end(2.2), 20, 2)).resize((640,360)))
sh=Image.new('RGB',(640*3,360*4))
for i,im in enumerate(imgs): sh.paste(im,((i%3)*640,(i//3)*360))
sh.save('build/sheet_all.png'); print('ok')
