import sfml.sf as sf
import numpy as np
from ifs_general import ifs_general
from ifs_cpu import ifs_cpu
from ifs_gpu import ifs_gpu
import math
class generator(object):
    winH=1080

    def __init__(self):
        self.view=sf.View(sf.Rect((0,0),(500,500)))
        self.view.viewport=(sf.Rect((0.474,0.2685),(0.26,0.463)))
        self.settings=sf.ContextSettings()
        self.settings.antialiasing_level=8
        self.window=sf.RenderWindow(sf.VideoMode(1920,1080),"IFS Flame Fractal Generator",sf.Style.DEFAULT,self.settings)
        self.window.framerate_limit=60
        self.general=ifs_general()
        self.cpu_ifs=ifs_cpu(self.general)
        self.gpu_ifs=ifs_gpu(self.general)
    def loop(self):
        pass

#test
gen=generator()
gen.general.affineNum=2
gen.general.unlinearNum=0
gen.general.zoom=1
gen.general.boundedFunctionScale=1
gen.general.gauss=9
gen.general.iterationsInMld=1
gen.general.affineTransforms[0]=[0.49,0.1,-0.2,
                                 -0.19,0.69,0.39,
                                 0.5,0.5]
gen.general.affineTransforms[1]=[0.59,-0.29,0.1,0.3,0.49,0.1,0.5,0.5]
gen.general.current[0]=0
#gen.cpu_ifs.render_to_file()

#img=sf.Image.create(2048,2048)
#img.create(2048, 2048)
#img.to_file("ifs.png")

gen.gpu_ifs.render_to_file()