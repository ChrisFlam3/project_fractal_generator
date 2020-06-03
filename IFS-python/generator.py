import sfml.sf as sf
import numpy as np
from ifs_general import ifs_general
from ifs_cpu import ifs_cpu
from ifs_gpu import ifs_gpu
from ui import ui
import math
import pynk

#IFS generator class
class generator(object):
    winH=1080
    
    #window and logic initialization
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
        self.gui=ui()

    #main loop
    def loop(self):
        while(self.window.is_open):
          
            #INPUT
            pynk.lib.nk_input_begin(self.gui.ctx)
            for event in self.window.events:
                if (event.type == sf.Event.CLOSED):
                    self.window.close()
                elif(event.type==sf.Event.RESIZED):
                    self.winH=event.items()._mapping.event.height
                    self.window.view=sf.View(sf.Rect((0,0),(event.items()._mapping.event.width, self.winH)))
                self.gui.eventsToGui(event)
            pynk.lib.nk_input_end(self.gui.ctx)

            #draw
            self.gui.gui_loop(self.winH,self.window,self.cpu_ifs,self.gpu_ifs,self.general)
            self.window.clear()
            self.gui.draw(self.window)
            if self.cpu_ifs.screen!=None:
                defaultView = self.window.view
                self.window.view=self.view
                self.window.draw(self.cpu_ifs.screen)
                self.window.view=defaultView
            self.window.display()
           

#test
gen=generator()
gen.loop()
