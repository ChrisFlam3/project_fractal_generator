import pynk
import sfml.sf as sf
import inspect
import math
from OpenGL.GL import *

#font baking
@pynk.ffi.def_extern()
def pynk_text_width_callback(handle,height,text,len):
        type=pynk.ffi.from_handle(handle.ptr)
        temp=sf.Text()
        temp.font=type
        temp.string=pynk.ffi.string(text, len).decode('ascii')
        temp.character_size=height
        text_width=temp.find_character_pos(len).x-temp.find_character_pos(0).x
        return text_width

#class for ui logic
class ui(object):
    def __init__(self):
        self.arial=sf.Font.from_file("arial.ttf")
        self.ctx=pynk.ffi.new("struct nk_context*", None)
        self.font=pynk.ffi.new("struct nk_user_font*",None)
        self.font_handle=pynk.ffi.new_handle(self.arial)
        self.font.userdata.ptr=self.font_handle
        self.font.height = 15
        self.font.width =pynk.lib.pynk_text_width_callback
        pynk.lib.nk_init_default(self.ctx,self.font)
        self.__strings = []
    #connect gui logic with graphics
        self.declare("affineNum",2,"int*")
        self.affineNum[0]=2
        self.declare("zoom",1,"float*")
        self.declare("boundedFunctionScale",1,"float*")
        self.declare("gauss",9,"int*")
        self.declare("iterationsInMld",1,"int*")
        self.declare("unlinearNum",0,"int*")
        self.unlinearNum=0
        self.declare("renderType",0,"int*")
        self.renderType=0
        self.declare("textLen",0,"int*")
        self.declare("current",(0,0,0,0,0,0),"int[6]")
        self.declare("popup",0,"int*")
        self.declare("currentName",0,"int*")
        self.declare("det",0,"int*")
        tmp=self.loadNames()
        self.namesNum=tmp[1]
        self.names=tmp[0]
        self.declare("affineTransforms",[],"float[6][8]")
        self.declare("postTransforms",[],"float[6][6]")
        for i in range(6):
            self.postTransforms[i][0] = 1
            self.postTransforms[i][4] = 1

        #self.affineTransforms = [[0 for i in range(8)] for y in range(6)]
        self.declare("temp",[],"float[6][8]")
        #self.temp=[[0 for i in range(8)] for y in range(6)]
        self.declare_string_array("functions", ["Spherical".encode('ascii'),"Swirl".encode('ascii'),"Handkerchief".encode('ascii'),"Sinusoidal".encode('ascii'),"Linear".encode('ascii'),"Horseshoe".encode('ascii'),"Polar".encode('ascii'),"Disc".encode('ascii'),"Heart".encode('ascii'),"Spiral".encode('ascii'),"Hyperbolic".encode('ascii'),"Diamond".encode('ascii'),"Ex".encode('ascii'),"Julia".encode('ascii')])
        self.INT_MIN=-4*10**9
        self.INT_MAX=4*10**9

    #draw calls for ui command queue
    def nuklearSfmlDrawRectFilled(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_rect_filled*",cmd)
        rectangle=sf.RectangleShape()
        rectangle.fill_color=sf.Color(p.color.r, p.color.g, p.color.b, p.color.a)
        rectangle.size=sf.Vector2(p.w, p.h)
        rectangle.position=sf.Vector2(p.x, p.y)
        window.draw(rectangle)

    def nuklearSfmlDrawText(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_text*",cmd)
        font = pynk.ffi.from_handle(p.font.userdata.ptr)
        text=sf.Text()
        text.font=font
        text.string=pynk.ffi.string(p.string, p.length).decode('ascii')
        text.character_size=p.height
        text.color=sf.Color(p.foreground.r,p.foreground.g,p.foreground.b,p.foreground.a)
        text.position=sf.Vector2(p.x,p.y)
        window.draw(text)

    #scissor test
    def nuklearSfmlDrawScissor(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_scissor*",cmd)
        glEnable(GL_SCISSOR_TEST)
        glScissor(GLint(p.x),window.size.y - (p.y + p.h),p.w,p.h)

    def nuklearSfmlDrawRectOutline(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_rect*",cmd)
        rect=sf.RectangleShape()
        rect.size=sf.Vector2(p.w, p.h)
        rect.position=sf.Vector2(p.x, p.y)
        rect.outline_thickness=p.line_thickness
        rect.fill_color=sf.Color(0,0,0,0)
        rect.outline_color=sf.Color(p.color.r, p.color.g, p.color.b, p.color.a)
        window.draw(rect)
    
    def nuklearSfmlDrawCircleFilled(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_circle_filled*",cmd)
        circle=sf.CircleShape()
        circle.radius=p.h/2
        circle.position=sf.Vector2(p.x, p.y)
        circle.fill_color=sf.Color(p.color.r, p.color.g, p.color.b, p.color.a)
        window.draw(circle)

    def nuklearSfmlDrawTriangleFilled(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_triangle_filled*",cmd)
        convex=sf.ConvexShape()
        convex.point_count=3
        convex.set_point(0,sf.Vector2(p.a.x, p.a.y))
        convex.set_point(1, sf.Vector2(p.b.x, p.b.y))
        convex.set_point(2, sf.Vector2(p.c.x, p.c.y))
        convex.fill_color=sf.Color(p.color.r, p.color.g, p.color.b, p.color.a)
        window.draw(convex)

    def nuklearSfmlDrawLine(self,cmd,window):
        p=pynk.ffi.cast("struct nk_command_line*",cmd)
        rect=sf.RectangleShape()
        rect.size=sf.Vector2(p.line_thickness, abs(p.begin.y-p.end.y))
        rect.position=sf.Vector2(p.begin.x, p.begin.y)
        rect.outline_thickness=p.line_thickness
        rect.fill_color=sf.Color(p.color.r, p.color.g, p.color.b, p.color.a)
        rect.outline_color=sf.Color(p.color.r, p.color.g, p.color.b, p.color.a)
        window.draw(rect)

    #nuklear utility for c data
    def declare(self, name, initialiser, ctype=None):
        if not hasattr(self, name):
            value = initialiser
            if callable(value):
                value = value()
            if ctype is not None:
                value = pynk.ffi.new(ctype, value)
            setattr(self, name, value)

    def declare_string_array(self, name, lst):
        keepalive = [pynk.ffi.new("const char[]", s) for s in lst]
        self.__strings += keepalive
        self.declare(name, keepalive, "const char*[]")

    def declare_string_buffers(self, name, num_strings, string_length):
        keepalive = [pynk.ffi.new("char[%s]" % string_length) for i in xrange(num_strings)]
        self.__strings += keepalive
        self.declare(name, keepalive, "char*[]")

    def tree_push(self, ctx, tree_type, title, state, unique_id_str):
        callerframerecord = inspect.stack()[1]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        return pynk.lib.nk_tree_push_hashed(ctx, tree_type, title, state, info.filename.encode('ascii'), len(info.filename), info.lineno+unique_id_str)


    def get_field_cdata(self, cdata, name_str):
        names = name_str.split(".")
        for name in names:
            cdata = pynk.ffi.addressof(cdata, name)
        return cdata
    def event_attr(self,event):
        return event.items()._mapping.event

    def saveOffset(self,name,length,affineNum,unlinearNum,zoom,gauss,iterationsInMld,affineTransforms,current,boundedFunctionScale,ctx):
        n = pynk.ffi.string(name)[0:length[0]].decode('ascii')
        if (length[0] > 20):
            return 1

        k=open("offset.txt","r")
        archive=k.read()
        if (archive.find("#" + str(n),0) != -1):
            return 2
        k.close()
        k=open("offset.txt","a+")
        k.write("\n#"+str(n)+"\n")
        k.write(str(affineNum[0]) + " " + str(unlinearNum) + " " + str(zoom[0]) + " " + str(boundedFunctionScale[0]) + " " + str(gauss[0]) + " " + str(iterationsInMld[0]) +"\n")
        for i in range(affineNum[0]):
            for y in range(8):
                k.write(str(affineTransforms[i][y]) + " ")
            k.write("\n")

        for i in range(unlinearNum * 5+1):
            k.write(str(current[i]) + " ")
        k.write("\n")
        k.write( "$end"+"\n")
        k.close()
        return 0

    def loadNames(self):
        namesNum = 0
        container="".encode('ascii')
        k=open("offset.txt","r")
        temp=k.read()
        start = 0
        end = 0
        i=-1
        while True:
            i=i+1
            start = temp.find("#",start)
            if (start == -1):
                break

            start=start+1
            end = temp.find("\n", start)
            container+=temp[start:end].encode('ascii')+'\0'.encode('ascii')
            namesNum = i + 1
        k.close()
        return (container,namesNum)

    def deleteOffset(self,deleteNum):
        k=open("offset.txt","r")
        archive=k.read()
        k.close()
        pos = 0
        end=0
        for i in range(deleteNum+1):
            pos = archive.find("#", pos + 1)

        pos =pos- 1
        end = archive.find("$", pos)
        end = end+  5
        archive=archive[0:pos]+archive[end:]
        k=open("offset.txt", "w")
        k.write(archive)
        k.close()

    def loadOffset(self):
        k=open("offset.txt","r")
        archive=k.read()

        pos = 0
        for i in range(self.currentName+1):
            pos = archive.find("#", pos + 1)

        pos = archive.find("\n", pos + 1)
        pos =pos+ 1


        end = archive.find(" ", pos)
        self.affineNum[0] = int(archive[pos:end])

        pos += end - pos + 1
        end = archive.find(" ", pos)
        self.unlinearNum = int(archive[pos:end])

        pos += end - pos + 1
        end = archive.find(" ", pos)
        self.zoom[0] = float(archive[pos:end])

        pos += end - pos + 1
        end = archive.find(" ", pos)
        self.boundedFunctionScale[0] = float(archive[pos:end])

        pos += end - pos + 1
        end = archive.find(" ", pos)
        self.gauss[0] = int(archive[pos:end])

        pos += end - pos + 1
        end = archive.find("\n", pos)
        self.iterationsInMld[0] = int(archive[pos:end])

        pos += end - pos + 1

        for i in range(self.affineNum[0]):
            for y in range(8):
                end = archive.find(" ", pos)
                self.affineTransforms[i][y] = float(archive[pos:end])
                pos += end - pos + 1

        for i in range(self.unlinearNum * 5+1):
            end = archive.find(" ", pos)
            self.current[i] = int(archive[pos:end])
            pos += end - pos + 1

        k.close()

    #pass data to algorithm
    def load_params(self,params):
        params.boundedFunctionScale=self.boundedFunctionScale[0]

        for i in range(6):
            for y in range(8):
                params.affineTransforms[i][y]=self.affineTransforms[i][y]

        for i in range(6):
            params.current[i]=self.current[i]
        params.unlinearNum=self.unlinearNum
        params.renderType=self.renderType
        params.affineNum=self.affineNum[0]
        params.zoom=self.zoom[0]
        params.gauss=self.gauss[0]
        params.iterationsInMld=self.iterationsInMld[0]

        for i in range(6):
            for y in range(6):
                params.postTransforms[i][y]=self.postTransforms[i][y]


    #gui logic loop
    def gui_loop(self,winH,window,ifs_cpu,ifs_gpu,parameters):
        EASY = 0
        HARD = 1
        self.declare("op", 0, "int*")
        self.declare("value", 0.6, "float*")
        self.declare("i", 20, "int*")
        self.declare("text", ["a".encode('ascii')]*30, "char[]")
        if pynk.lib.nk_begin(self.ctx, "LeftPanel".encode('ascii'), pynk.lib.nk_rect(0, 0, 400, winH),pynk.lib.NK_WINDOW_BORDER):
            #self.ctx.current.scrollbar.x+=50
            pynk.lib.nk_layout_row_static(self.ctx, 30, 360, 1)
            pynk.lib.nk_label(self.ctx, "IFS FLAME FRACTAL RENDERER".encode('ascii'), pynk.lib.NK_TEXT_CENTERED)
            


            pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 40, 2)
            
            pynk.lib.nk_layout_row_push(self.ctx, 170)
            pynk.lib.nk_label(self.ctx, "Affine transforms:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_push(self.ctx, 80)
            pynk.lib.nk_property_int(self.ctx, "#".encode('ascii'), 2, self.affineNum, 6, 1, 0)

            pynk.lib.nk_layout_row_end(self.ctx)



            pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 40, 2)

            pynk.lib.nk_layout_row_push(self.ctx, 170)
            pynk.lib.nk_label(self.ctx, "Zoom:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_push(self.ctx, 80)
            pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), 0.1, self.zoom, 10, 0.1, 0)

            pynk.lib.nk_layout_row_end(self.ctx)



            pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 40, 2)

            pynk.lib.nk_layout_row_push(self.ctx, 170)
            pynk.lib.nk_label(self.ctx, "Bounded function scale:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_push(self.ctx, 80)
            pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), 1, self.boundedFunctionScale, 100, 1, 0)

            pynk.lib.nk_layout_row_end(self.ctx)



            pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 40, 2)

            pynk.lib.nk_layout_row_push(self.ctx, 170)
            pynk.lib.nk_label(self.ctx, "Gaussian blur:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_push(self.ctx, 80)
            pynk.lib.nk_property_int(self.ctx, "#".encode('ascii'), 0, self.gauss, 10, 1, 0)

            pynk.lib.nk_layout_row_end(self.ctx)



            pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 40, 2)

            pynk.lib.nk_layout_row_push(self.ctx, 170)
            pynk.lib.nk_label(self.ctx, "Iterations in billions:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_push(self.ctx, 80)
            pynk.lib.nk_property_int(self.ctx, "#".encode('ascii'), 1, self.iterationsInMld, 100, 1, 0)

            pynk.lib.nk_layout_row_end(self.ctx)


            pynk.lib.nk_layout_row_static(self.ctx, 20, 300, 0)
            pynk.lib.nk_layout_row_static(self.ctx, 30, 300, 1)
            pynk.lib.nk_label(self.ctx, "Unlinear transforms:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_static(self.ctx, 30, 200, 1)
            if (pynk.lib.nk_option_label(self.ctx, "one for all affine".encode('ascii'), self.unlinearNum == 0)):
                self.unlinearNum = 0
            pynk.lib.nk_layout_row_static(self.ctx, 30, 200, 1)
            if (pynk.lib.nk_option_label(self.ctx, "one for each affine".encode('ascii'), self.unlinearNum == 1)):
                self.unlinearNum = 1

            pynk.lib.nk_layout_row_static(self.ctx, 20, 300, 0)
            pynk.lib.nk_layout_row_static(self.ctx, 30, 300, 1)
            pynk.lib.nk_label(self.ctx, "Render type:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
            pynk.lib.nk_layout_row_static(self.ctx, 30, 200, 1)
            if (pynk.lib.nk_option_label(self.ctx, "CPU 1 thread".encode('ascii'), self.renderType == 0)):
               self.renderType = 0
            pynk.lib.nk_layout_row_static(self.ctx, 30, 200, 1)
            if (pynk.lib.nk_option_label(self.ctx, "GPU CUDA".encode('ascii'), self.renderType == 1)):
                self.renderType = 1


            pynk.lib.nk_layout_row_static(self.ctx, 30, 300, 1)
            if (pynk.lib.nk_button_label(self.ctx, "Render low quality to overview".encode('ascii'))):
                self.load_params(parameters)
                ifs_cpu.render_one_core(1)
            pynk.lib.nk_layout_row_static(self.ctx, 10, 300, 0)
            pynk.lib.nk_layout_row_static(self.ctx, 30, 300, 1)
            if (pynk.lib.nk_button_label(self.ctx, "Render high quality to file".encode('ascii'))):
                if(self.renderType==0):
                    self.load_params(parameters)
                    ifs_cpu.render_one_core(0)
                else:
                    self.load_params(parameters)
                    ifs_gpu.render_to_file()
            pynk.lib.nk_layout_row_static(self.ctx, 30, 300, 0)
            
            
            if (self.tree_push(self.ctx, pynk.lib.NK_TREE_TAB, "Offsets".encode('ascii'), pynk.lib.NK_MINIMIZED,1)):
                pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 3)
                pynk.lib.nk_label(self.ctx, "Name:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                pynk.lib.nk_edit_string(self.ctx, pynk.lib.NK_EDIT_SIMPLE, self.text, self.textLen, 30, pynk.ffi.addressof(pynk.lib,"nk_filter_default"))
                if (pynk.lib.nk_button_label(self.ctx, "Save".encode('ascii'))):
                    self.popup = self.saveOffset(self.text, self.textLen, self.affineNum, self.unlinearNum, self.zoom, self.gauss, self.iterationsInMld, self.affineTransforms, self.current,self.boundedFunctionScale, self.ctx)
                    (self.names,self.namesNum)=self.loadNames()


                if (self.popup == 2):
                    s=pynk.lib.nk_rect(500, 540, 220, 90)
                    if (pynk.lib.nk_popup_begin(self.ctx, pynk.lib.NK_POPUP_STATIC, "Critical error".encode('ascii'), 0, s)):
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                        pynk.lib.nk_label(self.ctx, "Name already exists".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                        if (pynk.lib.nk_button_label(self.ctx, "OK".encode('ascii'))):
                            self.popup = 0
                            pynk.lib.nk_popup_close(self.ctx)

                        pynk.lib.nk_popup_end(self.ctx)
                elif(self.popup==1):
                    s=pynk.lib.nk_rect(500, 540, 220, 90)
                    if (pynk.lib.nk_popup_begin(self.ctx, pynk.lib.NK_POPUP_STATIC, "Critical error".encode('ascii'), 0, s)):
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                        pynk.lib.nk_label(self.ctx, "Name is too long".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                        if (pynk.lib.nk_button_label(self.ctx, "OK".encode('ascii'))):
                            self.popup = 0
                            pynk.lib.nk_popup_close(self.ctx)

                        pynk.lib.nk_popup_end(self.ctx)

                elif (self.popup == 3):
                    s=pynk.lib.nk_rect(500, 540, 220, 90)
                    if (pynk.lib.nk_popup_begin(self.ctx, pynk.lib.NK_POPUP_STATIC, "Confirm deletion".encode('ascii'), 0, s)):
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                        pynk.lib.nk_label(self.ctx, "Delete is permanently".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 2)
                        if (pynk.lib.nk_button_label(self.ctx, "OK".encode('ascii'))):
                            self.deleteOffset(self.currentName)
                            self.currentName = 0
                            (self.names,self.namesNum)=self.loadNames()
                            self.popup = 0
                            pynk.lib.nk_popup_close(self.ctx)

                    if (pynk.lib.nk_button_label(self.ctx, "Cancel".encode('ascii'))):
                        self.popup = 0
                        pynk.lib.nk_popup_close(self.ctx)

                    pynk.lib.nk_popup_end(self.ctx)

                pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 3)

                self.currentName = pynk.lib.nk_combo_string(self.ctx, self.names, int(pynk.ffi.cast("int", self.currentName)), self.namesNum, 30, pynk.lib.nk_vec2(200, 200))
                if (pynk.lib.nk_button_label(self.ctx, "Delete".encode('ascii'))):
                    self.popup = 3

                if (pynk.lib.nk_button_label(self.ctx, "Load".encode('ascii'))):
                    self.loadOffset()


                pynk.lib.nk_tree_pop(self.ctx)
            
            if (self.tree_push(self.ctx, pynk.lib.NK_TREE_TAB, "Affine transforms".encode('ascii'), pynk.lib.NK_MINIMIZED,2)):
               
                for i in range(self.affineNum[0]):
        
                    pynk.lib.nk_layout_row_static(self.ctx, 30, 300, 1)
                    pynk.lib.nk_label(self.ctx, "Transformation matrix:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 3)
                    pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.affineTransforms[i],0), self.INT_MAX, 0.1, 0)
                    pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.affineTransforms[i],1), self.INT_MAX, 0.1, 0)
                    pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.affineTransforms[i],2), self.INT_MAX, 0.1, 0)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 3)
                    pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.affineTransforms[i],3), self.INT_MAX, 0.1, 0)
                    pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.affineTransforms[i],4), self.INT_MAX, 0.1, 0)
                    pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.affineTransforms[i],5), self.INT_MAX, 0.1, 0)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 10, 0)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                    pynk.lib.nk_property_float(self.ctx, "#Color grayscale".encode('ascii'), 0, pynk.ffi.addressof(self.affineTransforms[i],6), 1, 0.1, 0)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 1)
                    pynk.lib.nk_property_float(self.ctx, "#Probability".encode('ascii'), 0, pynk.ffi.addressof(self.affineTransforms[i],7), 1, 0.1, 0)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 10, 0)
                    pynk.lib.nk_layout_row_static(self.ctx, 30, 100, 3)
                    if (pynk.lib.nk_button_label(self.ctx, "Clear".encode('ascii'))):
                        for y in range(8):
                            self.affineTransforms[i][y] = 0

                    self.det = self.affineTransforms[i][0] * self.affineTransforms[i][4] - self.affineTransforms[i][1] * self.affineTransforms[i][3]
                    pynk.lib.nk_label(self.ctx, "  Determinant:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                    pynk.lib.nk_label(self.ctx, str(self.det).encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                    if (self.tree_push(self.ctx, pynk.lib.NK_TREE_NODE, "Imminent transforms".encode('ascii'), pynk.lib.NK_MINIMIZED,i+4)):
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 4)
                        pynk.lib.nk_label(self.ctx, "Translation:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_property_float(self.ctx, "#x:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],0), self.INT_MAX, 0.1, 0.1)
                        pynk.lib.nk_property_float(self.ctx, "#y:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],1), self.INT_MAX, 0.1, 0.1)
                        if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                            self.affineTransforms[i][2] = self.affineTransforms[i][0] * self.temp[i][0] + self.affineTransforms[i][1] * self.temp[i][1] + self.affineTransforms[i][2]
                            self.affineTransforms[i][5] = self.affineTransforms[i][3] * self.temp[i][0] + self.affineTransforms[i][4] * self.temp[i][1] + self.affineTransforms[i][5]

                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 4)
                        pynk.lib.nk_label(self.ctx, "Scale:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_property_float(self.ctx, "#x:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],2), self.INT_MAX, 0.1, 0.1)
                        pynk.lib.nk_property_float(self.ctx, "#y:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],3), self.INT_MAX, 0.1, 0.1)
                        if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                            self.affineTransforms[i][0] = self.affineTransforms[i][0] * self.temp[i][2]
                            self.affineTransforms[i][1] = self.affineTransforms[i][1] * self.temp[i][3]

                            self.affineTransforms[i][3] = self.affineTransforms[i][3] * self.temp[i][2]
                            self.affineTransforms[i][4] = self.affineTransforms[i][4] * self.temp[i][3]

                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 4)
                        pynk.lib.nk_label(self.ctx, "Shear:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_property_float(self.ctx, "#x:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],4), self.INT_MAX, 0.1, 0.1)
                        pynk.lib.nk_property_float(self.ctx, "#y:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],5), self.INT_MAX, 0.1, 0.1)
                        if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                            self.affineTransforms[i][0] = self.affineTransforms[i][0]+ self.affineTransforms[i][1]*self.temp[i][5]
                            self.affineTransforms[i][1] = self.affineTransforms[i][0] * self.temp[i][4]+ self.affineTransforms[i][1]

                            self.affineTransforms[i][3] = self.affineTransforms[i][3] + self.affineTransforms[i][4]* self.temp[i][5]
                            self.affineTransforms[i][4] = self.affineTransforms[i][3] * self.temp[i][4]+ self.affineTransforms[i][4]

                        pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 30, 3)
                        pynk.lib.nk_layout_row_push(self.ctx, 82)
                        pynk.lib.nk_label(self.ctx, "Rotate:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_layout_row_push(self.ctx, 169)
                        pynk.lib.nk_property_float(self.ctx, "#Angle:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],6), self.INT_MAX, 0.1, 0.1)
                        pynk.lib.nk_layout_row_push(self.ctx, 83)
                        if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                            self.temp[i][7] = self.temp[i][6] * 3.141592/180
                            self.affineTransforms[i][0] = self.affineTransforms[i][0] * math.cos(self.temp[i][7]) + self.affineTransforms[i][1] * math.sin(self.temp[i][7])
                            self.affineTransforms[i][1] = -self.affineTransforms[i][0] * math.sin(self.temp[i][7]) + self.affineTransforms[i][1] * math.cos(self.temp[i][7])

                            self.affineTransforms[i][3] = self.affineTransforms[i][3] * math.cos(self.temp[i][7]) + self.affineTransforms[i][4] * math.sin(self.temp[i][7])
                            self.affineTransforms[i][4] = -self.affineTransforms[i][3] * math.sin(self.temp[i][7]) + self.affineTransforms[i][4] * math.cos(self.temp[i][7])
                            
                        pynk.lib.nk_tree_pop(self.ctx)
                        
                    if (self.tree_push(self.ctx, pynk.lib.NK_TREE_NODE, "Post Transform".encode('ascii'), pynk.lib.NK_MINIMIZED, i+5)):
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 3)
                        pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN,  pynk.ffi.addressof(self.postTransforms[i],0), self.INT_MAX, 0.1, 0)
                        pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN,  pynk.ffi.addressof(self.postTransforms[i],1), self.INT_MAX, 0.1, 0)
                        pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN,  pynk.ffi.addressof(self.postTransforms[i],2), self.INT_MAX, 0.1, 0)
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 3)
                        pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN,  pynk.ffi.addressof(self.postTransforms[i],3), self.INT_MAX, 0.1, 0)
                        pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN,  pynk.ffi.addressof(self.postTransforms[i],4), self.INT_MAX, 0.1, 0)
                        pynk.lib.nk_property_float(self.ctx, "#".encode('ascii'), self.INT_MIN,  pynk.ffi.addressof(self.postTransforms[i],5), self.INT_MAX, 0.1, 0)
                        pynk.lib.nk_layout_row_dynamic(self.ctx, 10, 0)
                        pynk.lib.nk_layout_row_static(self.ctx, 30, 100, 3)
                        if (pynk.lib.nk_button_label(self.ctx, "Clear".encode('ascii'))):
                            for y in range(6):
                                self.postTransforms[i][y] = 0

                        self.det = self.postTransforms[i][0] * self.postTransforms[i][4] - self.postTransforms[i][1] * self.postTransforms[i][3]
                        pynk.lib.nk_label(self.ctx, "  Determinant:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        pynk.lib.nk_label(self.ctx, str(self.det).encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                        if (self.tree_push(self.ctx, pynk.lib.NK_TREE_NODE, "Imminent post transforms".encode('ascii'), pynk.lib.NK_MINIMIZED, i+6)):
                            pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 4)
                            pynk.lib.nk_label(self.ctx, "Translation:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                            pynk.lib.nk_property_float(self.ctx, "#x:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],0), self.INT_MAX, 0.1, 0.1)
                            pynk.lib.nk_property_float(self.ctx, "#y:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],1), self.INT_MAX, 0.1, 0.1)
                            if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                                postTransforms[i][2] = postTransforms[i][0] * self.temp[i][0] + postTransforms[i][1] * self.temp[i][1] + postTransforms[i][2]
                                postTransforms[i][5] = postTransforms[i][3] * self.temp[i][0] + postTransforms[i][4] * self.temp[i][1] + postTransforms[i][5]

                            pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 4)
                            pynk.lib.nk_label(self.ctx, "Scale:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                            pynk.lib.nk_property_float(self.ctx, "#x:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],2), self.INT_MAX, 0.1, 0.1)
                            pynk.lib.nk_property_float(self.ctx, "#y:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],3), self.INT_MAX, 0.1, 0.1)
                            if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                                self.postTransforms[i][0] = self.postTransforms[i][0] * self.temp[i][2]
                                self.postTransforms[i][1] = self.postTransforms[i][1] * self.temp[i][3]

                                self.postTransforms[i][3] = self.postTransforms[i][3] * self.temp[i][2]
                                self.postTransforms[i][4] = self.postTransforms[i][4] * self.temp[i][3]

                            pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 4)
                            pynk.lib.nk_label(self.ctx, "Shear:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                            pynk.lib.nk_property_float(self.ctx, "#x:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],4), self.INT_MAX, 0.1, 0.1)
                            pynk.lib.nk_property_float(self.ctx, "#y:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],5), self.INT_MAX, 0.1, 0.1)
                            if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                                postTransforms[i][0] = postTransforms[i][0] + postTransforms[i][1] * self.temp[i][5]
                                postTransforms[i][1] = postTransforms[i][0] * self.temp[i][4] + postTransforms[i][1]

                                postTransforms[i][3] = postTransforms[i][3] + postTransforms[i][4] * self.temp[i][5]
                                postTransforms[i][4] = postTransforms[i][3] * self.temp[i][4] + postTransforms[i][4]

                            pynk.lib.nk_layout_row_begin(self.ctx, pynk.lib.NK_STATIC, 30, 3)
                            pynk.lib.nk_layout_row_push(self.ctx, 79)
                            pynk.lib.nk_label(self.ctx, "Rotate:".encode('ascii'), pynk.lib.NK_TEXT_LEFT)
                            pynk.lib.nk_layout_row_push(self.ctx, 162)
                            pynk.lib.nk_property_float(self.ctx, "#Angle:".encode('ascii'), self.INT_MIN, pynk.ffi.addressof(self.temp[i],6), self.INT_MAX, 0.1, 0.1)
                            pynk.lib.nk_layout_row_push(self.ctx, 79)
                            if (pynk.lib.nk_button_label(self.ctx, "Apply".encode('ascii'))):
                                self.temp[i][7] = self.temp[i][6] * 3.141592 / 180
                                postTransforms[i][0] = postTransforms[i][0] * cos(self.temp[i][7]) + postTransforms[i][1] * sin(self.temp[i][7])
                                postTransforms[i][1] = -postTransforms[i][0] * sin(self.temp[i][7]) + postTransforms[i][1] * cos(self.temp[i][7])

                                postTransforms[i][3] = postTransforms[i][3] * cos(self.temp[i][7]) + postTransforms[i][4] * sin(self.temp[i][7])
                                postTransforms[i][4] = -postTransforms[i][3] * sin(self.temp[i][7]) + postTransforms[i][4] * cos(self.temp[i][7])
                            
                            pynk.lib.nk_tree_pop(self.ctx)
                        pynk.lib.nk_tree_pop(self.ctx)
                    pynk.lib.nk_layout_row_dynamic(self.ctx, 30, 0)
                pynk.lib.nk_tree_pop(self.ctx)
            
            if (self.tree_push(self.ctx, pynk.lib.NK_TREE_TAB, "Unlinear transforms".encode('ascii'), pynk.lib.NK_MINIMIZED,3)):
                if (self.unlinearNum == 0):
                    pynk.lib.nk_layout_row_static(self.ctx, 25, 200, 1)
                    self.current[0] = pynk.lib.nk_combo(self.ctx, self.functions, len(self.functions), self.current[0], 25, pynk.lib.nk_vec2(200, 200))


                else:
                    for i in range(self.affineNum[0]):
                        pynk.lib.nk_layout_row_static(self.ctx, 25, 200, 1)
                        self.current[i] = pynk.lib.nk_combo(self.ctx, self.functions, len(self.functions), self.current[i], 25, pynk.lib.nk_vec2(200, 200))
                        pynk.lib.nk_layout_row_static(self.ctx, 10, 200, 0)


                pynk.lib.nk_tree_pop(self.ctx)

            pynk.lib.nk_layout_row_static(self.ctx, 300, 300, 0)
        
        pynk.lib.nk_end(self.ctx)
        #window.clear()

    #translate sfml input to nuklear
    def eventsToGui(self,evt):
        if (evt.type == sf.Event.MOUSE_BUTTON_PRESSED or evt.type == sf.Event.MOUSE_BUTTON_RELEASED):
            attr= self.event_attr(evt)
            down = evt.type == sf.Event.MOUSE_BUTTON_PRESSED
            x = attr.x
            y = attr.y
            if (attr.button == 0):
                pynk.lib.nk_input_button(self.ctx, pynk.lib.NK_BUTTON_LEFT, x, y, down)
            if (attr.button == 2):
                pynk.lib.nk_input_button(self.ctx, pynk.lib.NK_BUTTON_MIDDLE, x, y, down)
            if (attr.button == 1):
                pynk.lib.nk_input_button(self.ctx, pynk.lib.NK_BUTTON_RIGHT, x, y, down)
        elif (evt.type == sf.Event.MOUSE_MOVED):
            attr= self.event_attr(evt)
            pynk.lib.nk_input_motion(self.ctx, attr.x, attr.y)
        elif (evt.type == sf.Event.KEY_PRESSED or evt.type == sf.Event.KEY_RELEASED):
            attr= self.event_attr(evt)
            down = evt.type == sf.Event.KEY_PRESSED
            if (attr.code == sf.Keyboard.BACK_SPACE):
                pynk.lib.nk_input_key(self.ctx, pynk.lib.NK_KEY_BACKSPACE, down)
                pynk.lib.nk_input_key(self.ctx, pynk.lib.NK_KEY_BACKSPACE, 0)
            elif (attr.code == sf.Keyboard.RETURN):
                pynk.lib.nk_input_key(self.ctx, pynk.lib.NK_KEY_ENTER, down)

            if (down == 1):
                if (attr.code >= sf.Keyboard.NUM0 and attr.code <= sf.Keyboard.NUM9):
                    pynk.lib.nk_input_char(self.ctx, pynk.ffi.cast("char",attr.code + 22))
                elif (attr.code == sf.Keyboard.PERIOD):
                    pynk.lib.nk_input_char(self.ctx, pynk.ffi.cast("char",46))
                elif (attr.code == sf.Keyboard.DASH):
                    pynk.lib.nk_input_char(self.ctx, pynk.ffi.cast("char",45))
                elif (attr.code == sf.Keyboard.SPACE):
                    pynk.lib.nk_input_char(self.ctx, pynk.ffi.cast("char",32))
                elif (attr.code >= sf.Keyboard.A and attr.code <= sf.Keyboard.Z):
                    if (sf.Keyboard.is_key_pressed(sf.Keyboard.L_SHIFT)):
                        pynk.lib.nk_input_char(self.ctx, pynk.ffi.cast("char",attr.code + 65))
                    else:
                        pynk.lib.nk_input_char(self.ctx, pynk.ffi.cast("char",attr.code + 97))
    #ui command queue
    def draw(self,window):
        c= pynk.lib.nk__begin(self.ctx)
        while c:
            cType=c.type
            if cType==pynk.lib.NK_COMMAND_RECT_FILLED:
                self.nuklearSfmlDrawRectFilled(c, window)
            elif cType==pynk.lib.NK_COMMAND_TEXT:
                self.nuklearSfmlDrawText(c, window)
            elif cType==pynk.lib.NK_COMMAND_SCISSOR:
                self.nuklearSfmlDrawScissor(c, window)
            elif cType==pynk.lib.NK_COMMAND_RECT:
                self.nuklearSfmlDrawRectOutline(c, window)
            elif cType==pynk.lib.NK_COMMAND_CIRCLE_FILLED:
                self.nuklearSfmlDrawCircleFilled(c, window)
            elif cType==pynk.lib.NK_COMMAND_TRIANGLE_FILLED:
                self.nuklearSfmlDrawTriangleFilled(c, window)
            elif cType==pynk.lib.NK_COMMAND_LINE:
                self.nuklearSfmlDrawLine(c, window)
            else:
                print("fatal error",cType)
            c=pynk.lib.nk__next(self.ctx,c)
        pynk.lib.nk_clear(self.ctx)

