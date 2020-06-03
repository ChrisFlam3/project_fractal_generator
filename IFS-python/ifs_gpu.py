import sfml.sf as sf
import random
import math
from ifs_general import ifs_general
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os
import sys

#class for flam3 execution and data postprocessing using gpu
class ifs_gpu(object):
    def __init__(self,ifs_general):
        self.ifs_general=ifs_general
        self.initialize_kernel()
    def render_to_file(self):
        self.iterations=self.ifs_general.iterationsInMld*1000000000
        self.render_gpu()

    def render_gpu(self):
		#data load
        pass1_h=np.zeros(8)
        pass1_h = pass1_h.astype(np.int32)
        pass2_h=np.zeros(86)
        pass2_h = pass2_h.astype(np.float32)

        if self.ifs_general.unlinearNum==0:
            for i in range(1,6):
                self.ifs_general.current[i]=self.ifs_general.current[0]
        affineMatrix=[None for i in range(6)]
        postMatrix=[None for i in range(6)]

        #cudaMalloc(&pass1_d, 8 * sizeof(int));
        pass1_d=cuda.mem_alloc(8*4)
	    #cudaMalloc(&pass2_d, 87 * sizeof(float));
        pass2_d=cuda.mem_alloc(87*4)
	    #cudaMalloc(&pass3_d, 209715200 * sizeof(float));
        pass3_d=cuda.mem_alloc(209715200*4)
	    #cudaMemset(&pass3_d, 0, 209715200 * sizeof(float));

        for i in range(0,self.ifs_general.affineNum):
            affineMatrix[i]=sf.Transform.from_values(self.ifs_general.affineTransforms[i][0],self.ifs_general.affineTransforms[i][1],self.ifs_general.affineTransforms[i][2],self.ifs_general.affineTransforms[i][3],self.ifs_general.affineTransforms[i][4],
                                         self.ifs_general.affineTransforms[i][5],0,0,1)

            postMatrix[i]=sf.Transform.from_values(self.ifs_general.postTransforms[i][0],self.ifs_general.postTransforms[i][1],self.ifs_general.postTransforms[i][2],self.ifs_general.postTransforms[i][3],self.ifs_general.postTransforms[i][4],
                                       self.ifs_general.postTransforms[i][5],0,0,1)

            for y in range(6):
                #data feed
                pass2_h[6*i+y+6] = self.ifs_general.affineTransforms[i][y]
                #post transform feed
                pass2_h[6 * i + y + 50] = self.ifs_general.postTransforms[i][y]
                #data feed
                pass2_h[y] = self.ifs_general.affineTransforms[y][7]

            pass2_h[42+i] = self.ifs_general.affineTransforms[i][6]
        
        for i in range(6):
            pass1_h[i] = self.ifs_general.current[i];

        pass2_h[48] = self.ifs_general.zoom
        pass2_h[49] = self.ifs_general.boundedFunctionScale
        
		#gpu layout and data transmission
        if (self.ifs_general.iterationsInMld < 10):
            blocks = self.iterations // 1024000;
            loops = 1;
        else:
            blocks = 2500;
            loops = double(self.iterations) / 2560000000+0.7
        
        for i in range(loops):
            pass1_h[6] = random.random()*1000;
            #cudaMemcpy(&pass1_d[6], &pass1_h[6],  sizeof(int), cudaMemcpyHostToDevice);
            cuda.memcpy_htod(pass2_d, pass2_h)
            cuda.memcpy_htod(pass1_d,pass1_h)
            #cudaAcceleratedHistogram << <blocks, 1024 >> > (pass1_d, pass2_d, pass3_d);
            #cudaDeviceSynchronize();
            self.cudaAcceleratedHistogram(pass1_d, pass2_d, pass3_d,block = (1024, 1, 1), grid=(blocks, 1))
            
        #cudaMalloc(&image_d, 8388608 * sizeof(float));
        image_d=cuda.mem_alloc(8388608*sys.getsizeof(float()))
	    #cudaAcceleratedSupersampling << <4096, 1024 >> > (pass3_d, image_d,&pass1_d[7]);
        cuda.memcpy_dtoh(pass1_h,pass1_d)
        print("max",pass1_h[7])
        max_d=cuda.mem_alloc(4)
        cuda.memcpy_htod(max_d,pass1_h[7])
        self.cudaAcceleratedSupersampling(pass3_d, image_d,max_d,block = (1024, 1, 1), grid=(4096, 1))
        
        #cudaMalloc(&gauss_d, sizeof(int));
        gauss_d=cuda.mem_alloc(sys.getsizeof(int()))
        #cudaMemcpy(gauss_d, &gauss, sizeof(int), cudaMemcpyHostToDevice);
        cuda.memcpy_htod(gauss_d, np.array(self.ifs_general.gauss))
        #cudaAcceleratedGaussianBlur << <4096, 1024 >> > (image_d, gauss_d);
        self.cudaAcceleratedGaussianBlur(image_d, gauss_d,block = (1024, 1, 1), grid=(4096, 1))
        
        #float* image_h = (float*)malloc(8388608 * sizeof(float));
        image_h=np.zeros(8388608)
        image_h=image_h.astype(np.float32)
        #cudaMemcpy(image_h, image_d, 8388608 * sizeof(float), cudaMemcpyDeviceToHost);
        cuda.memcpy_dtoh(image_h, image_d)

		#save results
        img=sf.Image.create(2048, 2048)
        alpha=0.0
        for i in range(2048):
            for y in range(2048):
                if image_h[y * 2048 + i]>1:
                    image_h[y * 2048 + i]=1
                alpha = image_h[y * 2048 + i]
                col=sf.Color(alpha * 255, alpha * 255, alpha * 255, 255)
                img[(y, i)]= col

            img.to_file("ifs.png")

	#pycuda
	#split to other file
    def initialize_kernel(self):
        if (os.system("cl.exe")):
            os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")
        k=open("kernel.c","r")
        self.gpu_kernel=SourceModule(no_extern_c=True,source=k.read())
        self.cudaAcceleratedHistogram = self.gpu_kernel.get_function("cudaAcceleratedHistogram")
        self.cudaAcceleratedSupersampling=self.gpu_kernel.get_function("cudaAcceleratedSupersampling")
        self.cudaAcceleratedGaussianBlur=self.gpu_kernel.get_function("cudaAcceleratedGaussianBlur")