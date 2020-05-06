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
class ifs_gpu(object):
    def __init__(self,ifs_general):
        self.ifs_general=ifs_general
        self.initialize_kernel()
    def render_to_file(self):
        self.iterations=self.ifs_general.iterationsInMld*1000000000
        self.render_gpu()
    def render_gpu(self):

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
	    #####cudaMemset(&pass3_d, 0, 209715200 * sizeof(float));

        for i in range(0,self.ifs_general.affineNum):
            affineMatrix[i]=sf.Transform.from_values(self.ifs_general.affineTransforms[i][0],self.ifs_general.affineTransforms[i][1],self.ifs_general.affineTransforms[i][2],self.ifs_general.affineTransforms[i][3],self.ifs_general.affineTransforms[i][4],
                                         self.ifs_general.affineTransforms[i][5],0,0,1)####
            postMatrix[i]=sf.Transform.from_values(self.ifs_general.postTransforms[i][0],self.ifs_general.postTransforms[i][1],self.ifs_general.postTransforms[i][2],self.ifs_general.postTransforms[i][3],self.ifs_general.postTransforms[i][4],
                                       self.ifs_general.postTransforms[i][5],0,0,1)###

            #data feed
            pass2_h[6*i+6] = self.ifs_general.affineTransforms[i][0]
            pass2_h[6*i+1+6] = self.ifs_general.affineTransforms[i][1]
            pass2_h[6*i+2+6] = self.ifs_general.affineTransforms[i][2]
            pass2_h[6*i+3+6] = self.ifs_general.affineTransforms[i][3]
            pass2_h[6*i+4+6] = self.ifs_general.affineTransforms[i][4]
            pass2_h[6*i+5+6] = self.ifs_general.affineTransforms[i][5]

            pass2_h[42+i] = self.ifs_general.affineTransforms[i][6];

		    #post transform feed
            pass2_h[6 * i + 50] = self.ifs_general.postTransforms[i][0]
            pass2_h[6 * i + 1 + 50] = self.ifs_general.postTransforms[i][1]
            pass2_h[6 * i + 2 + 50] = self.ifs_general.postTransforms[i][2]
            pass2_h[6 * i + 3 + 50] = self.ifs_general.postTransforms[i][3]
            pass2_h[6 * i + 4 + 50] = self.ifs_general.postTransforms[i][4]
            pass2_h[6 * i + 5 + 50] = self.ifs_general.postTransforms[i][5]

        #data feed
        pass2_h[0] = self.ifs_general.affineTransforms[0][7];
        pass2_h[1] = pass2_h[0] + self.ifs_general.affineTransforms[1][7];
        pass2_h[2] = pass2_h[1] + self.ifs_general.affineTransforms[2][7];
        pass2_h[3] = pass2_h[2] + self.ifs_general.affineTransforms[3][7];
        pass2_h[4] = pass2_h[3] + self.ifs_general.affineTransforms[4][7];
        pass2_h[5] = pass2_h[4] + self.ifs_general.affineTransforms[5][7];
        
        for i in range(6):
            pass1_h[i] = self.ifs_general.current[i];

        pass2_h[48] = self.ifs_general.zoom
        pass2_h[49] = self.ifs_general.boundedFunctionScale
        

        if (self.ifs_general.iterationsInMld < 10):
            blocks = self.iterations // 1024000;
            loops = 1;
        else:
            blocks = 2500;
            loops = double(self.iterations) / 2560000000+0.7;
        
        for i in range(loops):
            pass1_h[6] = random.random()*1000;
            #cudaMemcpy(&pass1_d[6], &pass1_h[6],  sizeof(int), cudaMemcpyHostToDevice);
            cuda.memcpy_htod(pass2_d, pass2_h)
            cuda.memcpy_htod(pass1_d,pass1_h)
            #########cudaAcceleratedHistogram << <blocks, 1024 >> > (pass1_d, pass2_d, pass3_d);
            #cudaDeviceSynchronize();
            self.cudaAcceleratedHistogram(pass1_d, pass2_d, pass3_d,block = (1024, 1, 1), grid=(blocks, 1))
            print("end")
        #cudaMalloc(&image_d, 8388608 * sizeof(float));
        image_d=cuda.mem_alloc(8388608*sys.getsizeof(float()))
	    ############cudaAcceleratedSupersampling << <4096, 1024 >> > (pass3_d, image_d,&pass1_d[7]);
        cuda.memcpy_dtoh(pass1_h,pass1_d)
        print("max",pass1_h[7])
        max_d=cuda.mem_alloc(4)
        cuda.memcpy_htod(max_d,pass1_h[7])
        self.cudaAcceleratedSupersampling(pass3_d, image_d,max_d,block = (1024, 1, 1), grid=(4096, 1))
        print("end")
        #cudaMalloc(&gauss_d, sizeof(int));
        gauss_d=cuda.mem_alloc(sys.getsizeof(int()))
        #cudaMemcpy(gauss_d, &gauss, sizeof(int), cudaMemcpyHostToDevice);
        cuda.memcpy_htod(gauss_d, np.array(self.ifs_general.gauss))
        #############cudaAcceleratedGaussianBlur << <4096, 1024 >> > (image_d, gauss_d);
        self.cudaAcceleratedGaussianBlur(image_d, gauss_d,block = (1024, 1, 1), grid=(4096, 1))
        print("end")
        #float* image_h = (float*)malloc(8388608 * sizeof(float));
        image_h=np.zeros(8388608)
        image_h=image_h.astype(np.float32)
        #cudaMemcpy(image_h, image_d, 8388608 * sizeof(float), cudaMemcpyDeviceToHost);
        cuda.memcpy_dtoh(image_h, image_d)


        img=sf.Image.create(2048, 2048)
        alpha=0.0
        for i in range(2048):
            for y in range(2048):
                if image_h[y * 2048 + i]>1:
                    image_h[y * 2048 + i]=1
                alpha = image_h[y * 2048 + i];
                col=sf.Color(alpha * 255, alpha * 255, alpha * 255, 255);
                img[(y, i)]= col

            img.to_file("ifs.png")

    def initialize_kernel(self):
        if (os.system("cl.exe")):
            os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")
        self.gpu_kernel=SourceModule(no_extern_c=True,source="""
		#include "curand_kernel.h"
        //CUDA
		//CUDA math functions
		inline __host__ __device__ float2 operator*(float2 a, float2 b)
		{
			return make_float2(a.x * b.x, a.y * b.y);
		}
		inline __host__ __device__ void operator*=(float2 &a, float2 b)
		{
			a.x *= b.x;
			a.y *= b.y;
		}
		inline __host__ __device__ float2 operator*(float2 a, float b)
		{
			return make_float2(a.x * b, a.y * b);
		}
		inline __host__ __device__ float2 operator*(float b, float2 a)
		{
			return make_float2(b * a.x, b * a.y);
		}
		inline __host__ __device__ void operator*=(float2 &a, float b)
		{
			a.x *= b;
			a.y *= b;
		}
		inline __host__ __device__ void mulMatVec(float *data,int index,float2* point)//multiply 3x2 matrix by vector
		{
			float temp;
	
			temp = data[index] * point->x + data[index + 1] * point->y + data[index + 2];
			point->y= data[index+3] * point->x + data[index + 4] * point->y + data[index + 5];
			point->x = temp;
	
		}
		inline __device__ float calculateKernel_d(float x, float sigma)
		{
			return 0.39894*expf(-0.5*x*x / (sigma*sigma)) / sigma;
		}
		//CUDA variations functions
		__device__ float boundedFunctionScale_d = 1;
		inline __device__ void spherical(float2 &point, curandState& s) {
			point = 1 / (powf(point.x, 2) + powf(point.y, 2))*(point);

		}
		inline __device__ void swirl(float2 &point, curandState& s) {
			float r2 = powf(point.x, 2) + powf(point.y, 2);
			point = make_float2(point.x*sinf(r2) - point.y*cosf(r2), point.x*cosf(r2) + point.y*sinf(r2));
		}
		inline __device__ void handkerchief(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(r*sinf(theta + r), r*cosf(theta - r));
		}
		inline __device__ void sinusoidal(float2 &point, curandState& s) {
			point = make_float2(sinf(point.x)*boundedFunctionScale_d, sinf(point.y)*boundedFunctionScale_d);
		}
		inline __device__ void linear(float2 &point, curandState& s) {

		}
		inline __device__ void horseshoe(float2 &point, curandState& s) {
			float r = sqrtf(pow(point.x, 2) + powf(point.y, 2));
			point = make_float2(1 / r*(point.x - point.y)*(point.x + point.y), 1 / r * 2 * point.x*point.y);
		}
		inline __device__ void polar(float2 &point, curandState& s) {
			float r = sqrtf(pow(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(theta / 3.1415*boundedFunctionScale_d, (r - 1)*boundedFunctionScale_d);

		}
		inline __device__ void disc(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(theta / 3.1415*sinf(3.1415*r)*boundedFunctionScale_d, theta / 3.1415*cosf(3.1415*r)*boundedFunctionScale_d);

		}
		inline __device__ void heart(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(r*sinf(theta*r), r*(-cosf(theta*r)));
		}
		inline __device__ void spiral(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(1 / r*(cosf(theta) + sinf(r)), 1 / r*(sinf(theta) - cosf(r)));
		}
		inline __device__ void hyperbolic(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(sinf(theta) / r, r*cosf(theta));
		}
		inline __device__ void diamond(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			point = make_float2(sinf(theta)*cosf(r)*boundedFunctionScale_d, cosf(theta)*sinf(r)*boundedFunctionScale_d);
		}
		__device__ void ex(float2 &point, curandState& s) {
			float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
			float theta = atan2f(point.x, point.y);
			float p0 = sinf(theta + r);
			float p1 = cosf(theta - r);
			point = make_float2(r*(powf(p0, 3) + powf(p1, 3)), r*(powf(p0, 3) - powf(p1, 3)));
		}
		__device__ void julia(float2 &point, curandState& s) {

			float omega = (int)(curand_uniform(&s)+0.5)* 3.1415;
			float sqrtr = sqrtf(sqrtf(powf(point.x, 2) + powf(point.y, 2)));
			float theta = atan2f(point.x, point.y);
			point = make_float2(sqrtr*cosf(theta / 2 + omega)*boundedFunctionScale_d, sqrtr*sinf(theta / 2 + omega)*boundedFunctionScale_d);

		}
		__device__ void(*deviceFunctionArray[14])(float2&, curandState&) = { spherical,swirl,handkerchief,sinusoidal,linear,horseshoe,polar,disc,heart,spiral,hyperbolic,diamond,ex,julia };//function pointers for device

		//CUDA main algorithm
		//pass1 0-5 current
		//pass1 6 offset for random generator
		//pass1 7 max

		//pass2 0-5 probability
		//pass2 6-11 mat1
		//pass2 12-17 mat2
		//pass2 18-23 mat3
		//pass2 24-29 mat4
		//pass2 30-35 mat5
		//pass2 36-41 mat6
		//pass2 42-47 color
		//pass2 48 zoom
		//pass2 49 boundedFunctionScale
		//pass2 50+ postTransforms

		//pass3 0-104 857 599 histogram count
		//pass3 104 857 600-209 715 199 histogram color

		extern "C" __global__ void cudaAcceleratedHistogram(int* pass1,float* pass2,float* pass3) {
			boundedFunctionScale_d = pass2[49];
			curandState s;
			curand_init(blockIdx.x * 1024 + threadIdx.x+pass1[6], 0, 0, &s);

			float2 point = make_float2(0, 0);
			int temp, temp2;
			float cad = 0;
			for (int i = 0; i <1000; i++)
			{

				float r = curand_uniform(&s);
		
		

				if (r <= pass2[0]) { mulMatVec(pass2,6,&point); cad = (cad + pass2[42]) / 2; deviceFunctionArray[pass1[0]](point,s); mulMatVec(pass2, 50, &point);}
				else if (r <= pass2[1]) { mulMatVec(pass2, 12, &point); cad = (cad + pass2[43]) / 2; deviceFunctionArray[pass1[1]](point,s); mulMatVec(pass2, 56, &point);}
				else if (r <=pass2[2]) { mulMatVec(pass2, 18, &point); cad = (cad + pass2[44]) / 2; deviceFunctionArray[pass1[2]](point,s); mulMatVec(pass2, 62, &point);}
				else if (r <=pass2[3]) { mulMatVec(pass2, 24, &point) ; cad = (cad + pass2[45]) / 2; deviceFunctionArray[pass1[3]](point,s); mulMatVec(pass2, 68, &point);}
				else if (r <=pass2[4]) { mulMatVec(pass2, 30, &point); cad = (cad + pass2[46]) / 2; deviceFunctionArray[pass1[4]](point,s); mulMatVec(pass2, 74, &point);}
				else { mulMatVec(pass2, 36, &point); cad = (cad + pass2[47]) / 2; deviceFunctionArray[pass1[5]](point,s); mulMatVec(pass2, 80, &point);}
		
		
		
		
				if (i > 20) {
					if (abs(point.x) >= (5 * pass2[48])-0.001 || abs(point.y) >= (5 * pass2[48])-0.001)//GPU rounding makes exacly comp useless, need some space
						continue;
					temp = (int)((point.x + 5 * pass2[48]) * (1024 / pass2[48]));
					temp2 = (int)((point.y + 5 * pass2[48]) * (1024 / pass2[48]));
					temp2 = temp * 10240 + temp2;
					pass3[temp2]= (int)(pass3[temp2]+1);
					if (pass3[temp2] > pass1[7]){
						atomicExch(&pass1[7],pass3[temp2]);}
					pass3[104857600 + temp2] = (pass3[104857600 + temp2] + cad) / 2;

				}
			}
			
		}

		extern "C" __global__ void cudaAcceleratedSupersampling(float* histogram, float* image,int* max) {

			float freq = 0;
			float color = 0;
			float alpha = 0;
			int y = blockIdx.x % 2*1024 + threadIdx.x;
			int i = blockIdx.x / 2;
					for (int a = 0; a < 5; a++) {
						for (int b = 0; b < 5; b++) {
							freq = histogram[5 * y * 10240 + b * 10240 + (5 * i + a)];
							color = histogram[5 * y * 10240 + b * 10240 + (5 * i + a) + 104857600];
							if (freq != 0)
								alpha += color*powf((log10f(freq) / log10f(*max)), 1 / 2.2);
							image[y * 2048 + i+ 4194304] += histogram[5 * y * 10240 + b * 10240 + (5 * i + a)];
						}
					}
					alpha /= 25;

					image[y * 2048 + i] = alpha;

		
		}

		extern "C" __global__ void cudaAcceleratedGaussianBlur(float *image,int *gauss) {
			int x = blockIdx.x % 2 * 1024 + threadIdx.x;
			int y = blockIdx.x / 2;
			float kernel[20];

					int kernelWidth = *gauss / powf(image[x*2048+y+ 4194304], 0.4);
					if (kernelWidth >9|| kernelWidth <0)
						kernelWidth = *gauss;
					if (kernelWidth == 0)
						return;

					int kernelSize = kernelWidth * 2 + 1;
					float sigma = 6.;
					float Z = 0.0;
					float final_colour = 0;

					for (int j = 0; j <= kernelWidth; ++j)
					{
						kernel[kernelWidth + j] = kernel[kernelWidth - j] = calculateKernel_d(float(j), sigma);
					}

					for (int j = 0; j < kernelSize; ++j)
					{
						Z += kernel[j];
					}


					for (int i = -kernelWidth; i <= kernelWidth; ++i)
					{
						for (int j = -kernelWidth; j <= kernelWidth; ++j)
						{
							if (x + i < 0 || y + j < 0 || x + 1>2047 || y + j>2047)
								continue;
							final_colour += kernel[kernelWidth + j] * kernel[kernelWidth + i] * image[x*2048 + i*2048+y + j];

						}
					}
					final_colour /= (Z*Z);
					image[x * 2048 + y] = final_colour;
			
		}
		""")
        self.cudaAcceleratedHistogram = self.gpu_kernel.get_function("cudaAcceleratedHistogram")
        self.cudaAcceleratedSupersampling=self.gpu_kernel.get_function("cudaAcceleratedSupersampling")
        self.cudaAcceleratedGaussianBlur=self.gpu_kernel.get_function("cudaAcceleratedGaussianBlur")