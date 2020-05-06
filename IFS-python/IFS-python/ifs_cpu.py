import sfml.sf as sf
import random
import math
from ifs_general import ifs_general
import numpy as np

class ifs_cpu():

    def __init__(self,ifs_general):
        self.ifs_general=ifs_general
    def render_to_file(self):
        self.iterations=self.ifs_general.iterationsInMld*1000000000
        self.render_one_core()###
            
    def gaussian_density_blur(self,freqArchive,img):
        kernel=np.zeros(20)
        for y in range(400):
            for x in range(400):
                divider= pow(freqArchive[x][y], 0.4)
                if divider!=0:
                    kernelWidth=int(self.ifs_general.gauss / pow(freqArchive[x][y], 0.4))
                    #print(kernelWidth)
                else:
                    continue
                if (kernelWidth < 0):
                    kernelWidth = self.ifs_general.gauss
                if (kernelWidth == 0):
                    continue;

                kernelSize = kernelWidth * 2 + 1;
                #float* kernel = new float[kernelSize + 1];
                sigma = 6.0
                Z = 0.0
                final_colour = 0
                for j in range(kernelWidth+1):
                    #print(kernelWidth + j)
                    kernel[kernelWidth + j] = self.calculateKernel(float(j), sigma)
                    kernel[kernelWidth - j] = kernel[kernelWidth + j]
                    #print(kernel[kernelWidth - j])

                for j in range(kernelSize):
                    Z += kernel[j]

                for i in range(-kernelWidth,kernelWidth+1):
                    for j in range(-kernelWidth,kernelWidth+1):
                        if (x + i < 0 or y + j < 0 or x + i>400 or y + j>400):
                            continue
                        final_colour += kernel[kernelWidth + j] * kernel[kernelWidth + i] * img[(x + i, y + j)].r
                        #print(final_colour)
                final_colour /= (Z*Z);
                f=sf.Color(final_colour, final_colour, final_colour, 255);
                img[(x, y)]=f;

    def post_process_one_core(self,histogram,maxv):
        img=sf.Image.create(400, 400)
        #img.create(2048, 2048);
        freq = 0;
        color = 0;
        alpha = 0;
        freqArchive = np.zeros((400,400))
        for i in range(400):
            for y in range(400):
                #freq = histogram[y][i][0];
                #color = histogram[y][i][1];
                for a in range(5):
                    for b in range(5):
                        freq = histogram[5 * y + b][5 * i + a][0];
                        color = histogram[5 * y + b][5 * i + a][1];
                        if (freq != 0 and maxv>1):
                            alpha += color*pow((math.log(freq) / math.log(maxv)), 1 / 2.2);
                            #print(alpha)
                        freqArchive[y][i] += histogram[5 * y + b][5 * i + a][0]#histogram[y][i][0]


                alpha /= 25;

                col=sf.Color(alpha * 255, alpha * 255, alpha * 255, 255);
                img[(y, i)]= col
        self.gaussian_density_blur(freqArchive,img)###
        img.to_file("ifs.png")###



    def render_one_core(self):
        if self.ifs_general.unlinearNum==0:
            for i in range(1,6):
                self.ifs_general.current[i]=self.ifs_general.current[0]
        affineMatrix=[None for i in range(6)]
        postMatrix=[None for i in range(6)]
        for i in range(0,self.ifs_general.affineNum):
            affineMatrix[i]=sf.Transform.from_values(self.ifs_general.affineTransforms[i][0],self.ifs_general.affineTransforms[i][1],self.ifs_general.affineTransforms[i][2],self.ifs_general.affineTransforms[i][3],self.ifs_general.affineTransforms[i][4],
                                         self.ifs_general.affineTransforms[i][5],0,0,1)####
            postMatrix[i]=sf.Transform.from_values(self.ifs_general.postTransforms[i][0],self.ifs_general.postTransforms[i][1],self.ifs_general.postTransforms[i][2],self.ifs_general.postTransforms[i][3],self.ifs_general.postTransforms[i][4],
                                       self.ifs_general.postTransforms[i][5],0,0,1)###
            
        #array of unlinear transformations
        unlinearPointer=[self.spherical,self.swirl,self.handkerchief,self.sinusoidal,self.linear,self.horseshoe,self.polar,self.disc,self.heart,self.spiral,self.hyperbolic,self.diamond,self.ex,self.julia]
        
        maxv=0
        point=sf.Vector2(0,0)
        cad=0
        histogram=np.zeros((2001,2001,2))#[[[0 for i in range(2)] for k in range(10241)] for j in range(10241)]
        
        probability0=self.ifs_general.affineTransforms[0][7]
        probability1 = probability0 + self.ifs_general.affineTransforms[1][7]
        probability2 = probability1 + self.ifs_general.affineTransforms[2][7]
        probability3 = probability2 + self.ifs_general.affineTransforms[3][7]
        probability4 = probability3 + self.ifs_general.affineTransforms[4][7]
        probability5 = probability4 + self.ifs_general.affineTransforms[5][7]

        #Didn't use function calls because of overwhelming iterations number
        for i in range(20000000):
            #print(i)
            result=random.random()

            
            if result<probability0:
                point=affineMatrix[0].transform_point(point)
                cad=(cad+self.ifs_general.affineTransforms[0][6])/2
                point=unlinearPointer[self.ifs_general.current[0]](point);
                point=postMatrix[0].transform_point(point)
                #print(point)
            elif result<probability1:
                 point=affineMatrix[1].transform_point(point)
                 cad = (cad + self.ifs_general.affineTransforms[1][6]) / 2 
                 point=unlinearPointer[self.ifs_general.current[1]](point) 
                 point=postMatrix[1].transform_point(point)
            elif result<probability2:
                point=affineMatrix[2].transform_point(point)
                cad = (cad + self.ifs_general.affineTransforms[2][6]) / 2 
                point=unlinearPointer[self.ifs_general.current[2]](point)
                point=postMatrix[2].transform_point(point)
            elif result<probability3:
                point=affineMatrix[3].transform_point(point)
                cad = (cad + self.ifs_general.affineTransforms[3][6]) / 2 
                point=unlinearPointer[self.ifs_general.current[3]](point)
                point=postMatrix[3].transform_point(point)
            elif result<probability4:
                point=affineMatrix[4].transform_point(point)
                cad = (cad + self.ifs_general.affineTransforms[4][6]) / 2
                point=unlinearPointer[self.ifs_general.current[4]](point)
                point=postMatrix[4].transform_point(point)
            elif result<probability5:
                point=affineMatrix[5].transform_point(point)
                cad = (cad + self.ifs_general.affineTransforms[5][6]) / 2
                point=unlinearPointer[self.ifs_general.current[5]](point)
                point=postMatrix[5].transform_point(point)
                
                #first positions may be too far but transforms are strongly contractive so just drop first 20
            if i>20:
                if (abs(point.x) > 5 * self.ifs_general.zoom or abs(point.y) > 5 * self.ifs_general.zoom):
                    continue;
            temp =int((point.x + 5 * self.ifs_general.zoom) * (200 / self.ifs_general.zoom))
            temp2 =int((point.y + 5 * self.ifs_general.zoom) * (200 / self.ifs_general.zoom))
            if(temp<0 or temp2<0 or temp>2000 or temp2>2000):
                continue
            histogram[temp][temp2][0] += 1;
            #print(temp,temp2)
            if (histogram[temp][temp2][0] > maxv):
                maxv = histogram[temp][temp2][0]
            histogram[temp][temp2][1] = (histogram[temp][temp2][1] + cad) / 2
            #print("this",histogram[temp][temp2][0])

        self.post_process_one_core(histogram,maxv)

    def mix(self,x, y, a):
        return x*(1 - a) + y*a;

    def clamp(self,x, minVal, maxVal):
        return min(max(x, minVal), maxVal);
    
    def determinant(self,mat):
        matPointer = mat.matrix();
        return abs(matPointer[0] * matPointer[5] * matPointer[10] - matPointer[4] * matPointer[1] * matPointer[10]);

    def div(self,mat, x):###
        mat = sf.Transform(mat.matrix()[0] / x, mat.matrix()[4] / x, mat.matrix()[12], mat.matrix()[1] / x, mat.matrix()[5] / x, mat.matrix()[13], mat.matrix()[2], mat.matrix()[6], mat.matrix()[10]);

    #gaussian
    def calculateKernel(self,x, sigma):
        return 0.39894*math.exp(-0.5*x*x / (sigma*sigma)) / sigma;
    
    #Unlinear functions
    def spherical(self,point):
        mul=1 / (pow(point.x, 2) + pow(point.y, 2))
        return sf.Vector2(mul*point.x,mul*point.y);

    def swirl(self,point):
        r2 = pow(point.x, 2) + pow(point.y, 2);
        return sf.Vector2 (point.x*math.sin(r2) - point.y*math.cos(r2), point.x*math.cos(r2) + point.y*math.sin(r2));
 
    def handkerchief(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return sf.Vector2 (r*math.sin(theta + r), r*math.cos(theta - r));

    def sinusoidal(self,point):
        return sf.Vector2 (math.sin(point.x)*self.ifs_general.boundedFunctionScale, math.sin(point.y)*self.ifs_general.boundedFunctionScale);
 
    def linear(self,point):
        pass

    def horseshoe(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        return  sf.Vector2 (1 / r*(point.x - point.y)*(point.x + point.y), 1 / r * 2 * point.x*point.y);
    
    def polar(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (theta / 3.1415*self.ifs_general.boundedFunctionScale, (r - 1)*self.ifs_general.boundedFunctionScale);

    def disc(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (theta / 3.1415*math.sin(3.1415*r)*self.ifs_general.boundedFunctionScale, theta / 3.1415*math.cos(3.1415*r)*self.ifs_general.boundedFunctionScale);

    def heart(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (r*math.sin(theta*r), r*(-math.cos(theta*r)));

    def spiral(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (1 / r*(math.cos(theta) + math.sin(r)), 1 / r*(math.sin(theta) - math.cos(r)));

    def hyperbolic(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (math.sin(theta) / r, r*math.cos(theta));

    def diamond(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (math.sin(theta)*math.cos(r)*self.ifs_general.boundedFunctionScale, math.cos(theta)*math.sin(r)*self.ifs_general.boundedFunctionScale);

    def ex(self,point):
        r = math.sqrt(pow(point.x, 2) + pow(point.y, 2));
        theta = math.atan2(point.x, point.y);
        p0 = math.sin(theta + r);
        p1 = math.cos(theta - r);
        return  sf.Vector2 (r*(pow(p0, 3) + pow(p1, 3)), r*(pow(p0, 3) - pow(p1, 3)));

    def julia(self,point):

        omega = random.random()* 3.1415;
        sqrtr = math.sqrt(math.sqrt(pow(point.x, 2) + pow(point.y, 2)));
        theta = math.atan2(point.x, point.y);
        return  sf.Vector2 (sqrtr*math.cos(theta / 2 + omega)*self.ifs_general.boundedFunctionScale, sqrtr*math.sin(theta / 2 + omega)*self.ifs_general.boundedFunctionScale);

