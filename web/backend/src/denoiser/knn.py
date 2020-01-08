# https://github.com/AlainPaillou/PyCuda_Denoise_Filters/blob/master/PyCuda_KNN_Denoise_Colour.py

import time
import cv2
import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

class KNN_Denoiser(object):
    KNN_Colour_GPU = None

    # Set blocks et Grid sizes
    nb_ThreadsX = 8
    nb_ThreadsY = 8
    width = 2520
    height = 1080

    nb_blocksX = 0
    nb_blocksY = 0

    # Set KNN parameters
    KNN_Noise = 0.32
    Noise = 1.0/(KNN_Noise*KNN_Noise)
    lerpC = 0.2

    mod = SourceModule("""
    __global__ void KNN_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
    unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
    {
        
        #define KNN_WINDOW_RADIUS   3
        #define NLM_BLOCK_RADIUS    3
        #define KNN_WEIGHT_THRESHOLD    0.00078125f
        #define KNN_LERP_THRESHOLD      0.79f
        const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
        const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
        
        const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
        const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
        const float  x = (float)ix  + 1.0f;
        const float  y = (float)iy  + 1.0f;
        const float limxmin = NLM_BLOCK_RADIUS + 2;
        const float limxmax = imageW - NLM_BLOCK_RADIUS - 2;
        const float limymin = NLM_BLOCK_RADIUS + 2;
        const float limymax = imageH - NLM_BLOCK_RADIUS - 2;
    
        
        long int index4;
        long int index5;
        if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
            //Normalized counter for the weight threshold
            float fCount = 0;
            //Total sum of pixel weights
            float sumWeights = 0;
            //Result accumulator
            float3 clr = {0, 0, 0};
            float3 clr00 = {0, 0, 0};
            float3 clrIJ = {0, 0, 0};
            //Center of the KNN window
            index4 = x + (y * imageW);
            index5 = imageW * (iy + 1) + ix + 1;
            
            clr00.x = img_r[index4];
            clr00.y = img_g[index4];
            clr00.z = img_b[index4];
        
            for(float i = -NLM_BLOCK_RADIUS; i <= NLM_BLOCK_RADIUS; i++)
                for(float j = -NLM_BLOCK_RADIUS; j <= NLM_BLOCK_RADIUS; j++) {
                    long int index2 = x + j + (y + i) * imageW;
                    clrIJ.x = img_r[index2];
                    clrIJ.y = img_g[index2];
                    clrIJ.z = img_b[index2];
                    float distanceIJ = ((clrIJ.x - clr00.x) * (clrIJ.x - clr00.x)
                    + (clrIJ.y - clr00.y) * (clrIJ.y - clr00.y)
                    + (clrIJ.z - clr00.z) * (clrIJ.z - clr00.z)) / 65536.0;
                    //Derive final weight from color and geometric distance
                    float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;
                    clr.x += clrIJ.x * weightIJ;
                    clr.y += clrIJ.y * weightIJ;
                    clr.z += clrIJ.z * weightIJ;
                    //Sum of weights for color normalization to [0..1] range
                    sumWeights     += weightIJ;
                    //Update weight counter, if KNN weight for current window texel
                    //exceeds the weight threshold
                    fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
            }
            
            //Normalize result color by sum of weights
            sumWeights = 0.0039f / sumWeights;
            clr.x *= sumWeights;
            clr.y *= sumWeights;
            clr.z *= sumWeights;
            //Choose LERP quotient basing on how many texels
            //within the KNN window exceeded the weight threshold
            float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
            
            clr.x = clr.x + (clr00.x / 256.0 - clr.x) * lerpQ;
            clr.y = clr.y + (clr00.y / 256.0 - clr.y) * lerpQ;
            clr.z = clr.z + (clr00.z / 256.0 - clr.z) * lerpQ;
            
            dest_r[index5] = (int)(clr.x * 256.0);
            dest_g[index5] = (int)(clr.y * 256.0);
            dest_b[index5] = (int)(clr.z * 256.0);
        }
    }
    """)

    def __init__(self, width, height):
        self.KNN_Colour_GPU = self.mod.get_function("KNN_Colour")

        self.width = width
        self.height = height
        
        self.nb_blocksX = (self.width // self.nb_ThreadsX) + 1
        self.nb_blocksY = (self.height // self.nb_ThreadsY) + 1

    def __del__(self):
        return


    def denoiseFrame(self, frame):
        # Algorithm GPU using PyCuda
        print("Test GPU ",self.nb_blocksX*self.nb_blocksY," Blocks ",self.nb_ThreadsX*self.nb_ThreadsY," Threads/Block")
        tps1 = time.time()


        In_File = "test.jpg"

        frame = cv2.imread(In_File,cv2.IMREAD_COLOR)

        b,g,r = cv2.split(frame)
        print(b.size)
        print(b.dtype.itemsize)
        b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
        img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
        drv.memcpy_htod(b_gpu, b)
        drv.memcpy_htod(img_b_gpu, b)
        res_b = np.empty_like(b)
                    
        g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
        drv.memcpy_htod(g_gpu, g)
        img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
        drv.memcpy_htod(img_g_gpu, g)
        res_g = np.empty_like(g)
                    
        r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
        drv.memcpy_htod(r_gpu, r)
        img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
        drv.memcpy_htod(img_r_gpu, r)
        res_r = np.empty_like(r)

        self.KNN_Colour_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(self.width),np.intc(self.height), np.float32(self.Noise), \
                np.float32(self.lerpC), block=(self.nb_ThreadsX,self.nb_ThreadsY,1), grid=(self.nb_blocksX,self.nb_blocksY))

        drv.memcpy_dtoh(res_r, r_gpu)
        drv.memcpy_dtoh(res_g, g_gpu)
        drv.memcpy_dtoh(res_b, b_gpu)
        r_gpu.free()
        g_gpu.free()
        b_gpu.free()
        img_r_gpu.free()
        img_g_gpu.free()
        img_b_gpu.free()

        image_brut_GPU=cv2.merge((res_r,res_g,res_b))

        tps_GPU = time.time() - tps1

        print("PyCuda treatment OK")
        print ("GPU treatment time : ",tps_GPU)
        print("")

        return image_brut_GPU