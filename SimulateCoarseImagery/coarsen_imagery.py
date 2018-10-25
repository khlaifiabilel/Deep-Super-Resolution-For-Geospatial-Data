import numpy as np
import gdal
import os
import datetime
import glob
import sys
import cv2
from tqdm import tqdm
from scipy import misc

### Code to simulate coarser imagery
#python3 coarsen_imagery.py /input/dir/ /output/dir/ GausianSigma ScalingFactor
#python3 coarsen_imagery.py /input/dir/ /output/dir/ 1 2
#python3 coarsen_imagery.py /input/dir/ /output/dir/ 4 8
#python3 coarsen_imagery.py /input/dir/ /output/dir/ 2.5 5

def image_blur(input_directory,output_directory,sigma,scale):
    os.chdir(input_directory)
    images = glob.glob('*.tif')
    driver = gdal.GetDriverByName("GTiff")
    outputs=[]

    for stackclip in tqdm(images):
        sigtext=str(sigma)
        sigtext=sigtext.split(".")
        OUTPUT=output_directory+stackclip
        outputs.append(OUTPUT)
        interp = gdal.Open(os.path.abspath(stackclip))
        geo=interp.GetGeoTransform()
        interpproj=interp.GetProjection()
        Array = interp.ReadAsArray()
        out=np.swapaxes(Array,0,2)

        if sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigma)
            out = cv2.resize(out, (int(out.shape[1]/scale),int(out.shape[0]/scale)), interpolation=cv2.INTER_AREA)
        else:
            out = cv2.resize(out, (int(out.shape[1]/scale),int(out.shape[0]/scale)), interpolation=cv2.INTER_AREA)
        out=np.swapaxes(out,2,0)
        #print(out.shape)
        scale1=float(Array.shape[2]/out.shape[2])
        scale2=float(Array.shape[1]/out.shape[1])
        pixW=float(geo[1])*scale1
        pixH=float(geo[5])*scale2
        geo=[geo[0],pixW,geo[2],geo[3],geo[4],pixH]

        
        diagmap_out = driver.Create( OUTPUT, out.shape[2], out.shape[1], out.shape[0], gdal.GDT_Byte)
        diagmap_out.SetGeoTransform( geo )
        diagmap_out.SetProjection( interpproj )
        for i, image in enumerate(out, 1):
            diagmap_out.GetRasterBand(i).WriteArray( image )
        del diagmap_out
        
    print( "Done")
    return outputs

    
    
if __name__ == "__main__":
    image_blur(sys.argv[1],sys.argv[2],float(sys.argv[3]),int(sys.argv[4]))