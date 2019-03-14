import data
import predict
import numpy as np
import tensorflow as tf
import os
import sys
import gdal
from tqdm import tqdm

#python3 Create_SR_NoGEO.py "input/data/" "/output/data/" 2

def SR_it(input_dir,output_dir,scaling_factor):
    base_dir=os.getcwd()
    file_names = []
    SF=scaling_factor
    if input_dir.endswith("/"):
        O=input_dir.split("/")[-2]
    else:
        O=input_dir.split("/")[-1]
    with tf.Session() as session:
        network = predict.load_model(session)

        driver = gdal.GetDriverByName("GTiff")
        os.chdir(input_dir)
        os.chdir(base_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for file_name in tqdm(os.listdir(input_dir)):
                    file_names.append(file_name)

        for set_name in [O]:
            for scaling_factor in [SF]:
                dataset = data.SR_Run(set_name, scaling_factors=[scaling_factor])
                for I, file_name in tqdm(zip(dataset.images,file_names)):
                    Im=[I]
                    prediction = predict.predict(Im, session, network, targets=None, border=scaling_factor)
                    prediction=prediction[0]
                    prediction=np.swapaxes(prediction,-1,0)
                    prediction=np.swapaxes(prediction,-1,1)
                    out=output_dir+str(file_name)
                    DataSet = driver.Create(out, prediction.shape[2], prediction.shape[1], prediction.shape[0], gdal.GDT_Byte)
                    for i, image in enumerate(prediction, 1):
                        DataSet.GetRasterBand(i).WriteArray( image )
                    del DataSet

if __name__ == "__main__":
    SR_it(sys.argv[1],sys.argv[2],int(sys.argv[3]))
