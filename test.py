import data
import predict2
import numpy as np
import tensorflow as tf


with tf.Session() as session:
    network = predict2.load_model(session)

    for set_name in ['val_images_544']:
        for scaling_factor in [2,4,8]:
            dataset = data.TestSet(set_name, scaling_factors=[scaling_factor])
            predictions, psnr, ssim = predict2.predict(dataset.images, session, network, targets=dataset.targets,
                                                border=int(scaling_factor))

            print('Dataset "%s", scaling factor = %d. Mean PSNR = %.2f.' % (set_name, scaling_factor, np.mean(psnr)))
            print("SSIM:",np.mean(ssim))
