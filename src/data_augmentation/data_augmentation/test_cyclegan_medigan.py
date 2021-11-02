from src.data_augmentation.CycleGAN_high_density import *

if __name__ == '__main__':
    #TODO IMPORTANT Set the package name in CycleGAN_high_density/src/models/__init__.py
    # conda activate /home/lidia/source/BreastCancer/envs_torch
    # Generate zip file 

    # Test single input/output image
    # input_image_path = './src/data_augmentation/CycleGAN_high_density/src/images/sample_input_image.png'
    # output_path = './src/data_augmentation/medigan_cyclegan_sample_output.png'
    # Test input/output folder
    input_path = './src/data_augmentation/CycleGAN_high_density/images'
    output_path = './src/data_augmentation'

    num_samples = 1
    model_file = './src/data_augmentation/CycleGAN_high_density/CycleGAN_high_density.pth'
    #model_file = './src/data_augmentation/CycleGAN_high_density/latest_net_G_B.pth'
    gpu_id = 0 # Set as argument
    image_size = (1332, 800) # Set as argument
    save_images = False

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    model_file = os.path.abspath(model_file)
    generate_GAN_images(model_file, image_size, num_samples, output_path, gpu_id, 
                        input_path, save_images)

## Citation
# If you use this code for your research, please cite our papers.
# ```
# @inproceedings{CycleGAN2017,
#   title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
#   author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
#   booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
#   year={2017}
# }


# @inproceedings{isola2017image,
#   title={Image-to-Image Translation with Conditional Adversarial Networks},
#   author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
#   booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
#   year={2017}
# }
# ```