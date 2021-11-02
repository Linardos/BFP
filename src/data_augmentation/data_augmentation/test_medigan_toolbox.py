# Create new python/conda environment
# conda activate /home/lidia/source/BreastCancer/medigan
# pip install -e /home/lidia/source/medigan
# Create a zip file and add a new entry to global.json
# Generate a UUID as model name
# TODO: Explain sections in global.json
# Install the missing dependencies of the selected model manually
# Call generate method selecting the correct model_id and input arguments
# TODO: Add Troubleshooting or common errors section / FAQs ?

from medigan import Generators

generators = Generators()
#generators.generate(model_id="2d29d505-9fb7-4c4d-b81f-47976e2c7dbf",num_samples=10)
#generators.generate(model_id="8f933c5e-72fc-461a-a5cb-73cbe65af6fc",num_samples=10)

#Input arguments
#model_id: cycleGAN model UUID (bf4857b8-d411-45f7-abcf-7a7d8e26bedd)
#input_path: Path to input images folder or a single filename if given an image as input
#output_path: Path to output folder or filename if given a single image as input
#image_size: output images size
#gpu_id: GPU ID to generate images, set to -1 to use CPU instead
#num_samples: not used
#save_images: if save images is False, the image numpy array or a list of images is returned

generators.generate(model_id="bf4857b8-d411-45f7-abcf-7a7d8e26bedd",
                    #input_image_path='./src/data_augmentation/CycleGAN_high_density/src/images/sample_input_image.png',
                    #output_path="./src/data_augmentation/test_toolbox.png"
                    #image_size=(1332, 800),
                    #gpu_id=-1,
                    #save_images=True
                    )
