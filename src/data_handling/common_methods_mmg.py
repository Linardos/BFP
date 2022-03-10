from src.data_handling.common_classes_mmg import *
import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))
        
def save_images_paths(dataset, txt_out_file, pathologies = None):
    # Save image list in txt
    image_paths = []
    for client in dataset:
        if pathologies:
            images = client.get_images_by_pathology(pathologies)
            for image in images:
                image_paths.append(image.path)
        else:
            for study in client:
                for image in study:
                    image_paths.append(image.path)

    with open(txt_out_file, 'w') as f:
        f.write("\n".join(image_paths))

def save_images_ids(dataset, txt_out_file, pathologies = None):
    # Save image list in txt
    image_ids = []
    for client in dataset:
        if pathologies:
            images = client.get_images_by_pathology(pathologies)
            for image in images:
                image_ids.append(str(image.id))
        else:
            for study in client:
                for image in study:
                    image_ids.append(str(image.id))

    with open(txt_out_file, 'w') as f:
        f.write("\n".join(image_ids))
