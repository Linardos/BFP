
import os

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__ == '__main__':

    config_file = '/home/lidia/source/BreastCancer/src/configs/preprocess_optimam.yaml'
    run_sripts= "PYTHONPATH=${PYTHONPATH}:./:./../../ python /home/lidia/source/BreastCancer/src/preprocessing/optimam_dataset.py "

    #dirname = '/mnt/sdc1/OPTIMAM/image_db/sharing/omi-db/images'
    dirname = '/home/lidia/Datasets/OPTIMAM/omi-db/data'
    dirfiles = os.listdir(dirname)
    print(f'Loading {len(dirfiles)} clients:')
    first_run = 1
    chunks = 1000
    for counter, group in enumerate(chunker(dirfiles, chunks)):
        print(f'Processing clients from {counter*chunks} to {(counter+1)*chunks} ...')
        os.system(run_sripts + config_file + " " + str(chunks) + " " + str(first_run) + " " + str(counter))
        first_run = 0
    
    print("All clients finished!")