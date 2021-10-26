
from tqdm import tqdm
from src.data_handling.bcdr_dataset import BCDRDataset, BCDRSegmentationDataset

def load_dataset(path, info_csv, outlines_csv=None):

    for idx, (p, info, outlines) in enumerate(zip(path, info_csv, outlines_csv)):
        # Load all dataset data
        if info == 'None':
            df = BCDRSegmentationDataset(outlines_csv=outlines, dataset_path=p)
        else:
            if outlines == 'None':
                outlines = None
            df = BCDRDataset(info_file=info, outlines_csv=outlines, dataset_path=p)
        # df.print_distribution()
        with tqdm(total=len(df)) as pbar:
            # Iterate trought the dictionary
            for case in df:
                scan_img_path = case['scan']
                mask_path = case['mask']
                classification = case['classification']
                # TODO Convert TIF to NIFTI

if __name__ == '__main__':

    # Change the paths to your dataset folder
    path =  ['/home/lidia/Datasets/BCDR/BCDR-D01_dataset',
                '/home/lidia/Datasets/BCDR/BCDR-D02_dataset',
                '/home/lidia/Datasets/BCDR/BCDR-F01_dataset',
                '/home/lidia/Datasets/BCDR/BCDR-F02_dataset',
                '/home/lidia/Datasets/BCDR/BCDR-F03_dataset/BCDR-F03'
                ]
    info_csv = ['/home/lidia/Datasets/BCDR/BCDR-D01_dataset/bcdr_d01_img.csv',
                '/home/lidia/Datasets/BCDR/BCDR-D02_dataset/bcdr_d02_img.csv',
                '/home/lidia/Datasets/BCDR/BCDR-F01_dataset/bcdr_f01_img.csv',
                '/home/lidia/Datasets/BCDR/BCDR-F02_dataset/bcdr_f02_img.csv',
                'None'
                ]
    outlines_csv =  ['/home/lidia/Datasets/BCDR/BCDR-D01_dataset/bcdr_d01_outlines.csv',
                    '/home/lidia/Datasets/BCDR/BCDR-D02_dataset/bcdr_d02_outlines.csv',
                    '/home/lidia/Datasets/BCDR/BCDR-F01_dataset/bcdr_f01_outlines.csv',
                    '/home/lidia/Datasets/BCDR/BCDR-F02_dataset/bcdr_f02_outlines.csv',
                    '/home/lidia/Datasets/BCDR/BCDR-F03_dataset/BCDR-F03/bcdr_f03_outlines.csv'
                    ]
    
    # Load and process all BCDR datasets
    load_dataset(path, info_csv, outlines_csv=outlines_csv)