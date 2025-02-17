import subprocess
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="arguments for generating object masks")
    parser.add_argument("--datasetname", required=True)

    args = parser.parse_args()

    # dirs = os.listdir('./data/' + args.datasetname + '/spectra/')
    os.chdir('./Tracking-Anything-with-DEVA')
    print("Current working directory: {0}".format(os.getcwd()))
    # for dir in dirs:
    #     spectra_dir =  '../data/' + args.datasetname + '/spectra/' + dir
    #     try:
    #         subprocess.run("python demo/demo_automatic.py --chunk_size 4 --img_path " + spectra_dir + " --amp --temporal_setting semionline --size -1 --output \"./example/output_gaussian_dataset/" +args.datasetname + "/"+ dir + "\" --suppress_small_objects --SAM_PRED_IOU_THRESHOLD 0.7 --use_short_id", shell=True, check=True)
    #         subprocess.run("cp -r ./example/output_gaussian_dataset/" +args.datasetname + "/"+ dir + "/Annotations "+ spectra_dir + "/object_mask", shell=True)
    #     except Exception as e:
    #         print(e)
    
    image_dir = '../data/' + args.datasetname + '/images/'
    subprocess.run("python demo/demo_automatic.py --chunk_size 4 --img_path " + image_dir + " --amp --temporal_setting semionline --size -1 --output \"./example/output_gaussian_dataset/" +args.datasetname + "\" --SAM_PRED_IOU_THRESHOLD 0.1 --use_short_id", shell=True, check=True)
    subprocess.run("cp -r ./example/output_gaussian_dataset/" +args.datasetname + "/Annotations "+ '../data/' + args.datasetname + "/object_mask", shell=True)
    

