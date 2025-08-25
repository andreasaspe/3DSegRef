import os

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    os.environ['nnUNet_raw'] = "/home/awias/data/nnUNet/nnUNet_raw"
    os.environ['nnUNet_preprocessed'] = "/home/awias/data/nnUNet/nnUNet_preprocessed"
    os.environ['nnUNet_results'] = "/home/awias/data/nnUNet/nnUNet_results"

    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    
    # os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
    # multiprocessing.set_start_method("spawn")
    
    print("Hello??????????????????????????")
    
    get_train_loader()
