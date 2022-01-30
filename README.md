# Lw-GCN
The test code of Lw-GCN

## Prerequisite

 - PyTorch 1.7.1
 - Cuda 11.0
 - g++ 5.4.0

## Compile cuda extensions

  ```
  cd ./model/Temporal_shift
  bash run.sh
  ```

## Data Preparation

 - Download the raw data of [NTU-RGBD](https://github.com/shahroudy/NTURGB-D) and [NTU-RGBD120](https://github.com/shahroudy/NTURGB-D). Put NTU-RGBD data under the directory `./data/nturgbd_raw`. Put NTU-RGBD120 data under the directory `./data/nturgbd120_raw`. 
 
 - For NTU-RGBD, preprocess data with `python data_gen/ntu_gendata.py`. For NTU-RGBD120, preprocess data with `python data_gen/ntu120_gendata.py`. 
  
 - Generate the bone data with `python data_gen/gen_bone_data.py`.

 - Generate the motion data with `python data_gen/gen_motion_data.py`.

## Testing

To test the model, refer to `./evaluations/README.md`
  