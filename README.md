# hls_lab_finn
Experiments for FINN flow. <br/>

This experiment achieve end-to-end NN deployment on a normal laptop, i.e., from NN training, Bitstream generation, to FPGA deployment, all can be done on a laptop. <br/>

* System requirement 
  * CUDA compatible computer: Nvidia graphic cards 
    * install CUDA and CUDNN follow official instruction 
    * to check if CUDA is set correctly:
```shell
$ nvidia-smi          # use this command in shell to see following print-out related to detected GPU
Sun Nov 28 14:28:21 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 496.13       Driver Version: 496.13       CUDA Version: 11.5     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   50C    P8    13W /  N/A |    164MiB /  6144MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
  * Python
    * Python can be installed via pip or conda. 
    * install python's torch library. To see if CUDA is accessible from it: 
```python
# in Python shell 
import torch
torch.cuda.is_available()
```
  * docker is used to setup environment 
    * allow non-root to run docker. Detailed setting can be found in docker's official webpage. 
    * https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user 
  * Install Xilinx 2020.1 

