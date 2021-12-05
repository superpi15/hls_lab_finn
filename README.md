# hls_lab_finn
Experiments for [FINN flow](https://github.com/Xilinx/finn). <br/>

This experiment achieves end-to-end NN deployment on a normal laptop, i.e., from NN training, Bitstream generation, to FPGA deployment, all can be done on a laptop. <br/>

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
  * Setup [FINN](https://finn.readthedocs.io/en/latest/getting_started.html)

  * Source [init.sh](init.sh) in your terminal. Its content is shown as follows. Please modify it according to your Xilinx installation and device ip. 
```shell
source /tools/Xilinx/Vitis/2020.1/settings64.sh 
export FINN_XILINX_PATH=/tools/Xilinx/
export FINN_XILINX_VERSION=2020.1
export VIVADO_PATH=/tools/Xilinx/Vivado/2020.1
export PYNQ_BOARD=Pynq-Z2
export FINN_HOST_BUILD_DIR=/home/${USER}/build
export PYNQ_IP=<YOUR_FPGA_IP>
```

## Part-1 

The end-to-end flow is based on FINN tutorial. The original jupyter-notebook of the flow is located at the path: "finn/notebooks/end2end_example/bnn-pynq/cnv_end2end_example.ipynb". <br/>
We provide a improved version of it. Please put the file ["finn/notebooks/end2end_example/bnn-pynq/cnv_end2end_example_new.ipynb"](finn/notebooks/end2end_example/bnn-pynq/cnv_end2end_example_new.ipynb) to the corresponding FINN path. 
The improvement resolve the driver issue based on the suggestion of [Pynq-driver-issue](https://github.com/Xilinx/finn/discussions/442#discussioncomment-1675720). <br/><br/>
Specifically, we add the following code to the deployment part of the notebook. 

```python
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver

model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_synth.onnx")
model = model.transform(MakePYNQDriver(platform="zynq-iodma"))
model.save(build_dir + "/end2end_cnv_w1a1_synth-driver.onnx")
```

The execution result :

![](part1/image/Screenshot%202021-11-27%20215332.png)

The images of each intermediate ONNX can be found here: https://github.com/superpi15/hls_lab_finn/tree/main/part1/image 


## Part-2

We do part 2 based on the offered brevitas training code and the jupyter-notebook located at "finn/notebooks/end2end_example/bnn-pynq/cnv_end2end_example.ipynb". <br/>
Some modifications are needed, including those in "LabD_FINN.pdf" and those posted in the discussion on ntu cool.  <br/>
Moreover, we have to modify the padding of the convolution layers when we implement the model architecture assigned by part 2, which is shown in the following figure: <br/>
![vgg9Archi](figures/vgg9_modelArchitecture.png)
In the original traing code, the convolution layers have padding being 0, which will lead the input size of the third pooling layer to be 1x1 and thus the pooling cannot be performed. <br/>
To solve this, we modified the padding of all the convolution layers to 1. <br/>
When doing so, the input size of the first fully connected layer will be 256x4x4, and we have to set the number of input features of the first FC layer accordingly. <br/>

When deploying on FPGA, the number of PE and SIMD have to be decreased compared with the original values in cnv_end2end_example.ipynb so that the hardware resources needed can fit in the PYZQ-Z2 FPGA board. <br/>
It may due to the fact that adjusting the padding make the model much bigger than the original model. <br/>
We set them as follows:
```python
# each tuple is (PE, SIMD, in_fifo_depth) for a layer
folding = [
    (8, 3, 128),
    (8, 16, 128),
    (4, 16, 128),
    (4, 16, 128),
    (2, 16, 81),
    (2, 16, 81),
    (1, 16, 2),
    (1, 4, 2),
    (1, 8, 128),
    (2, 1, 3),
]
```

The testing accuracy on software is shown below:
![accSW](part2/accSW.png)

The testing accuracy on hardware is shown below:
![accHW](part2/accHW_.png)

We can observe that there is a gap between the two accuracy values (50% v.s. 35.66%). <br/>
So far, we are still trying to figure out what causes it. <br/>
A few items we've investigated: 
* Try different PE and SIMD combinations for different layers does not help. 
* Deploy to other FPGA (Ultra96v2) does not help. Thus it is less likely to be compatibility issue between FINN and FPGAs. 
* Train CNV_1W1A for Cifar100. There is no accuracy loss. Thus the problem could be now narrow down to VGG9. 

## Part-3 

In this part we seek to improve the throughput and runtime of a given fully-connected neural network when compiling it with FINN by adjusting the folding parameter, PE and SIMD. The dataset we use in this design is MNIST, which means the input tensor size is 784 and the output tensor size is 10. The network architecture is given in the lab description as below.

<img src="part3/part3.png" alt="part3" width="50%"/>

The original PE and SIMD parameters is given in below in the FINN example notebook.
Layer 1: PE=16, SIMD=49
Layer 2: PE=8,  SIMD=8
Layer 3: PE=8, SIMD=8
Layer 4: PE=10, SIMD=8


Here is the screenshot of the metrics with the default PE and SIMD.

<img src="part3/metrics_default.png" alt="part3default" width="50%"/>

We are asked to change the PE and SIMD of layer 2 to 1 of the network. The result is attached below. We can see that runtime and throughput were severely degraded by these changes. We think that the reason is because we make the second layer be the bottleneck of this design. In the original setting, the bottleneck is the first layer, which has II = 784/49 * 512/16 = 512, where as we have II = 512/1 * 64/1 = 32768. When we divide the new II by the old II and multiply it with the original runtime, it’s quite close to the new runtime. 

<img src="part3/metrics_Layer2_exp.png" alt="part3l2exp" width="50%"/>

