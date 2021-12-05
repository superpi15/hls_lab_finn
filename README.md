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

This section go through the FINN flow. It takes as input the weight files '\*.pth', 'best.tar', or sample weights, and generate a deployable onnx package with a sequence of transformation. The transformation includes (1) adding labels for users to observe the network (2) adding device information for deployment (3) [streamlining](https://arxiv.org/pdf/1709.04060.pdf) maps floating operations into integer operations. 

The execution result :

<img src="part1/image/Screenshot%202021-11-27%20215332.png" alt="screen" width="100%"/>

The images of each intermediate ONNX can be found here: https://github.com/superpi15/hls_lab_finn/tree/main/part1/image 


## Part-2

We do part 2 based on the offered brevitas training code and the jupyter-notebook located at "finn/notebooks/end2end_example/bnn-pynq/cnv_end2end_example.ipynb". <br/>
Some modifications are needed, including those in "LabD_FINN.pdf" and those posted in the discussion on ntu cool.  <br/>
Moreover, we have to modify the padding of the convolution layers when we implement the model architecture assigned by part 2, which is shown in the following figure: <br/>
<img src="figures/vgg9_modelArchitecture.png" alt="vgg9Archi" width="100%"/>
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
<img src="part2/accSW.png" alt="accSW" width="100%"/>

The testing accuracy on hardware is shown below:
<img src="part2/accHW_.png" alt="accHW" width="100%"/>

We can observe that there is a gap between the two accuracy values (50% v.s. 35.66%). <br/>
So far, we are still trying to figure out what causes it. <br/>
A few items we've investigated: 
* Try different PE and SIMD combinations for different layers does not help. 
* Deploy to other FPGA (Ultra96v2) does not help. Thus it is less likely to be compatibility issue between FINN and FPGAs. 
* Train CNV_1W1A for Cifar100. There is no accuracy loss. Thus the problem could be now narrow down to VGG9. 

## Part-3 

In this part we seek to improve the throughput and runtime of a given fully-connected neural network when compiling it with FINN by adjusting the folding parameter, PE and SIMD. The dataset we use in this design is MNIST, which means the input tensor size is 784 and the output tensor size is 10. The network architecture is given in the lab description as below.

<img src="part3/part3.png" alt="part3" width="100%"/>

We are asked to achieve at least 90% accuracy on the deployed network, and the result is 96.77% when running `validate.py` on pynq. We attach the screenshot of accuracy as below.

<img src="part3/part3_accuracy_on_pynq.png" alt="part3acc" width="50%"/>

The original PE and SIMD parameters is given in below in the FINN example notebook.
Layer 1: PE=16, SIMD=49
Layer 2: PE=8,  SIMD=8
Layer 3: PE=8, SIMD=8
Layer 4: PE=10, SIMD=8


Here is the screenshot of the metrics with the default PE and SIMD.

<img src="part3/metrics_default.png" alt="part3default" width="80%"/>

We are asked to change the PE and SIMD of layer 2 to 1 of the network in exp1.
The result is attached below. We can see that runtime and throughput were severely degraded by these changes.
We think that the reason is because by making these changes, we make the second layer be the bottleneck of this design. 
In the original setting, the bottleneck is the first layer, which has `II = 784/49 * 512/16 = 512`, where as we have `II = 512/1 * 64/1 = 32768` now.
When we divide the new II by the old II and multiply it with the original runtime, it’s quite close to the new runtime. 

<img src="part3/metrics_exp1.png" alt="part3exp1" width="80%"/>

Then in exp2, we need to optimize the design by adjusting PE and SIMD in each layers. Below we give the three setting of PE and SIMD of each layers in a table.

| Setting | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
| ----- | --- | --- | --- | --- |
| 1  | PE = 32; SIMD = 49;  | PE = 8; SIMD = 16; | PE = 8; SIMD = 8; | PE = 10; SIMD = 8; |
| 2  | PE = 32; SIMD = 98;  | PE = 16; SIMD = 16; | PE = 8; SIMD = 8; | PE = 10; SIMD = 8; |
| 3  | PE = 16; SIMD = 49;  | PE = 8; SIMD = 8; | PE = 8; SIMD = 8; | PE = 10; SIMD = 8; |

In this design, the bottleneck is the first two layer with II = 512. In order to improve the throughput of the design, we should adjust the PE and SIMD of both layer simultaneously.
In Setting 1 and 2, we try to achieve a maximum II with 256 and 128. By seeing the metric of the synthesized network, it is quite clear that we manage is speed up the runtime of the
design by roughly 2 and 4 times faster than the default runtime. Unfortunately, FINN cannot synthesis with larger PE and SIMD due to the resource limitation. So we turn to prove that 
adjusting only one layer is not going to affect the runtime in setting 3. The result does meet our expectation, with runtime and throughput are roughly the same as the original setting.
We attach the metrics of three setting as below.

- Setting 1
<img src="part3/metrics_opt_setting1.png" alt="partopt1" width="80%"/>

- Setting 2
<img src="part3/metrics_opt_setting2.png" alt="partopt2" width="80%"/>

- Setting 3
<img src="part3/metrics_opt_setting3.png" alt="part3opt3" width="80%"/>

