# ConvL2LFlow
A convolutional-flow-based generative machine learning model for calorimeter shower simulation.

# Table of Contents
1. [Dependencies](#Dependencies)
3. [Run Code](#Run-Code)
2. [Configuration File](#Configuration-File)

## Dependencies
* [pytorch](https://pytorch.org) - for the neural network
* [nflows](https://github.com/bayesiains/nflows) - for flow transformations
* [numpy](https://numpy.org) - for some numeric calculations
* [scipy](https://www.scipy.org) - to calculate wasserstein distances
* [matplotlib](https://matplotlib.org) - for plotting
* [h5py](https://github.com/h5py/h5py) - reading/writing .h5 files
* [pyyaml](https://github.com/yaml/pyyaml) - reading/writing .yaml files
* [torchinfo](https://github.com/TylerYep/torchinfo) - to summarize torch models
* [torchmetrics](https://github.com/Lightning-AI/torchmetrics) - for the evaluation of classifier results
* [scikit-learn](https://scikit-learn.org) - to calibrate classifier 

## Run Code
You can use dataset 3 of the [Calo Challenges](https://calochallenge.github.io/homepage) to train the networks. Note that the code was tested on a computer with 512 GB CPU RAM and 40 GB GPU RAM and might need adjustments ot run on machines with less memory. 

First set up the environment using pip or conda.

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate convL2LFlows
```
**Or** using pip:
```bash
# ensure to use python3.10
# consider using a venv (https://docs.python.org/3/library/venv.html)
pip install -r requirements.txt
```

To download the data set and start the training run the following.
```bash
# download data sets
mkdir data
for i in {1..4}
do
    echo "get dataset_3_$i.hdf5"
    wget https://zenodo.org/record/6366324/files/dataset_3_$i.hdf5\?download\=1 -O data/dataset_3_$i.hdf5
done

# prepare data sets
python scripts/convert_challenge.py --shape 45 50 18 data/dataset_3_1.hdf5 data/dataset_3_2.hdf5 data/dataset_3.h5
python scripts/convert_challenge.py --shape 45 50 18 data/dataset_3_3.hdf5 data/dataset_3_4.hdf5 data/dataset_3_test.h5
python scripts/calc_layer_e.py -t 1.515e-5 data/dataset_3.h5
python scripts/calc_layer_e.py -t 1.515e-5 data/dataset_3_test.h5

# generate result directories
python mkresultdir.py conf/energy.yaml
python mkresultdir.py -f 0-44 conf/causal.yaml
```
There should be a folder named results containing two sub folders one for the energy distribution flow and one for the causal flows. Both contain a ```run.sh``` file whish you can use to start the training. After the training has finished you can generate new samples using following command (you have to adapt the names of the result folders):
```bash
python src/generator.py -n 100000 -b 10000 results/???_energy results/???_shower --log --veto 2.6
``` 
If you want to test the results you can run (you have to adapt the names of the result folders):
```bash
# train low level classifier
python src/classifier.py -c 1.515e-5 -m CConv -g data/dataset_3_test.h5 results/???_shower/samples00.h5
# train high level classifier
python src/high_level_classi.py -c 1.515e-5 -g data/dataset_3_test.h5 results/???_shower/samples00.h5
# calculate mean and std over the wasserstein distances between generated and test data
# the wasserstein distances are calculated 10 times using 10 equal sized splits of the data
# output: name, mean, std
python src/wasserstein.py -c 1.515e-5 -g data/dataset_3_test.h5 results/???_shower/samples00.h5
``` 
If you want to run the [calo challenge evaluation script](https://github.com/CaloChallenge/homepage/tree/main/code) to produce some plots you have to convert the data format. (you have to adapt the names of the result folders)
```bash
python scripts/convert_challenge.py -i -n 2 --t 1.515e-5 results/???_shower/samples00.h5 data/dataset_3_conv_flow_{}.hdf5
```
If you have cloned the calo challenge evaluation script you can run it from within the code folder using something like:
```bash
python evaluate.py --input_file data/dataset_3_conv_flow_1.hdf5 --input_file_2 data/dataset_3_conv_flow_2.hdf5 --reference_file data/dataset_3_3.hdf5 --reference_file_2 data/dataset_3_4.hdf5 --dataset 3 --mode all --save_mem
```

## Configuration File
This is a list of the parameters that can be specified in yaml parameter files ```conf/*.yaml```. Many have default values, such that not all parameters have to be specified.

### run parameters
Parameter               | Type    | Description
----------------------- | ------- | ------------------------------------------------------------
run\_name               | string  | name for the output folder

### dataset parameters
Parameter               | Type    | Description
----------------------- | ------- | --------------------------------------------------------------------
class                   | string  | "ShowerDataset" or "EnergyDataset"
data\_file              | string  | path of the .h5-file containing the training data
noise\_mode             | string  | noise added to all entries ["uniform", "gaussian", "log_normal", null]
noise\_mean             | float   | mean of the noise distribution (for "log_normal" it should be given in log10 space)
noise\_std              | float   | option only valid for "gaussian" and "log_normal"
extra\_noise\_mode      | string  | only for "ShowerDataset" noise to fill zero voxels with
extra\_noise\_mean      | float   | equivalent to noise\_mean
extra\_noise\_std       | float   | equivalent to noise\_std
samples\_trafo          | list    | preprocessing for sample, will be inverted during generation
cond\_trafo             | list    | preprocessing for incident energy
cond2\_trafo            | list    | preprocessing for layer energy, only for "ShowerDataset"
device                  | string  | where to store the data during training (if not the device you train on, data will be moved bach wise)
random\_shift           | boolean | enable data augmentation by random shift

### trainer parameters
Parameter               | Type    | Description
----------------------- | ------- | --------------------------------------------------------------------
learning\_rate          | float   | learning rate
scheduler               | string  | type of LR scheduling: "Step", "Exponential" or "OneCycle"
weight\_decay           | float   | L2 weight decay
num\_epochs             | integer | number of training epochs
grad\_clip              | float   | if given, a L2 gradient clip with the given value is applied

### dataloader parameters
Parameter               | Type    | Description
----------------------- | ------- | --------------------------------------------------------------------
batch\_size             | integer | batch size
pin\_memory             | boolean | use memory pinning (only possible if dataset.device is cpu)

### flow parameters
Parameter               | Type    | Description
----------------------- | ------- | --------------------------------------------------------------------
class                   | string  | "ConvFlow" or "MAFlow"
num\_blocks             | integer | number of coupling or MADE blocks 
num\_layers             | integer | number of layers in each MADE block (only for "MAFlow")
dropout                 | float   | dropout fraction for the sub-networks (only for "MAFlow")
num\_bins               | integer | number of spline bins
coupling\_block         | string  | "additive", "affine", (piecewise) "linear", (piecewise) "quadratic", (piecewise) "cubic" or (piecewise) "rational_quadratic" (only for "ConvFlow")
use\_act\_norm          | boolean | use activation norm (only for "ConvFlow")
use\_one\_by\_ones      | boolean | replace permutations by GLOW 1x1 convolutions (only for "ConvFlow")
squeeze                 | integer | squeeze factor (only for "ConvFlow")
subnet\_args            | dict    | arguments for the sub-networks, see U-Net parameters (only for "ConvFlow")

### U-Net parameters
Parameter               | Type    | Description
----------------------- | ------- | --------------------------------------------------------------------
hidden                  | integer | hidden layer size
cyclic_padding          | boolean | use cyclic instead of zero padding in last dimension
downsamples             | list    | list containing the kernel sizes of all down sample operations inside the U-Net
identity_init           | boolean | initialism the last layer to output zeros always

---
For questions/comments about the code contact: thorsten.buss@uni-hamburg.de

This code was written for the paper:

**Convolutional L2LFlows: Generating Accurate Showers in Highly Granular Calorimeters Using Convolutional Normalizing Flows**<br/>
[https://arxiv.org/abs/2405.20407](https://arxiv.org/abs/2405.20407)<br/>
*Thorsten Buss, Frank Gaede, Gregor Kasieczka, Claudius Krause, David Shih*
