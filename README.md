# RAFT
This repository is forked from the source code of the following paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>

## Requirements
The environment can be installed using conda by running
```Shell
conda env create -f environment.yaml
conda activate raft
```

## Demos
You can demo a trained model on a given frame by running
```Shell
bash inference.sh
```

## Dataset
By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets are in the `datasets` folder

```Shell
├── datasets
    ├── <dataset name>
        ├── <frame_1>
        ├── <frame_2>
        |── <flow>
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
