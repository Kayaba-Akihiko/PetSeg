# PetSeg


## Test Environment:  
Ubuntu 20.04 server  
Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz  
RTX A6000 48GB  
RAM 256GB  
docker: akihikokayaba/petseg

python 3.10  
pytorch 1.11+cu11.3  
opencv  
scipy  
pillow 

## Train
````bash
python train.py \
--gpu_id 0 \
--batch_size 400 \
--n_epoch 200 \
--preload_dataset True
````
Around 6 hours for finishing training.
## Training monitor
## Training loss
![alt text](figs/Train_Loss.png)
## Test DC
![alt text](figs/Test_DC.png)
## Test ASSD
![alt text](figs/Test_ASSD.png)
## Intermedia result of epoch 1, 2, and 3.
![alt text](figs/intermedia_result.png)
## Model Structure
![alt text](figs/structure.png)
## Model weights
![alt text](figs/weights.png)

## Evaluate
````bash
python eval_and_visual.py \
--gpu_id 0 \
--batch_size 20 \
--pretrain_loading_epoch 40
````

## Result
![alt text](figs/boxplot.png)
![alt text](figs/samples.png)