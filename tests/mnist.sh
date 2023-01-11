#!bin/bash

python3 ../main.py --log_name 'mnist_training_CNN' \
    --model 'CNN' \
    --datasets 'MNIST' \
    --batch_size 128 \
    --test_batch_size 1000 \
    --epochs 10 \
    --lr 0.01 \
    --momentum 0.9 \
    --inversion_lr 0.1 \
    --inversion_momentum 0.9 \
    --save_model \
    --model_save_name 'mnist_cnn.pt' \
    --model_loaded \
    --k 3 \
    --nn_index 10