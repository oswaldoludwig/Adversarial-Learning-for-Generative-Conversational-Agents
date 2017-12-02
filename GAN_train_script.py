__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import os

Epochs = 15

print('Transferring the weights from the pre-trained baseline bot to the GAN model...')
file = 'python transfer_weights_baseline_GAN.py'
os.system(file)

print('Padding the baseline data...')
file = 'python padding_data_baseline.py'
os.system(file)

for i in range (Epochs):
    
    print('Producing machine-generated data (epoch #%d)...'%(i + 1))
    file = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python simulate_bot_GAN.py'
    os.system(file)
        
    print('Padding the generated data (epoch #%d)...'%(i + 1))
    file = 'python padding_data_GAN.py'
    os.system(file)
    
    print('Training the discriminator with the generated examples...')
    file = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python train_discriminator_GAN.py'
    os.system(file)
    
    print('GAN training of the bot for one epoch...')
    file = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python train_bot_GAN.py'
    os.system(file)
    
    print('Transferring the weights from the GAN trained model to the baseline bot...')
    file = 'python transfer_weights_GAN_baseline.py'
    os.system(file)
    
    print('Training the bot for one epoch using simple teacher forcing and the human-generated training data...')
    file = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python train_bot_baseline.py'
    os.system(file)

    print('Transferring the weights from the baseline bot back to the GAN trained one...')
    file = 'python transfer_weights_baseline_GAN.py'
    os.system(file)

print('Training the bot once again for one epoch using teacher forcing and the human-generated training data...')
file = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python train_bot_baseline.py'
os.system(file)
