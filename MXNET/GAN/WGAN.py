# Modified from example/autograd/{data,dcgan}.py to make it standalone.
import argparse
import mxnet as mx
from mxnet.gluon import nn
from mxnet.contrib import autograd
import numpy as np
import matplotlib.pyplot as PL
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=2500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
ctx = mx.gpu()


netG = nn.Sequential()
# input is Z, going into a convolution
netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, in_filters=nz, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf * 8))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 4 x 4
netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, in_filters=ngf * 8, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf * 4))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 8 x 8
netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, in_filters=ngf * 4, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf * 2))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 16 x 16
netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, in_filters=ngf * 2, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 32 x 32
netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, in_filters=ngf, use_bias=False))
netG.add(nn.Activation('tanh'))
# state size. (nc) x 64 x 64


netD = nn.Sequential()
# input is (nc) x 64 x 64
netD.add(nn.Conv2D(ndf, 4, 2, 1, in_filters=nc, use_bias=False))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 32 x 32
netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, in_filters=ndf, use_bias=False))
netD.add(nn.BatchNorm(num_features=ndf * 2))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 16 x 16
netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, in_filters=ndf * 2, use_bias=False))
netD.add(nn.BatchNorm(num_features=ndf * 4))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 8 x 8
netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, in_filters=ndf * 4, use_bias=False))
netD.add(nn.BatchNorm(num_features=ndf * 8))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 4 x 4
netD.add(nn.Conv2D(1, 4, 1, 0, in_filters=ndf * 8, use_bias=False))
# netD.add(nn.Activation('sigmoid'))


netG.params.initialize(mx.init.Normal(0.05), ctx=ctx)
netD.params.initialize(mx.init.Normal(0.05), ctx=ctx)
for p in netD.params.values():
    p.set_data(mx.nd.clip(p.data(ctx=ctx), -0.01, 0.01))


optimizerG = nn.Optim(netG.params, 'rmsprop', {'learning_rate': opt.lr})
optimizerD = nn.Optim(netD.params, 'rmsprop', {'learning_rate': opt.lr})


real_label = mx.nd.ones((opt.batchSize,), ctx=ctx)
fake_label = mx.nd.zeros((opt.batchSize,), ctx=ctx)

def train_sched():
    i = 0
    while True:
        i += 1
        if ((i % 100 == 0) or (i > 2500 and i % 5 == 0 and i % 2500 > 100)):
            yield True
        else:
            yield False

sched = train_sched()
errD_list = []
errG_list = []
iter_ = 0

def cifar10_iterator(batch_size, data_shape, resize=-1):
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/cifar/train.rec')) or \
       (not os.path.exists('data/cifar/test.rec')) or \
       (not os.path.exists('data/cifar/train.lst')) or \
       (not os.path.exists('data/cifar/test.lst')):
        print("Download dataset...")
        os.system("wget -q http://data.mxnet.io/mxnet/data/cifar10.zip -P data/")
        os.chdir("./data")
        os.system("unzip -u cifar10.zip")
        os.chdir("..")
    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        # mean_img    = "data/cifar/mean.bin",
        resize      = resize,
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        # mean_img    = "data/cifar/mean.bin",
        resize      = resize,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size)

    return train, val

train_iter, val_iter = cifar10_iterator(opt.batchSize, (3, 64, 64), 64)

for epoch in range(opt.niter):
    train_iter.reset()
    for batch in train_iter:
        train_G = sched.next()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = batch.data[0].copyto(ctx) / 255. * 2. - 1.
        noise = mx.nd.random_normal(0, 1, shape=(opt.batchSize, nz, 1, 1), ctx=ctx)

        with autograd.train_section():
            errD_real = netD(data)
            #output = output.reshape((opt.batchSize, 2))
            #errD_real = nn.loss.softmax_cross_entropy_loss(output, real_label)

            fake = netG(noise)
            errD_fake = netD(fake.detach())
            #output = output.reshape((opt.batchSize, 2))
            #errD_fake = nn.loss.softmax_cross_entropy_loss(output, fake_label)
            errD = errD_real - errD_fake
            errD.backward()

        optimizerD.step(opt.batchSize)
        for p in netD.params.values():
            p.set_data(mx.nd.clip(p.data(ctx=ctx), -0.01, 0.01))
        print('D: %.6f (%.6f, %.6f)' % (
                mx.nd.mean(errD).asscalar(),
                mx.nd.mean(errD_real).asscalar(),
                mx.nd.mean(errD_fake).asscalar()
                ))
        errD_list.append(mx.nd.mean(errD).asscalar())

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if train_G:
            with autograd.train_section():
                errG = netD(fake)
                #output = output.reshape((opt.batchSize, 2))
                #errG = nn.loss.softmax_cross_entropy_loss(output, real_label)
                errG.backward()

            optimizerG.step(opt.batchSize)
            print('G:', mx.nd.mean(errG).asscalar())
            errG_list.append(mx.nd.mean(errG).asscalar())
            # Refresh statistics of D
        else:
            errG_list.append(np.nan)

        iter_ += 1
        if iter_ > 2500 and iter_ % 2500 == 0:
            fig, ax = PL.subplots(2)
            ax[0].plot(errD_list, 'r-')
            ax[0].plot(errG_list, 'b.')

            buff = np.zeros((opt.imageSize * 5, opt.imageSize * 5, 3))
            noise = mx.nd.random_normal(0, 1, shape=(25, nz, 1, 1), ctx=ctx)
            fake = netG(noise).asnumpy()

            for idx in range(25):
                i = (idx / 5) * opt.imageSize
                j = (idx % 5) * opt.imageSize
                buff[i:i+opt.imageSize, j:j+opt.imageSize] = (fake[idx].transpose(1, 2, 0) + 1) / 2
            ax[1].imshow(buff)
            PL.show()