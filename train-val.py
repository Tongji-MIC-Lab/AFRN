import torch
import torch.nn as nn
import torch.utils.data
import dataset as dsets
from torch.autograd import Variable
import net
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr

parser = argparse.ArgumentParser()
# parser.add_argument('--feapath', type=str, default='../data/me15res/me15tsn_v3_rgb,../data/me15res/me15tsn_v3_flow,../data/me15res/audioset')
# parser.add_argument('--dataset', type=str, default='AIMT15')
# parser.add_argument('--numclasses', type=int, default=3)

parser.add_argument('--feapath', type=str, default='../data/me16res/me16tsn_v3_rgb,../data/me16res/me16tsn_flow,../data/me16res/audioset')
parser.add_argument('--dataset', type=str, default='EIMT16')
parser.add_argument('--numclasses', type=int, default=1)

# parser.add_argument('--feapath', type=str, default='../data/VideoEmotion/tsnv3_rgb_pool/,../data/VideoEmotion/tsnv3_flow_pool,../data/VideoEmotion/audioset/')
# parser.add_argument('--dataset', type=str, default='VideoEmotion')
# parser.add_argument('--numclasses', type=int, default=4)
parser.add_argument('--split', type=int, default=1)

parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--dtype', type=str, default='arousal')
parser.add_argument('--seqlen', type=int, default=18)
parser.add_argument('--numhidden', type=int, default=1024)
parser.add_argument('--numlayers', type=int, default=1)
parser.add_argument('--numepochs', type=int, default=5)
parser.add_argument('--learate', type=float, default=0.0001)
parser.add_argument('--batchsize', type=int, default=10)
# parser.add_argument('--learate', type=float, default=0.00005)
# parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--ver', type=str, default='V1.0')
arg = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu
# seed = 1111
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

seed = 1111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


if not os.path.exists('../outmodels'):
   os.mkdir('../outmodels')

def testClassify(test_loader, ld, arg):
    module = net.afrnn(arg)
    module.cuda()
    dict = torch.load(ld['dict'])
    module.load_state_dict(dict)
    module.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images).cuda()
            outputs = module(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        ret = 100 * float(correct) / total
    print('Epoch:%d iter:%d Test Accuracy: %.3f %%, correct=%d, total=%d' %
          (ld['epoch'], ld['iter'], ret,correct,total))
    del dict
    del module
    torch.cuda.empty_cache()
    return ret

def trainClassify(train_loader, arg):
    module = net.afrnn(arg)
    module.cuda()
    ldict = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=arg.learate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=1e-6, amsgrad=True)
    for epoch in range(arg.numepochs):
        module.train()
        closs = 0
        for i, (images, labels) in enumerate(train_loader):
            module.train()
            optimizer.zero_grad()
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            outputs = module(images)
            loss = criterion(outputs, labels)
            closs += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            optimizer.step()
            if 0 == (i+1)%100:
                dicpath = '../outmodels/%s-%d_%s_%d_%d.pt' % (arg.dataset, arg.numclasses, arg.dtype, epoch + 1, i + 1)
                ldict.append({'epoch': epoch + 1, 'iter': i + 1, 'dict': dicpath})
                torch.save(module.state_dict(), dicpath)
        print('Epoch [%d/%d], Step [%d], Loss: %.4f' % (epoch + 1, arg.numepochs, i + 1, closs / (i + 1)))
        if arg.dataset == 'VideoEmotion':
            dicpath = '../outmodels/%s-%d_%d_%d.pt' % (arg.dataset, arg.numclasses, arg.split, epoch + 1)
        else:
            dicpath = '../outmodels/%s-%d_%s_%d_%d.pt' % (arg.dataset, arg.numclasses, arg.dtype, epoch + 1, i+1)
        ldict.append({'epoch': epoch + 1, 'iter': i + 1, 'dict': dicpath})
        torch.save(module.state_dict(), dicpath)
    del module
    return ldict

def testEIMT16(test_loader, ld, arg):
    module = net.afrnn(arg)
    module.cuda()
    dict = torch.load(ld['dict'])
    module.load_state_dict(dict)
    module.eval()
    total = 0
    lgt = []
    lpred = []
    for images, labels in test_loader:
        images = Variable(images).cuda()
        outputs = module(images)
        lgt.extend(labels)
        pt = np.float32(outputs.data.cpu().numpy())
        lpred.extend(pt)
    gt = np.array(lgt).tolist()
    pred = np.squeeze(np.array(lpred)).tolist()
    mse = mean_squared_error(gt,pred)
    pcc = pearsonr(gt,pred)
    print('MSE: %.3f, PCC=%.3f, total=%d, -MSE+PCC=%.3f' % (mse,pcc[0],total,-mse+pcc[0]))
    del module
    return mse, pcc[0]


def trainEIMT16(train_loader, arg):
    module = net.afrnn(arg)
    module.cuda()
    ldict = []
    optimizer = torch.optim.Adam(module.parameters(), lr=arg.learate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=1e-6, amsgrad=True)
    criterion = nn.MSELoss()
    for epoch in range(arg.numepochs):
        closs = 0
        count = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = module(images)
            outputs = torch.squeeze(outputs, dim=1)
            loss = criterion(outputs, labels)
            closs += loss.data.item()
            loss.backward()
            count += 1
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            optimizer.step()
            if 0 == (i+1)%100:
                dicpath = '../outmodels/%s-%d_%s_%d_%d.pt' % (arg.dataset, arg.numclasses, arg.dtype, epoch + 1, i + 1)
                ldict.append({'epoch': epoch + 1, 'iter': i + 1, 'dict': dicpath})
                torch.save(module.state_dict(), dicpath)
        print('Epoch [%d/%d], Step [%d], Loss: %.4f'% (epoch + 1, arg.numepochs, i + 1, closs / count))
        dicpath = '../outmodels/%s-%d_%s_%d.pt' % (arg.dataset, arg.numclasses, arg.dtype, epoch + 1)
        ldict.append({'epoch': epoch + 1, 'iter': i + 1, 'dict': dicpath})
        torch.save(module.state_dict(), dicpath)
    del module
    return ldict

def main():
    if 'AIMT15' == arg.dataset:
        train_dataset = dsets.AIMT15(arg.feapath, True, dimtype=arg.dtype, seqlen=arg.seqlen)
        test_dataset = dsets.AIMT15(arg.feapath, False, dimtype=arg.dtype, seqlen=arg.seqlen)
    elif 'EIMT16' == arg.dataset:
        train_dataset = dsets.EIMT16(arg.feapath, True, dimtype=arg.dtype, seqlen=arg.seqlen)
        test_dataset = dsets.EIMT16(arg.feapath, False, dimtype=arg.dtype, seqlen=arg.seqlen)
    elif 'VideoEmotion' == arg.dataset:
        train_dataset = dsets.VideoEmotion(arg.feapath, True, seqlen=arg.seqlen, split=arg.split, numclass=arg.numclasses)
        test_dataset = dsets.VideoEmotion(arg.feapath, False, seqlen=arg.seqlen, split=arg.split, numclass=arg.numclasses)
    arg.feadim = train_dataset.feadim
    arg.numfeas = len(arg.feadim)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=arg.batchsize,
                                               shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=8,
                                              shuffle=False, num_workers=3)
    if 'AIMT15' == arg.dataset or 'VideoEmotion' == arg.dataset:

        ldict = trainClassify(train_loader, arg)
        macc = 0
        mepoch = 0
        mi = 0
        for ld in ldict:
            ret = testClassify(test_loader, ld, arg)
            if ret > macc:
                macc = ret
                mepoch = ld['epoch']
                mi = ld['iter']
        print('max acc=%.3f epoch=%d i=%d' % (macc, mepoch, mi))
    else:
        ldict = trainEIMT16(train_loader, arg)
        mmse = 100
        mpcc = 0
        mepoch = 0
        mi = 0
        for ld in ldict:
            ret, pcc = testEIMT16(test_loader, ld, argl)
            if ret < mmse:
                mmse = ret
                mpcc = pcc
                mepoch = ld['epoch']
                mi = ld['iter']
        print('min MSE=%.3f PCC=%.3f epoch=%d i=%d' % (mmse, mpcc, mepoch, mi))


if __name__ == "__main__":
    main()

