import torch
import torch.utils.data
import dataset as dsets
from torch.autograd import Variable
import net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--feapath', type=str, default='./data/VideoEmotion/tsnv3_rgb_pool/,./data/VideoEmotion/tsnv3_flow_pool,./data/VideoEmotion/audioset/')
parser.add_argument('--dataset', type=str, default='VideoEmotion')
parser.add_argument('--models', type=str, default='./models')
parser.add_argument('--numclasses', type=int, default=4)
parser.add_argument('--numepochs', type=str, default=29)
# parser.add_argument('--numclasses', type=int, default=8)
# parser.add_argument('--numepochs', type=int, default=28)

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dtype', type=str, default='valence')
parser.add_argument('--net', type=str, default='afrnn')
parser.add_argument('--seqlen', type=int, default=18)
parser.add_argument('--numhidden', type=int, default=1024)
parser.add_argument('--numlayers', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--learate', type=float, default=0.00005)
parser.add_argument('--ver', type=str, default='V1.0')
arg = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu
seed = 1111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def testClassify(test_loader, module, path):
    dict = torch.load(path)
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
    del dict
    torch.cuda.empty_cache()
    return ret

def main():
    if 'AIMT15' == arg.dataset or 'VideoEmotion' == arg.dataset:
        if 'AIMT15' == arg.dataset:
            test_dataset = dsets.AIMT15(arg.feapath, False, dimtype=arg.dtype, seqlen=arg.seqlen)
        elif 'VideoEmotion' == arg.dataset:
            test_dataset = dsets.VideoEmotion(arg.feapath, False, seqlen=arg.seqlen, split=1,
                                              numclass=arg.numclasses)
        arg.feadim = test_dataset.feadim
        arg.numfeas = len(arg.feadim)
        module = net.afrnn(arg)
        module.cuda()
        lacc = []
        for i in range(1,11):
            if 'AIMT15' == arg.dataset:
                path = '%s/%s-%s_%d_%s.pt' % (arg.models, arg.dataset, arg.dtype, i, arg.numepochs)
                test_dataset = dsets.AIMT15(arg.feapath, False, dimtype=arg.dtype, seqlen=arg.seqlen)
            elif 'VideoEmotion' == arg.dataset:
                path = '%s/%s-%d_%d_%s.pt' % (arg.models, arg.dataset, arg.numclasses, i, arg.numepochs)
                test_dataset = dsets.VideoEmotion(arg.feapath, False, seqlen=arg.seqlen, split=i, numclass=arg.numclasses)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8,
                                                      shuffle=False, num_workers=3)
            ret = testClassify(test_loader, module, path)
            lacc.append(ret)
        np.set_printoptions(precision=3)
        print(np.array(lacc))
        print('MACC=%.1f'%(np.mean(lacc)))

if __name__ == "__main__":
    main()

