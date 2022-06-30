import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--numclasses', type=int, default=8)
parser.add_argument('--ver', type=str, default='V1.0')
arg = parser.parse_args()

def readData(inpath):
    lacc = []
    lloss = []
    with open(inpath, 'r') as rf:
        lines = rf.readlines()
        num = len(lines)
        for l in range(num):
            tmp = lines[l]
            if tmp.startswith('Epoch:'):
                istart = tmp.find('Accuracy:') + len('Accuracy:')
                acc = np.float(tmp[istart : istart+7])
                lacc.append(acc)
            if tmp.startswith('Epoch ['):
                istart = tmp.find('Loss: ') + len('Loss: ')
                loss = np.float(tmp[istart:istart+6])
                lloss.append(loss)
    return np.hstack(lacc), lloss

if __name__ == '__main__':
    lacc = []
    for i in range(1, 11):
        v, loss = readData('./log/VE%d_split=%d.log' % (arg.numclasses, i))
        lacc.append(v)
    acc = np.vstack(lacc)
    imax = np.argmax(acc[0], axis=0)
    max1 = acc[:, imax]
    print('AFRN-LSTM MACC=%.1f%%' % (np.mean(max1)))
    amax = np.max(acc, axis=1)
    amean = np.mean(amax)
    print('AFRN-LSTM-Best MACC=%.1f%%' % amean)

