import torch.utils.data as data
import numpy as np
import os
import os.path
import torch

def norm2dl2(dat):
    norm = np.sqrt(np.sum(dat*dat,axis=1)) + 1e-30
    dt = np.transpose(np.transpose(dat)/norm)
    return dt  

class AIMT15(data.Dataset):
    lstrain = './list/AIMT15_train.list'
    lstest = './list/AIMT15_test.list'
    def _readlist_(self, listf, dimtype='arousal'):
        vid = []
        label = []
        with open(listf,'r') as lfo:
            print('load list ...%s'%listf)
            lines = lfo.readlines()
            for l in lines:
                line = l.split(' ')
                if len(line) != 3:
                    print('[ERROR] [%s], line [%s]  error'%(listf, l))
                    exit(0)
                vid.append(line[0].split('/')[-1])
                if dimtype == 'arousal':
                    clabel= int(line[2]) + 1
                elif dimtype == 'valence':
                    clabel= int(line[1]) + 1
                else:
                    exit(0)
                label.append(clabel)
        return vid, label

    def __init__(self, root, train=True, dimtype='arousal',seqlen=160):
        self.root = root.split(',')
        self.train = train
        self.vid = []
        self.label = []
        self.seqlen = seqlen

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            self.vid, self.label = self._readlist_(self.lstrain, dimtype)
        else:
            self.vid, self.label = self._readlist_(self.lstest, dimtype)

        self.feadim = []
        for r in self.root:
            fpath = '%s/%s.npy' % (r, self.vid[0])
            if r.find('audioset') == -1:
                self.feadim.append(2 * np.load(fpath).shape[1])
            else:
                self.feadim.append(np.load(fpath).shape[1])

    def __getitem__(self, index):
        lvfea = []
        for r in self.root:
            target = self.label[index]
            fpath = '%s/%s.npy'%(r,self.vid[index])
            vdata = np.load(fpath)
            nv = vdata.shape[0]
            span = 25
            if r.find('audioset') == -1:
                maxlen = self.seqlen * span
            else:
                maxlen = self.seqlen
            if nv < maxlen:
                while vdata.shape[0] < maxlen:
                    vdata = np.vstack((vdata, vdata))
            nv = vdata.shape[0]
            if r.find('audioset') == -1:
                vfea = np.zeros((self.seqlen,2 * vdata.shape[1]), dtype=vdata.dtype)
                for i in range(self.seqlen):
                    dmean = np.mean(vdata[i*span:(i+1)*span,:],axis=0)
                    dstd = np.std(vdata[i*span:(i+1)*span,:],axis=0)
                    vfea[i, :] = np.concatenate((dmean, dstd), axis=0)
                vfea = np.float32(norm2dl2(vfea))
            else:
                vfea = np.float32(norm2dl2(vdata[:self.seqlen,:]))
            lvfea.append(vfea)
        fea = np.hstack(lvfea)
        return fea, target

    def __len__(self):
        return len(self.label)

    def _check_exists(self):
        return os.path.exists(self.root[0])

class EIMT16(data.Dataset):
    lstrain = './list/EIMT16_train.list'
    lstest = './list/EIMT16_test.list'

    def _readlist_(self, listf, dimtype='arousal'):
        vid = []
        label = []
        with open(listf,'r') as lfo:
            print('load list ...%s'%listf)
            lines = lfo.readlines()
            for l in lines:
                line = l.split(' ')
                if len(line) != 3:
                    print('[ERROR] [%s], line [%s]  error'%(listf, l))
                    exit(0)
                vid.append(line[0].split('/')[-1])

                if dimtype == 'arousal':
                    clabel= np.float32(line[2])
                elif dimtype == 'valence':
                    clabel= np.float32(line[1])
                else:
                    exit(0)

                label.append(clabel)
        return vid, label

    def __init__(self, root, train=True, dimtype='arousal',seqlen=160):
        self.root = root.split(',')
        self.train = train
        self.vid = []
        self.label = []
        self.seqlen = seqlen

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            self.vid, self.label = self._readlist_(self.lstrain, dimtype)
        else:
            self.vid, self.label = self._readlist_(self.lstest, dimtype)
        self.feadim = []
        for r in self.root:
            fpath = '%s/%s.npy' % (r, self.vid[0])
            if r.find('audioset') == -1:
                self.feadim.append(2 * np.load(fpath).shape[1])
            else:
                self.feadim.append(np.load(fpath).shape[1])

    def __getitem__(self, index):
        lvfea = []
        for r in self.root:
            target = self.label[index]
            fpath = '%s/%s.npy'%(r,self.vid[index])
            vdata = np.load(fpath)
            nv = vdata.shape[0]
            span = 25
            if r.find('audioset') == -1:
                maxlen = self.seqlen * span
            else:
                maxlen = self.seqlen
            if nv < maxlen:
                while vdata.shape[0] < maxlen:
                    vdata = np.vstack((vdata, vdata))
            nv = vdata.shape[0]
            if r.find('audioset') == -1:
                vfea = np.zeros((self.seqlen,2 * vdata.shape[1]), dtype=vdata.dtype)
                for i in range(self.seqlen):
                    dmean = np.mean(vdata[i*span:(i+1)*span,:],axis=0)
                    dstd = np.std(vdata[i*span:(i+1)*span,:],axis=0)
                    vfea[i, :] = np.concatenate((dmean, dstd), axis=0)
                vfea = np.float32(norm2dl2(vfea))
            else:
                vfea = np.float32(norm2dl2(vdata[:self.seqlen,:]))
            lvfea.append(vfea)
        fea = np.hstack(lvfea)
        return fea, target

    def __len__(self):
        return len(self.label)

    def _check_exists(self):
        return os.path.exists(self.root[0])


class VideoEmotion(data.Dataset):
    def _readlist_(self, listf):
        vid = []
        label = []
        with open(listf,'r') as lfo:
            print('load list ...%s'%listf)
            lines = lfo.readlines()
            for l in lines:
                line = l.split(' ')
                if len(line) != 2:
                    print('[ERROR] [%s], line [%s]  error'%(listf, l))
                    exit(0)
                vid.append(line[0].split('/')[-1])
                clabel = int(line[1])
                label.append(clabel)
        return vid, label

    def __init__(self, root, train=True,seqlen=160, split=1, numclass=8):
        if 8 == numclass:
            self.lstrain = './list/VE_%d_Trainset.list'%(split)
            self.lstest = './list/VE_%d_Testset.list'%(split)
        else:
            self.lstrain = './list/VE4_%d_Trainset.list'%(split)
            self.lstest = './list/VE4_%d_Testset.list'%(split)
        self.root = root.split(',')
        self.train = train
        self.vid = []
        self.label = []
        self.seqlen = seqlen
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            self.vid, self.label = self._readlist_(self.lstrain)
        else:
            self.vid, self.label = self._readlist_(self.lstest)
        self.feadim = []
        for r in self.root:
            fpath = '%s/%s.npy' % (r, self.vid[0])
            if r.find('audioset') == -1:
                self.feadim.append(2 * np.load(fpath).shape[1])
            else:
                self.feadim.append(np.load(fpath).shape[1])

    def __getitem__(self, index):
        lvfea = []
        for r in self.root:
            target = self.label[index]
            fpath = '%s/%s.npy'%(r,self.vid[index])
            vdata = np.load(fpath)
            nv = vdata.shape[0]
            span = 25
            if r.find('audioset') == -1:
                maxlen = self.seqlen * span
            else:
                maxlen = self.seqlen
            if nv < maxlen:
                while vdata.shape[0] < maxlen:
                    vdata = np.vstack((vdata, vdata))
            nv = vdata.shape[0]
            if r.find('audioset') == -1:
                vfea = np.zeros((self.seqlen,2 * vdata.shape[1]), dtype=vdata.dtype)
                for i in range(self.seqlen):
                    dmean = np.mean(vdata[i*span:(i+1)*span,:],axis=0)
                    dstd = np.std(vdata[i*span:(i+1)*span,:],axis=0)
                    vfea[i, :] = np.concatenate((dmean, dstd), axis=0)
                vfea = np.float32(norm2dl2(vfea))
            else:
                vfea = np.float32(norm2dl2(vdata[:self.seqlen,:]))
            lvfea.append(vfea)
        fea = np.hstack(lvfea)
        return fea, target

    def __len__(self):
        return len(self.label)

    def _check_exists(self):
        return os.path.exists(self.root[0])

def main():
    # train_dataset = AIMT15('../data/me15res/me15tsn_v3_rgb', True, 'arousal', 18)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=10,
    #                                            shuffle=True, num_workers=3)

    test_dataset = VideoEmotion('../data/VideoEmotion/tsnv3_rgb_pool', False, 18, 2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=10,
                                              shuffle=False)
    for i, (images, labels) in enumerate(test_loader):
        print(images.shape)

if __name__ == "__main__":
    main()