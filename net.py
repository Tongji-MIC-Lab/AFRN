import torch
import torch.nn as nn
import torch.nn.init as init

def init_lstm(lstm, init_weight=0.1):
    init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)
    if lstm.bias:
        init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
        init.zeros_(lstm.bias_hh_l0.data)
    return lstm


class TempFusionLayer(nn.Module):
    def __init__(self, seqlen, numhidden, numclasses):
        super(TempFusionLayer, self).__init__()
        self.seqlen = seqlen
        self.w = nn.Parameter(torch.ones(self.seqlen),requires_grad=True)
        self.fc = nn.Linear(numhidden * 2, numclasses)

    def forward(self, x):
        lout = []
        seqlen = x.size(1)
        for s in range(seqlen):
            ofc = self.fc(x[:, s, :]) * self.w[s]
            ofc = torch.sigmoid(ofc)
            lout.append(ofc)
        out = torch.stack(lout, dim=1)
        out = torch.mean(out, dim=1)
        return out

class ModalFusionLayer(nn.Module):
    def __init__(self, numfeas):
        super(ModalFusionLayer, self).__init__()
        self.numfeas = numfeas
        self.w = nn.Parameter(torch.ones(self.numfeas), requires_grad=True)

    def forward(self, x):
        lout = []
        for i in range(self.numfeas):
            tmp = x[i] * self.w[i]
            lout.append(tmp)
        out = torch.stack(lout, dim=1)
        out = torch.mean(out, dim=1)
        return out

class afrnn(nn.Module):
    def __init__(self, arg):
        super(afrnn, self).__init__()
        self.num_layers = arg.numlayers
        self.feadim = arg.feadim
        self.numfeas = arg.numfeas
        self.dropout = nn.Dropout(0.3)
        for i in range(self.numfeas):
            rnn = init_lstm(nn.LSTM(self.feadim[i], arg.numhidden, arg.numlayers,batch_first=True,
                          bidirectional=True, dropout=0))
            self.__setattr__('lstm%d' % i, rnn)

        self.tf = TempFusionLayer(arg.seqlen, arg.numhidden, arg.numclasses)
        self.mf = ModalFusionLayer(self.numfeas)
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-initrange, initrange)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        lo = []
        dstart = 0
        for f in range(self.numfeas):
            dend = dstart + self.feadim[f]
            txd = x[:,:,dstart:dend]
            txd = self.dropout(txd)
            out, _ = self.__getattr__('lstm%d'%f).forward(txd)
            out = self.tf(out)
            dstart = dend
            lo.append(out)
        out = self.mf(lo)
        return out
