import os, sys

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, w):
        '''
        add a new word to the vocabulary
        '''
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1

    def __call__(self, w):
        '''
        :return corresponding index of the given word
        '''
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        '''
        get number of words in the vocabulary
        '''
        return self.nwords


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)