# coding: utf-8
import os
import sys
sys.path.append('..')
sys.path.append('../ch07')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq

#import load_data_t
import load_data

def eval_s2s(model, question, correct, id_to_word, verbos=False):
    correct = correct.flatten()
    
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 문자열로 변환
    question = ' '.join([id_to_word[int(c)] for c in question.flatten() if (c != 0) ])
    correct = ' '.join([id_to_word[int(c)] for c in correct])
    guess = ' '.join([id_to_word[int(c)] for c in guess])

    if verbos:
        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0


# 데이터 읽기
#(x_train, t_train), (x_test, t_test) = load_data_t.load_data()
(x_train, t_train), (x_test, t_test) = load_data.load_data()
#word_to_id, id_to_word = load_data_t.get_vocab()
word_to_id, id_to_word = load_data.get_vocab()
datasize = x_train.shape[0]

# 하이퍼파라미터 설정
vocab_size = len(word_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
#batch_size = 32
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_s2s(model, question, correct, id_to_word, verbose)
    
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('정확도 %.3f%%' % (acc * 100))

model.save_params()

# 그래프 그리기
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()