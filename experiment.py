import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from jieba import posseg
from model import RhythmPredictor


def score(y: list, pred: list):
    tag2idx = {'#0': 0, '#1': 1, '#2': 2, '#3': 3, '#4': 4}
    res = [[0 for _ in range(5)] for _ in range(5)]
    for yi, predi in zip(y, pred):
        res[tag2idx[yi]][tag2idx[predi]] += 1
    for i in range(5):
        tot = sum(res[i])
        res[i] = [res[i][j] / tot for j in range(5)]
        print('#{}:'.format(i), res[i])


def make_data():
    words_batch, labels_batch = [], []
    with open('data.txt', 'r') as f:
        for line in f.readlines():
            left, _, right = line[:-1].partition('|')
            labels_batch.append(right.split(' ')[:-1])
            words_batch.append(left.split(' '))
    feat_as_sentence, label_as_sentence = [], []
    feat_all, label_all = [], []
    bar = tqdm(total=len(words_batch))
    for i in range(len(words_batch)):
        words, labels = words_batch[i], labels_batch[i]
        # Extract features from a sentence
        sentence_feats = RhythmPredictor.extract_features(words)
        # Features and labels as a sentence
        feat_as_sentence.append(sentence_feats)
        label_as_sentence.append(labels)
        # All features and labels
        feat_all.extend(sentence_feats)
        label_all.extend(labels)
        bar.update(1)
    bar.close()
    with open('dataset.pkl', 'wb') as f:
        data = {
            'feat_as_sentence': feat_as_sentence,
            'label_as_sentence': label_as_sentence,
            'feat_all': feat_all,
            'label_all': label_all
        }
        pickle.dump(data, f)


def make_model():
    model = RhythmPredictor(max_depth=50, n_estimators=20, n_jobs=-1)
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    feat_all = pd.DataFrame(data['feat_all'],
                            columns=RhythmPredictor.ALL_COLUMNS)
    label_all = data['label_all']
    # Fit tree using all data at one time
    train_x, train_y = feat_all[:-10000], label_all[:-10000]
    model.fit(train_x, train_y)
    # Fit CRF using data from one sentence at a time
    # feat_as_sentence = data['feat_as_sentence'][:-10000]
    # label_as_sentence = data['label_as_sentence'][:-10000]
    # model.fit_crf(feat_as_sentence[:10000], label_as_sentence[:10000])
    model.PREDICT_WITH_CRF = False
    pred = model.predict(feat_all[-10000:])
    score(label_all[-10000:], pred)
    model.dump('tree.pkl')


def test_data():
    model = RhythmPredictor()
    model.load('tree.pkl', 'crf.pt')
    model.PREDICT_WITH_CRF = False
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    count = {label: [0, 0] for label in model.RHYTHM_TAGS}
    total = [0, 0]
    feat_as_sentence = data['feat_as_sentence']
    label_as_sentence = data['label_as_sentence']
    progress_bar = tqdm(total=len(feat_as_sentence))
    for feats, labels in zip(feat_as_sentence, label_as_sentence):
        if len(feats) > 0:
            pred = model.predict(feats)
            for i in range(len(labels)):
                count[labels[i]][pred[i] == labels[i]] += 1
                total[pred[i] == labels[i]] += 1
        progress_bar.update(1)
    progress_bar.close()
    # Show prediction result
    print('Total acc: ', total[1] / sum(total))
    for label, count_pair in count.items():
        print('Label: {}, acc: {}'.format(label,
                                          count_pair[1] / sum(count_pair)))


def cross_validate_test():
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    cv = KFold(n_splits=10, shuffle=True)
    x, y = np.array(dataset['feat_all']), np.array(dataset['label_all'])
    for train_index, valid_index in cv.split(x):
        train_x, train_y = x[train_index], y[train_index]
        valid_x, valid_y = x[valid_index], y[valid_index]
        model = RhythmPredictor()
        model.PREDICT_WITH_CRF = False
        model.fit(train_x, train_y, max_depth=30)
        # Cross validation
        progress_bar = tqdm(total=len(valid_x))
        count = {label: [0, 0] for label in model.RHYTHM_TAGS}
        total = [0, 0]
        for feats, labels in zip(valid_x, valid_y):
            if len(feats) > 0:
                pred = model.predict(feats)
                for i in range(len(labels)):
                    count[labels[i]][pred[i] == labels[i]] += 1
                    total[pred[i] == labels[i]] += 1
            progress_bar.update(1)
        progress_bar.close()
        # Show prediction result
        print('Total acc: ', total[1] / sum(total))
        for label, count_pair in count.items():
            print('Label: {}, acc: {}'.format(label,
                                              count_pair[1] / sum(count_pair)))


def test_sentences():
    model = RhythmPredictor()
    model.load(tree_path='tree_50_20_95.pkl')
    model.PREDICT_WITH_CRF = False
    sentences = [
        '现在的医院越建越大病人却像赶集一样人满为患。',
        '哦有的我喜欢打羽毛球乒乓球以及玩电脑游戏。',
        '北极熊先生热情挽留。',
        '我身上分文没有。',
        '那些庄稼田园在果果眼里感觉太亲切了。',
        '她把鞋子拎在手上光着脚丫故意踩在水洼里。',
        '我为男主角感到有点遗憾。',
        '她把他那件整洁的上装的衣扣统统扣上。',
    ]
    for sentence in sentences:
        pairs = [tuple(pair) for pair in posseg.cut(sentence)]
        words = [pair[0] for pair in pairs]
        poses = [pair[1] for pair in pairs]
        start = time.time()
        labels = model.predict_words(words, poses)
        print('Time: ', time.time() - start)
        print(' '.join(words))
        print(' '.join(labels))


if __name__ == '__main__':
    # make_data()
    # make_model()
    # test_data()
    # cross_validate_test()
    test_sentences()
