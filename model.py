import pickle
import logging
import pandas as pd
import numpy as np
import torch
from jieba import posseg
from pypinyin import pinyin, Style
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from torch import optim
from tqdm import tqdm
from crf import CRF

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('RhythmPredictor')


class RhythmPredictor(object):
    """A simple rhythm predict model."""

    NUMERIC_COLUMNS = ['num_left', 'num_right', 'size_left', 'size_right',
                       'len_left', 'len_right']
    CATEGORICAL_COLUMNS = ['pos_left_2', 'pos_left', 'pos_right', 'pos_right_2',
                           'tone_left', 'tone_right']
    ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    RHYTHM_TAGS = ['#0', '#1', '#2', '#3', '#4']
    PREDICT_WITH_CRF = False

    def __init__(self, **kwargs):
        self._forest = RandomForestClassifier(**kwargs)
        self._crf = CRF(self.RHYTHM_TAGS)
        self._transformer = None
        self._opt = None

    def fit(self, features: pd.DataFrame, tags: np.ndarray,
            **kwargs) -> 'RhythmPredictor':
        """Training classifier using extracted features from sentences.

        Args:
            features: unencoded dataframe extracted from sentences
            tags: tags corresponds to each feature row
            kwargs: additional parameters for fitting
        """
        logger.info('Encoding feartures...')
        x = self.encode_features(features, is_train=True)
        # Use masks to train different classifiers
        logger.info('Building classifier for tags...')
        self._forest.fit(x, tags, **kwargs)
        return self

    def fit_crf(self, feat_as_sentence: list, tag_as_sentence: list,
                **kwargs) -> 'RhythmPredictor':
        """Training CRF model using extracted features from a sentence.

        Args:
            feat_as_sentence: list of features in each sentence
            tag_as_sentence: list of tags in each sentence
            kwargs: training parameters for crf
        """
        # Encode features first
        encoded_feats = []
        logger.info('Encoding features...')
        progress_bar = tqdm(total=len(feat_as_sentence))
        for feats in feat_as_sentence:
            encoded_feats.append(self.encode_features(
                pd.DataFrame(feats, columns=self.ALL_COLUMNS)))
            progress_bar.update(1)
        progress_bar.close()

        # Fit for several epoches
        epoches = kwargs.get('epoches', 10)
        loss_thres = kwargs.get('loss_threshold', 1e-2)
        self._crf = CRF(self.RHYTHM_TAGS)
        self._opt = optim.Adam(params=self._crf.parameters(),
                               lr=kwargs.get('lr', 0.03),
                               weight_decay=kwargs.get('weight_decay', 1e-3))
        logger.info('Training CRF layer:')
        prev_loss = 0.
        for e in range(1, epoches + 1):
            total_loss = 0.
            logger.info('Epoch {}:'.format(e))
            progress_bar = tqdm(total=len(tag_as_sentence))
            for feats, tags in zip(encoded_feats, tag_as_sentence):
                if len(tags) > 0:
                    # CRF is only applicable to sentences with 2 or more tags
                    scores = self._forest.predict_proba(feats)
                    # Training ...
                    # Clear grad history for each batch
                    self._crf.zero_grad()
                    # Calclate loss and backward propagation
                    loss = self._crf.neg_log_likelihood(scores, tags)
                    total_loss += loss.item()
                    if loss.requires_grad:
                        loss.backward()
                        # Update parameters, once for a batch
                        self._opt.step()
                progress_bar.update(1)
            progress_bar.close()
            logger.info('Total loss for epoch {}: {}'.format(e, total_loss))
            if np.abs(total_loss - prev_loss) <= loss_thres:
                logger.info('Loss converged.')
                break
            prev_loss = total_loss
        return self

    def predict_words(self, words: list, pos_list: list = None) -> list:
        """Predict rhythms by words split from a sentence.
        We first extract features for seperations between each pair of words,
        and encode string features to integer ones, then use decision tree
        model to predict tags.

        Args:
            words: cut from a sentence.
            pos_list: list of POS of words.

        Returns:
            tags: predicted tag list of rhythm.
        """
        feats = self.extract_features(words, pos_list)
        # Predict rhythm tags
        return self.predict(feats)

    def predict(self, features: list) -> list:
        """Predict rhythms by features in a dataframe.

        Args:
            features: Unencoded features, shape = (num_seperations, num_features)

        Returns:
            tags: predicted tag list of rhythm.
        """
        # Encode string features
        features = pd.DataFrame(features, columns=self.ALL_COLUMNS)
        x = self.encode_features(features, is_train=False)
        # Predict by all trees
        emit_scores = self._forest.predict_proba(x)
        if self.PREDICT_WITH_CRF:
            # Aggregate results using CRF
            return self._crf.forward(emit_scores)
        else:
            # Aggregate results by choosing tag with maximum probability
            tag_indices = emit_scores.argmax(axis=1)
            tags = [self.RHYTHM_TAGS[idx] for idx in tag_indices]
            return tags

    @staticmethod
    def extract_features(words: list, pos_list: list = None) -> list:
        """Extract features for each seperation and form a dataframe
        Features include: num_left, num_right, size_left, size_right, len_left,
                          len_right, pos_left_2, pos_left, pos_right,
                          pos_right_2, tone_left, tone_right.

        Args:
            words: list of words cut from a sentence.
            pos_list: list of POS of each word.

        Returns:
            feats: list of features of each seperation.
        """

        def extract_pinyin(char: str) -> tuple:
            # Extract pinyin information from a character
            ph = list(pinyin(char, style=Style.TONE3))[0][0]
            if ph[-1].isdigit():
                tone, ph = ph[-1], ph[:-1]
            else:
                tone = '5'
            return ph, tone

        # Extract features from cuts in a sentence
        feats = []
        # Get POS
        if not pos_list:
            pos_list = []
            for word in words:
                # POS of the first tokenized word in original word sequence
                _, pos = list(posseg.cut(word))[0]
                pos_list.append(pos)
        # Iterates on all cut between 2 words
        num_cut = len(words) - 1
        for index in range(num_cut):
            # Numeric features
            # Number of words, size of left / right part string
            num_left, num_right = index + 1, num_cut - index
            size_left, size_right = len(words[index]), len(words[index + 1])
            len_left, len_right = sum(map(len, words[:index])), sum(
                map(len, words[index + 1:]))
            # Categorical features
            # POS information
            pos_left_2 = pos_list[index - 1] if index > 0 else 'BEG'
            pos_right_2 = pos_list[index + 2] if index < num_cut - 1 else 'END'
            pos_left, pos_right = pos_list[index], pos_list[index + 1]
            # Pinyin information
            _, tone_left = extract_pinyin(words[index][-1])
            _, tone_right = extract_pinyin(words[index + 1][0])
            # Summarize all information
            feat = [num_left, num_right, size_left, size_right, len_left,
                    len_right, pos_left_2, pos_left, pos_right,
                    pos_right_2, tone_left, tone_right]
            feats.append(feat)
        return feats

    def encode_features(self, features: pd.DataFrame,
                        is_train: bool = False) -> pd.DataFrame:
        """Encode string-typed features into integer ones.
        Encode is based on a copy of original feature dataframe.

        Args:
            features: original features extracted from seperations.
            is_train: whether is training (generating and save new encoders), or predicting (using old encoders).

        Returns:
            encoded: encoded features.
        """
        encoded = features
        # Deal with unseen tags.
        if is_train:
            # Deal with columns
            transformer = ColumnTransformer(transformers=[
                ('numeric', 'passthrough', self.NUMERIC_COLUMNS),
                ('categorical', OneHotEncoder(handle_unknown='ignore'),
                 self.CATEGORICAL_COLUMNS)
            ])
            encoded = transformer.fit_transform(encoded)
            self._transformer = transformer
        else:
            encoded = self._transformer.transform(encoded)
        return encoded

    def load(self, tree_path: str = None,
             crf_path: str = None) -> 'RhythmPredictor':
        # Load encoders and tree models from pickle file, and also CRF model.
        if tree_path is not None:
            with open(tree_path, 'rb') as f:
                model_dict = pickle.load(f)
            self._forest = model_dict['forest']
            self._transformer = model_dict['transformer']
        if crf_path is not None:
            self._crf.load_state_dict(torch.load(crf_path))
        return self

    def dump(self, tree_path: str = None,
             crf_path: str = None) -> 'RhythmPredictor':
        # Dump encoders and tree models into pickle file, and also CRF model.
        if tree_path is not None:
            tree_dict = {'forest': self._forest,
                         'transformer': self._transformer}
            with open(tree_path, 'wb') as f:
                pickle.dump(tree_dict, f)
        if crf_path is not None:
            torch.save(self._crf.state_dict(), crf_path)
        return self
