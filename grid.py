import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from model import RhythmPredictor

params = {'max_depth': [25, 30, 35, 40, 45, 50, 55, 60, None],
          'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
          'n_jobs': [-1]}
grid = GridSearchCV(RandomForestClassifier(),
                    param_grid=params,
                    scoring='f1_micro',
                    n_jobs=-1,
                    iid=False,
                    cv=5,
                    verbose=20)

# Load data
with open('dataset.pkl', 'rb') as f:
    data = pickle.load(f)
feat_all = pd.DataFrame(data['feat_all'][:100000],
                        columns=RhythmPredictor.ALL_COLUMNS)
label_all = data['label_all'][:100000]

# Encode data
model = RhythmPredictor()
feat_all = model.load(tree_path='tree.pkl').encode_features(feat_all,
                                                            is_train=False)
grid.fit(feat_all, label_all)
print(grid.cv_results_, grid.best_params_, grid.best_score_)

# Dump grid search results
with open('grid_results.pkl', 'wb') as f:
    pickle.dump(grid, f)
