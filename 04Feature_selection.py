"""This script extract the most N salient features"""

from helper.analysis import get_dataset_from_json
from helper.analysis import JSON_FOLDER
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

N = 10  # N most salient features


feature_files = [
    file for file in JSON_FOLDER.iterdir() if file.name.startswith("feature")
]

# test with just one feature file
X_dict, y_str = get_dataset_from_json(feature_files[11])

dict_vectorizer = DictVectorizer(sparse=True)
encoder = LabelEncoder()

X, y = dict_vectorizer.fit_transform(X_dict), encoder.fit_transform(y_str)

chi2_values, p_values = chi2(X, y)

indices = chi2_values.argsort()[-N:][::-1]  # indices of N largest chi2 values

# matrix with only the N most salient features
# X_reduced = X[:,indices]
# Or just use SelectKBest(chi2, k=N)
# chi2_selector = SelectKBest(chi2, k=N)
# X_kbest = chi2_selector.fit_transform(X, y)
#
# Names:
# names = np.array(dict_vectorizer.get_feature_names())
# print(names[chi2_selector.get_support()])

feature_names = np.array(dict_vectorizer.get_feature_names())

print(feature_names[indices])
