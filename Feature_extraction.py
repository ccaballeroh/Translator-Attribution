from helper.analysis import get_dataset_from_json
from helper.analysis import JSON_FOLDER
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

N = 100  # N most salient features

files = [file for file in JSON_FOLDER.iterdir() if file.name.startswith("feature")]

X_dict, y_str = get_dataset_from_json(files[0])

dict_vectorizer = DictVectorizer(sparse=True)
encoder = LabelEncoder()

X, y = dict_vectorizer.fit_transform(X_dict), encoder.fit_transform(y_str)

chi2_values, p_values = chi2(X, y)

indices = chi2_values.argsort()[-N:][::-1]  # indices of N largest chi2 values

feature_names = np.array(dict_vectorizer.get_feature_names())

print(feature_names[indices])
