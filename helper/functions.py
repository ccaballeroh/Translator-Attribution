class Featureset:
    def __init__(self, features_list):
        self.__features_list = features_list
    @property
    def featureset(self):
        return self.__features_list

    def tf_by_translator(self):
        pass


    def tfidf_by_translator(self):
        pass