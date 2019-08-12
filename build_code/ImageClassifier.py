
import Augmentor

class ImageClassifier(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG

    def augmentation(self, path_to_data):
        print("In ImageClassifier, augmentation: {0}".format(path_to_data))
        try:
           p = Augmentor.Pipeline(path_to_data)
        except Exception as e:
            print("Unable to run augmentation: {0}".format(e))

    