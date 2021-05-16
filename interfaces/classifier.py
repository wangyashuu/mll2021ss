from abc import ABC, abstractmethod

class Classifier(ABC):

    @abstractmethod
    def learn(self, features, target):
        pass


    @abstractmethod
    def infer(self, features):
        pass

