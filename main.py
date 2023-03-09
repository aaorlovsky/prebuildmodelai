from imageai.Classification import ImageClassification
import os


execution_path = os.getcwd()
prediction = ImageClassification()


prediction.setModelTypeAsResNet50()
prediction.setModelPath('resnet50-19c8e357.pth')
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "persons.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)

