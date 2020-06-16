import algorithm.extra_feature_densenet as extra_feature_densenet
import algorithm.classify_cnn as classifier_cnn
import time


startTime = time.time()

# 执行DenseNet201方法进行特征提取
extra_feature_densenet.run(method_generateFaceRS='densenet')

# 执行CNN方法进行分类
classifier_cnn.run(method_readFaceRS='densenet')

endTime = time.time()

print('DenseNet_CNN costs: ' + str(endTime - startTime))
