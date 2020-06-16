import algorithm.extra_feature_resnet50 as extra_feature_resnet50
import algorithm.classify_xgboost as classifier_xgboost
import time

startTime = time.time()

# 执行ResNet50方法进行特征提取
extra_feature_resnet50.run(method_generateFaceRS='resnet')

# 执行XGBoost方法进行分类，输出ResNet_XGBoost犯错矩阵
classifier_xgboost.run(method_readFaceRS='resnet')

endTime = time.time()

print('\nResNet_XGBoost costs %.2f seconds.' % (endTime - startTime))
