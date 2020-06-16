import algorithm.extra_feature_lbp as extra_feature_lbp
import algorithm.classify_random as classify_random
import algorithm.lower_dimen_pca as lower_dimen_pca
import time

startTime = time.time()

# 用LBP方法进行特征预处理
extra_feature_lbp.run(method_generateFaceRS='lbp')

# 读取LBP特征处理的结果，并用PCA方法进行特征降维
lower_dimen_pca.run(
    method_readFaceRS='lbp', method_generateUpdateFaceRS='lbp_pca',
    n_components=99
)

# 用RandomForest进行分类
classify_random.run(method_readFaceRS='lbp_pca')

endTime = time.time()

print('\nLBP_PCA_Random costs %.2f seconds.' % (endTime - startTime))
