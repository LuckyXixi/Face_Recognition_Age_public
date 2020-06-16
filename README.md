# Face_Recognition_Age

## 项目介绍

参见：[Face Recognition Project](https://courses.media.mit.edu/2004fall/mas622j/04.projects/faces/)

## 工作总结

对所提供的原始数据及其对应标签进行系统性整合，自定义数据接口，重新划分数据集，并进行“特征提取-特征降维-分类”三个步骤，实现了较为完整的模式识别图像有监督分类任务。在初步对比17种算法组合之后，最终确定适合本数据集有监督学习任务的最优算法:

经典：LBP-(PCA)-Adaboost、HOG-(PCA)-SVM;

智能：DenseNet201-(KPCA)-CNN、ResNet50-(KPCA)-XGBoost.

并且，将上述算法结合有监督分类任务进行功能扩展、性能优化等工作。最终，上述四种算法组合，在分类准确率、精确度等分类结果评估指标上取得明显优于同类算法的成绩。

实验过程中，对于同一套算法，发现不进行特征降维的情况下，即便取得较高的分类准确程度，但计算机运行效率较低、内存开销较大。因此，在特征提取与分类环节之间，分别采用经典与改进的特征降维算法，成功地在不丢失分类精度的前提下，减少图像分类执行过程所耗费的时间，降低运行所消耗的内存。

## 初步结果

<table>
    <tr>
        <th>算法组合</th>
        <th>score</th>
    </tr>
    <tr>
        <th>SIFT-KNN (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-KNN (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-PCA-KNN (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-PCA-KNN (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-SVM (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-SVM (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-PCA-SVM (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>SIFT-PCA-SVM (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-CNN</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-SVM (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-SVM (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-KNN (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-KNN (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-PCA-KNN (OVO)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-PCA-KNN (OVR)</th>
        <th>0.795</th>
    </tr>
    <tr>
        <th>LBP-RandomForest (OVO)</th>
        <th>0.802</th>
    </tr>
    <tr>
        <th>LBP-RandomForest (OVR)</th>
        <th>0.811</th>
    </tr>
    <tr>
        <th>LBP-PCA-RandomForest (OVO)</th>
        <th>0.856</th>
    </tr>
    <tr>
        <th>LBP-PCA-RandomForest (OVR)</th>
        <th>0.855</th>
    </tr>
    <tr>
        <th>LBP-Adaboost (OVO)</th>
        <th>0.910</th>
    </tr>
    <tr>
        <th>LBP-Adaboost (OVR)</th>
        <th>0.920</th>
    </tr>
    <tr>
        <th>LBP-PCA-Adaboost (OVO)</th>
        <th>0.910</th>
    </tr>
    <tr>
        <th>LBP-PCA-Adaboost (OVR)</th>
        <th>0.900</th>
    </tr>
    <tr>
        <th>HOG-SVM (OVO)</th>
        <th>0.928</th>
    </tr>
    <tr>
        <th>HOG-SVM (OVR)</th>
        <th>0.927</th>
    </tr>
    <tr>
        <th>HOG-PCA-SVM (OVO)</th>
        <th>0.945</th>
    </tr>
    <tr>
        <th>HOG-PCA-SVM (OVR)</th>
        <th>0.945</th>
    </tr>
    <tr>
        <th>ResNet50-SVM (OVO)</th>
        <th>0.928</th>
    </tr>
    <tr>
        <th>ResNet50-SVM (OVR)</th>
        <th>0.927</th>
    </tr>
    <tr>
        <th>ResNet50-KPCA-XGBoost (OVO)</th>
        <th>0.945</th>
    </tr>
    <tr>
        <th>ResNet50-KPCA-XGBoost (OVR)</th>
        <th>0.945</th>
    </tr>
    <tr>
        <th>DenseNet201-CNN</th>
        <th>0.865</th>
    </tr>
    <tr>
        <th>DenseNet201-KPCA-CNN</th>
        <th>0.805</th>
    </tr>

</table>

对于LBP-(PCA)-Adaboost、HOG-(PCA)-SVM、ResNet50-(KPCA)-XGBoost，进一步完善实验结果，打印分类报告、混淆矩阵、犯错矩阵、ROC曲线等评估指标并进行部分数据可视化。