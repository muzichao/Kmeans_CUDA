#ifndef KMEANSCUDA_H
#define KMEANSCUDA_H

#include "ClassParameter.h"

/**
* 功能：并行 Kmeans 聚类
* 输入：objData 样本数据
* 输出：objClusterIdx 每个样本的类别索引
* 输出：clusterData 聚类中心
* 输入：myPatameter 输入参数
*/
void KmeansCUDA(float *objData, int *objClusterIdx, float*clusterData, sParameter myParameter);

#endif // KMEANSCUDA_H