#ifndef CLASSPARAMETER_H
#define CLASSPARAMETER_H

// 参数
class sParameter
{
public:
	int objNum; // 样本数
	int objLength; // 样本维度
	int clusterNum; // 聚类数
	int minClusterNum; // 最少的聚类数
	int minObjInClusterNum; // 每个聚类中的最少样本数
	int maxKmeansIter; // 最大迭代次数
};

#endif // !CLASSPARAMETER_H
