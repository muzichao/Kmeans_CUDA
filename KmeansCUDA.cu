#include "KmeansCUDA.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <iostream>

#include <stdlib.h>

#define BLOCKSIZE_16 16
#define BLOCKSIZE_32 32
#define OBJLENGTH 75

/**
* 功能：初始化每个样本的类索引
* 输出：objClusterIdx_Dev 每个样本的类别索引
* 输入：objNum 样本个数
* 输入：maxIdx 索引的最大值
*/
__global__ void KmeansCUDA_Init_ObjClusterIdx(int *objClusterIdx_Dev, int objNum, int maxIdx)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; 

	curandState s;
	curand_init(index, 0, 0, &s);

	if (index < objNum) objClusterIdx_Dev[index] = (int(curand_uniform(&s) * maxIdx));
}


/**
* 功能：更新 Kmeans 的聚类中心
* 输入：objData_Dev 样本数据
* 输入：objClusterIdx_Dev 每个样本的类别索引
* 输出：clusterData_Dev 聚类中心
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_Update_Cluster(float *objData_Dev, int *objClusterIdx_Dev, float *clusterData_Dev, sParameter myParameter)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标
	
	if (x_id < myParameter.objLength && y_id < myParameter.objNum)
	{
		int index = y_id * myParameter.objLength + x_id;
		int clusterIdx = objClusterIdx_Dev[y_id];

		atomicAdd(&clusterData_Dev[clusterIdx * myParameter.objLength + x_id], objData_Dev[index]);
	}
}

/**
*功能：更新 Kmeans 的聚类中心
* 输入：objClusterIdx_Dev 每个样本的类别索引
* 输出：objNumInCluster 每个聚类中的样本数
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_Count_objNumInCluster(int *objClusterIdx_Dev, int *objNumInCluster, sParameter myParameter)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < myParameter.objNum)
	{
		int clusterIdx = objClusterIdx_Dev[index];

		atomicAdd((int*)&objNumInCluster[clusterIdx], 1); // 计数
	}
}

/**
*功能：更新 Kmeans 的聚类中心
* 输入：objClusterIdx_Dev 每个样本的类别索引
* 输出：objNumInCluster 每个聚类中的样本数
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_Count_objNumInCluster1(int *objClusterIdx_Dev, int *objNumInCluster, sParameter myParameter)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ int sData[80];

	if (threadIdx.x < myParameter.clusterNum)
		sData[threadIdx.x] = 0;

	__syncthreads();

	if (index < myParameter.objNum)
	{
		int clusterIdx = objClusterIdx_Dev[index];
		atomicAdd((int*)&sData[clusterIdx], 1);
	}

	__syncthreads();

	if (threadIdx.x < myParameter.clusterNum)
		atomicAdd((int*)&objNumInCluster[threadIdx.x], sData[threadIdx.x]); // 计数
}

/**
*功能：平均 Kmeans 的聚类中心
* 输出：clusterData_Dev 聚类中心
* 输出：objNumInCluster 每个聚类中的样本数
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_Scale_Cluster(float *clusterData_Dev, int *objNumInCluster, sParameter myParameter)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标
	
	if (x_id < myParameter.objLength && y_id < myParameter.clusterNum)
	{
		int index = y_id * myParameter.objLength + x_id;
		clusterData_Dev[index] /= float(objNumInCluster[y_id]);
	}
}


/**
* 功能：计算两个向量的欧拉距离
* 输入：objects 样本数据
* 输出：clusters 聚类中心数据
* 输入：objLength 样本长度
*/
__device__ inline static float EuclidDistance(float *objects, float *clusters, int objLength)
{
	float dist = 0.0f;

	for (int i = 0; i < objLength; i++)
	{
		float onePoint = objects[i] - clusters[i];
		dist = onePoint * onePoint + dist;
	}

	return(dist);
}

/**
* 功能：计算所有样本与聚类中心的欧式距离
* 输入：objData_Dev 样本数据
* 输入：objClusterIdx_Dev 每个样本的类别索引
* 输入：clusterData_Dev 聚类中心
* 输出：distOfObjAndCluster_Dev 每个样本与聚类中心的欧式距离
* 输入：objNumInCluster_Dev 每个聚类中的样本数
* 输入：iter 迭代次数
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_distOfObjAndCluster(float *objData_Dev, int *objClusterIdx_Dev, float *clusterData_Dev, float *distOfObjAndCluster_Dev, int *objNumInCluster_Dev, int iter, sParameter myParameter)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	const int oneBlockData = OBJLENGTH * BLOCKSIZE_16;
	__shared__ float objShared[oneBlockData]; // 存样本
	__shared__ float cluShared[oneBlockData]; // 存聚类中心

	/* 数据读入共享内存 */
	if (y_id < myParameter.objNum)
	{
		float *objects = &objData_Dev[myParameter.objLength * blockDim.y * blockIdx.y]; // 当前块需要样本对应的首地址
		float *clusters = &clusterData_Dev[myParameter.objLength * blockDim.x * blockIdx.x]; // 当前块需要聚类中心对应的首地址

		for (int index = BLOCKSIZE_16 * threadIdx.y + threadIdx.x; index < oneBlockData; index = BLOCKSIZE_16 * BLOCKSIZE_16 + index)
		{
			objShared[index] = objects[index];
			cluShared[index] = clusters[index];
		}

		__syncthreads();
	}

	if (x_id < myParameter.clusterNum && y_id < myParameter.objNum)
	{
		 //if (objNumInCluster_Dev[x_id] < myParameter.minObjInClusterNum && iter >= myParameter.maxKmeansIter - 2)
			// distOfObjAndCluster_Dev[y_id * myParameter.clusterNum + x_id] = 3e30;
		 //else
			 distOfObjAndCluster_Dev[y_id * myParameter.clusterNum + x_id] = EuclidDistance(&objShared[myParameter.objLength * threadIdx.y], &cluShared[myParameter.objLength * threadIdx.x], myParameter.objLength);
	}
}

/**
* 功能：计算所有样本与聚类中心的欧式距离
* 输入：objData_Dev 样本数据
* 输入：objClusterIdx_Dev 每个样本的类别索引
* 输入：clusterData_Dev 聚类中心
* 输出：distOfObjAndCluster_Dev 每个样本与聚类中心的欧式距离
* 输入：objNumInCluster_Dev 每个聚类中的样本数
* 输入：iter 迭代次数
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_distOfObjAndCluster1(float *objData_Dev, int *objClusterIdx_Dev, float *clusterData_Dev, float *distOfObjAndCluster_Dev, int *objNumInCluster_Dev, int iter, sParameter myParameter)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	__shared__ float objShared[BLOCKSIZE_16][OBJLENGTH]; // 存样本
	__shared__ float cluShared[BLOCKSIZE_16][OBJLENGTH]; // 存聚类中心

	float *objects = &objData_Dev[myParameter.objLength * blockDim.y * blockIdx.y]; // 当前块需要样本对应的首地址
	float *clusters = &clusterData_Dev[myParameter.objLength * blockDim.x * blockIdx.x]; // 当前块需要聚类中心对应的首地址

	/* 数据读入共享内存 */
	if (y_id < myParameter.objNum)
	{
		for (int xidx = threadIdx.x; xidx < OBJLENGTH; xidx += BLOCKSIZE_16)
		{
			int index = myParameter.objLength * threadIdx.y + xidx;
			objShared[threadIdx.y][xidx] = objects[index];
			cluShared[threadIdx.y][xidx] = clusters[index];
		}

		__syncthreads();
	}

	if (x_id < myParameter.clusterNum && y_id < myParameter.objNum)
	{
		if (objNumInCluster_Dev[x_id] < myParameter.minObjInClusterNum && iter >= myParameter.maxKmeansIter - 2)
			distOfObjAndCluster_Dev[y_id * myParameter.clusterNum + x_id] = 3e30;
		else
			distOfObjAndCluster_Dev[y_id * myParameter.clusterNum + x_id] = EuclidDistance(objShared[threadIdx.y], cluShared[threadIdx.x], myParameter.objLength);
	}
}

/**
* 功能：计算所有样本与聚类中心的欧式距离
* 输出：objClusterIdx_Dev 每个样本的类别索引
* 输入：distOfObjAndCluster_Dev 每个样本与聚类中心的欧式距离
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_Update_ObjClusterIdx1(int *objClusterIdx_Dev, float *distOfObjAndCluster_Dev, sParameter myParameter)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < myParameter.objNum)
	{
		float *objIndex = &distOfObjAndCluster_Dev[index * myParameter.clusterNum];
		int idx = 0;
		float dist = objIndex[0];

		for (int i = 1; i < myParameter.clusterNum; i++)
		{
			if (dist > objIndex[i])
			{
				dist = objIndex[i];
				idx = i;
			}
		}
		objClusterIdx_Dev[index] = idx;
	}
}

/**
* 功能：计算所有样本与聚类中心的欧式距离（优化后的）
* 输出：objClusterIdx_Dev 每个样本的类别索引
* 输入：distOfObjAndCluster_Dev 每个样本与聚类中心的欧式距离
* 输入：myPatameter 输入参数
*/
__global__ void KmeansCUDA_Update_ObjClusterIdx(int *objClusterIdx_Dev, float *distOfObjAndCluster_Dev, sParameter myParameter)
{
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	__shared__ float sData[BLOCKSIZE_16][BLOCKSIZE_16]; // 样本与聚类中心距离
	__shared__ int sIndx[BLOCKSIZE_16][BLOCKSIZE_16]; // 距离对应的类标号

	sData[threadIdx.y][threadIdx.x] = 2e30;
	sIndx[threadIdx.y][threadIdx.x] = 0;

	__syncthreads();

	if (y_id < myParameter.objNum)
	{
		float *objIndex = &distOfObjAndCluster_Dev[y_id * myParameter.clusterNum];
		sData[threadIdx.y][threadIdx.x] = objIndex[threadIdx.x];
		sIndx[threadIdx.y][threadIdx.x] = threadIdx.x;

		__syncthreads();

		/* 每 BLOCKSIZE_16 个进行比较 */
		for (int index = threadIdx.x + BLOCKSIZE_16; index < myParameter.clusterNum; index += BLOCKSIZE_16)
		{
			float nextData = objIndex[index];
			if (sData[threadIdx.y][threadIdx.x] > nextData)
			{
				sData[threadIdx.y][threadIdx.x] = nextData;
				sIndx[threadIdx.y][threadIdx.x] = index;
			}
		}

		/* BLOCKSIZE_16 个的内部规约，到只剩 2 个 */
		for (int step = BLOCKSIZE_16 / 2; step > 1; step = step >> 1)
		{
			int idxStep = threadIdx.x + step;
			if (threadIdx.x < step && sData[threadIdx.y][threadIdx.x] > sData[threadIdx.y][idxStep])
			{
				sData[threadIdx.y][threadIdx.x] = sData[threadIdx.y][idxStep];
				sIndx[threadIdx.y][threadIdx.x] = sIndx[threadIdx.y][idxStep];
			}
			//__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			objClusterIdx_Dev[y_id] = sData[threadIdx.y][0] < sData[threadIdx.y][1] ? sIndx[threadIdx.y][0] : sIndx[threadIdx.y][1];
		}
	}
}


/**
* 功能：并行 Kmeans 聚类
* 输入：objData_Host 样本数据
* 输出：objClassIdx_Host 每个样本的类别索引
* 输出：centerData_Host 聚类中心
* 输入：myPatameter 输入参数
*/
void KmeansCUDA(float *objData_Host, int *objClassIdx_Host, float*centerData_Host, sParameter myParameter)
{
	/* 开辟设备端内存 */
	float *objData_Dev, *centerData_Dev;
	cudaMalloc((void**)&objData_Dev, myParameter.objNum * myParameter.objLength * sizeof(float));
	cudaMalloc((void**)&centerData_Dev, myParameter.clusterNum * myParameter.objLength * sizeof(float));
	cudaMemcpy(objData_Dev, objData_Host, myParameter.objNum * myParameter.objLength * sizeof(float), cudaMemcpyHostToDevice);

	int *objClassIdx_Dev;
	cudaMalloc((void**)&objClassIdx_Dev, myParameter.objNum * sizeof(int));

	float *distOfObjAndCluster_Dev; // 每个样本与聚类中心的欧式距离
	cudaMalloc((void**)&distOfObjAndCluster_Dev, myParameter.objNum * myParameter.clusterNum * sizeof(float));

	int *objNumInCluster_Dev; // 每个聚类中的样本数
	cudaMalloc((void**)&objNumInCluster_Dev, myParameter.clusterNum * sizeof(int));


	/* 线程块和线程格 */
	dim3 dimBlock1D_16(BLOCKSIZE_16 * BLOCKSIZE_16);
	dim3 dimBlock1D_32(BLOCKSIZE_32 * BLOCKSIZE_32);
	dim3 dimGrid1D_16((myParameter.objNum + BLOCKSIZE_16 * BLOCKSIZE_16 - 1) / dimBlock1D_16.x);
	dim3 dimGrid1D_32((myParameter.objNum + BLOCKSIZE_32 * BLOCKSIZE_32 - 1) / dimBlock1D_32.x);

	dim3 dimBlock2D(BLOCKSIZE_16, BLOCKSIZE_16);
	dim3 dimGrid2D_Cluster((myParameter.objLength + BLOCKSIZE_16 - 1) / dimBlock2D.x, (myParameter.clusterNum + BLOCKSIZE_16 - 1) / dimBlock2D.y);
	dim3 dimGrid2D_ObjNum_Objlen((myParameter.objLength + BLOCKSIZE_16 - 1) / dimBlock2D.x, (myParameter.objNum + BLOCKSIZE_16 - 1) / dimBlock2D.y);
	dim3 dimGrid2D_ObjCluster((myParameter.clusterNum + BLOCKSIZE_16 - 1) / dimBlock2D.x, (myParameter.objNum + BLOCKSIZE_16 - 1) / dimBlock2D.y);
	dim3 dimGrid2D_ObjNum_BLOCKSIZE_16(1, (myParameter.objNum + BLOCKSIZE_16 - 1) / dimBlock2D.y);

	// 记录时间
	cudaEvent_t start_GPU, end_GPU;
	float elaspsedTime;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	/* 样本聚类索引的初始化*/
	KmeansCUDA_Init_ObjClusterIdx<<<dimGrid1D_16, dimBlock1D_16>>>(objClassIdx_Dev, myParameter.objNum, myParameter.clusterNum);

	for (int i = 0; i < myParameter.maxKmeansIter; i++)
	{
		cudaMemset(centerData_Dev, 0, myParameter.clusterNum * myParameter.objLength * sizeof(float));
		cudaMemset(objNumInCluster_Dev, 0, myParameter.clusterNum * sizeof(int));

		/* 统计每一类的样本和 */
		KmeansCUDA_Update_Cluster<<<dimGrid2D_ObjNum_Objlen, dimBlock2D>>>(objData_Dev, objClassIdx_Dev, centerData_Dev, myParameter);

		/* 统计每一类的样本个数 */
		//KmeansCUDA_Count_objNumInCluster1<<<dimGrid1D_16, dimBlock1D_16>>>(objClassIdx_Dev, objNumInCluster_Dev, myParameter);
		KmeansCUDA_Count_objNumInCluster<<<dimGrid1D_32, dimBlock1D_32>>>(objClassIdx_Dev, objNumInCluster_Dev, myParameter);

		/* 聚类中心平均 = 样本和 / 样本个数 */
		KmeansCUDA_Scale_Cluster<<<dimGrid2D_Cluster, dimBlock2D>>>(centerData_Dev, objNumInCluster_Dev, myParameter);

		/* 计算每个样本与每个聚类中心的欧式距离 */
		KmeansCUDA_distOfObjAndCluster<<<dimGrid2D_ObjCluster, dimBlock2D>>>(objData_Dev, objClassIdx_Dev, centerData_Dev, distOfObjAndCluster_Dev, objNumInCluster_Dev, i, myParameter);

		/* 根据每个样本与聚类中心的欧式距离更新样本的类标签 */
		//KmeansCUDA_Update_ObjClusterIdx1<<<dimGrid1D_16, dimBlock1D_16>>>(objClassIdx_Dev, distOfObjAndCluster_Dev, myParameter);
		KmeansCUDA_Update_ObjClusterIdx<<<dimGrid2D_ObjNum_BLOCKSIZE_16, dimBlock2D>>>(objClassIdx_Dev, distOfObjAndCluster_Dev, myParameter);
	}

	
	// 计时结束
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	std::cout << "Kmeans 的运行时间为：" << elaspsedTime << "ms." << std::endl;

	/* 输出从设备端拷贝到内存 */
	cudaMemcpy(objClassIdx_Host, objClassIdx_Dev, myParameter.objNum * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(centerData_Host, centerData_Dev, myParameter.objNum * myParameter.objLength * sizeof(float), cudaMemcpyDeviceToHost);

	/* 释放设备端内存 */
	cudaFree(objData_Dev);
	cudaFree(objClassIdx_Dev);
	cudaFree(centerData_Dev);
	cudaFree(distOfObjAndCluster_Dev);
	cudaFree(objNumInCluster_Dev);
}