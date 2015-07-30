#include "KmeansCUDA.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ClassParameter.h"
#include "ReadSaveImage.h"

#include <iostream>

using std::cout;
using std::endl;

int main()
{
	sParameter myParameter{63504, 75, 80, 40, 150, 14};

	float *objData = (float*)malloc(myParameter.objNum * myParameter.objLength * sizeof(float));
	float *centerData = (float*)malloc(myParameter.clusterNum * myParameter.objLength * sizeof(float));
	int *objClassIdx = (int*)malloc(myParameter.objNum * sizeof(int));

	ReadData(objData, myParameter);

	KmeansCUDA(objData, objClassIdx, centerData, myParameter);

	SaveData(objClassIdx, myParameter);
	
	cudaDeviceReset();
	return 0;
}


