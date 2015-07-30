#include "ReadSaveImage.h"
#include <iostream>
#include <fstream>

/**
* 功能：从txt文件中读取数据
* 输出：fileData 输出数据的头指针
* 输入：fileName 读取的文本文件的文件名
* 输入：dataNum 读取的数据个数
*/
void ReadFile(float *fileData, std::string fileName, int dataNum)
{
	std::fstream file;
	file.open(fileName, std::ios::in);
	
	if (!file.is_open())
	{
		std::cout << "不能打开文件" << std::endl;
		return;
	}

	// 读入数据到内存中
	for (int i = 0; i < dataNum; i++) file >> fileData[i];

	file.close();
}

void ReadData(float *fileData, sParameter myParameter)
{
	std::string str = "D:\\Document\\vidpic\\CUDA\\Kmeans\\objData.txt";  //文件路径
	ReadFile(fileData, str, myParameter.objNum*myParameter.objLength);
}

/**
* 功能：把数据保存到文本文件中
* 输入：fileData 输入数据的头指针
* 输入：fileName 保存的文本文件的文件名
* 输入：dataNum 保存的数据个数
*/
void SaveFile(int *fileData, std::string fileName, int dataNum)
{
	std::fstream file;
	file.open(fileName, std::ios::out);

	if (!file.is_open())
	{
		std::cout << "不能打开文件" << std::endl;
		return;
	}

	// 读入数据到内存中
	for (int i = 0; i < dataNum; i++) file << fileData[i] << std::endl;

	file.close();
}

void SaveData(int *fileData, sParameter myParameter)
{
	std::string str = "D:\\Document\\vidpic\\CUDA\\Kmeans\\objClusterIdxPre.txt";  //文件路径
	SaveFile(fileData, str, myParameter.objNum);
}