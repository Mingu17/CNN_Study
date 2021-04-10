#include <iostream>
#include <iomanip>
#include "Core/Matrix.h"
#include "Network/MultiLayerNetwork.h"
#include "Network/ConvolutionNetwork.h"
#include "Optimizer/Adam.h"
#include "Optimizer/AdaGrad.h"
#include "Optimizer/RMSProp.h"
#include "Optimizer/Nesterov.h"
#include "Core/DataUtil.h"

#include "Core/NumUtil.h"
#include <random>

using namespace std;

int main(int argc, char** argv) 
{
	vector<Matrix> trainData;
	vector<Matrix> trainLabel;
	vector<Matrix> testData;
	vector<Matrix> testLabel;

	vector<Matrix*> dataPtr;
	vector<Matrix*> labelPtr;

	int trainNum = DataUtil::load_mnist("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", trainData, trainLabel, false);
	int testNum = DataUtil::load_mnist("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", testData, testLabel, false);
	cout << "Train data count : " << trainData.size() << endl;
	
	ConvolutionNetwork network;
	//trainNum = trainNum / 10;
	//testNum = testNum / 10;
	int batchSize = 100;// / 10;
	int iterPerEpoch = trainNum / batchSize;
	int itersNum = 40000;// / 10;
	float learningRate = 0.001f;

	vector<LayerInfo> layerInfo;
	DataUtil::create_cnn_info3(layerInfo, batchSize);
	Adam optimizer;
	//RMSProp optimizer;
	//Nesterov optimizer;
	optimizer.initParams(learningRate);
	network.init(layerInfo, optimizer, learningRate, 0.01f, 0.0001f);

	cout << "Learning Rate : " << learningRate << endl;

	int i = 0;
	
	//ordered selected
	vector<int> shuffleIdx;
	random_device rd;
	mt19937_64 rnd(rd());
	for (i = 0; i < trainNum; ++i) shuffleIdx.push_back(i);
	shuffle(shuffleIdx.begin(), shuffleIdx.end(), rnd);

	for (i = 1; i <= itersNum; ++i) {
		//DataUtil::random_select(trainData, trainLabel, dataPtr, labelPtr, batchSize);
		DataUtil::ordered_select(shuffleIdx, (i - 1) % iterPerEpoch,
			trainData, trainLabel, dataPtr, labelPtr, batchSize);

		network.computeGradient(dataPtr, labelPtr);
		network.updateNetwork(optimizer);

		cout << fixed;
		if (i % iterPerEpoch == 0) {
			cout << "\rComplete : 100.0" << " %";
			cout << setprecision(6);
			//network.updatePerEpoch(iterPerEpoch);
			float train_acc = network.getAccuracy(trainData, trainLabel);
			float test_acc = network.getAccuracy(testData, testLabel);
			float loss = network.getLoss(dataPtr, labelPtr);
			cout << "\rEpoch " << setfill('0') << setw(2) << i / iterPerEpoch << " // ";
			cout << "Train Acc. : " << train_acc << " // ";
			cout << "Test Acc. : " << test_acc << " // ";
			cout << "Loss : " << loss << endl;

			//optimizer.updateLR();
			shuffle(shuffleIdx.begin(), shuffleIdx.end(), rnd);
			optimizer.updateEpoch(i / iterPerEpoch);
		}
		else {
			cout << setprecision(2);
			float t = static_cast<float>(i % iterPerEpoch);
			float rate = t / static_cast<float>(iterPerEpoch) * 100.0f;
			cout << "\rComplete : " << rate << " %";
		}
	}

	cout << "============================= Final ================================" << endl;
	cout << fixed;
	cout << setprecision(6);
	float train_acc = network.getAccuracy(trainData, trainLabel);
	float test_acc = network.getAccuracy(testData, testLabel);
	float loss = network.getLoss(dataPtr, labelPtr);
	cout << "Epoch " << setfill('0') << setw(2) << i / iterPerEpoch + 1 << " // ";
	cout << "Train Acc. : " << train_acc << " // ";
	cout << "Test Acc. : " << test_acc << " // ";
	cout << "Loss : " << loss << endl;

	///////////////////////////////////////////////////////////////////////////////////////////////////

	//MultiLayerNetwork network;
	//int batchSize = 200;
	//int iterPerEpoch = trainNum / batchSize;
	//int itersNum = 10000;
	//vector<int> layerSize = { 784, 100, 100, 100, 100, 100, 10 };
	//Adam optimizer;
	//optimizer.initParams();
	//network.init(layerSize, batchSize, optimizer, 0.1f, 0.01f);

	//int i = 0;

	//for (i = 0; i < itersNum; ++i) {
	//	DataUtil::random_select(trainData, trainLabel, dataPtr, labelPtr, batchSize);
	//	network.computeGradient(dataPtr, labelPtr);
	//	network.updateNetwork(optimizer);
	//	cout << fixed;
	//	cout << setprecision(6);
	//	if (i % iterPerEpoch == 0) {
	//		float train_acc = network.getAccuracy(trainData, trainLabel);
	//		float test_acc = network.getAccuracy(testData, testLabel);
	//		float loss = network.getLoss(dataPtr, labelPtr);
	//		cout << "Epoch " << setfill('0') << setw(2) << i / iterPerEpoch << " // ";
	//		cout << "Train Acc. : " << train_acc << " // ";
	//		cout << "Test Acc. : " << test_acc << " // ";
	//		cout << "Loss : " << loss << endl;
	//	}
	//}
	//cout << "============================= Final ================================" << endl;
	//cout << fixed;
	//cout << setprecision(6);
	//float train_acc = network.getAccuracy(trainData, trainLabel);
	//float test_acc = network.getAccuracy(testData, testLabel);
	//float loss = network.getLoss(dataPtr, labelPtr);
	//cout << "Epoch " << setfill('0') << setw(2) << i / iterPerEpoch + 1 << " // ";
	//cout << "Train Acc. : " << train_acc << " // ";
	//cout << "Test Acc. : " << test_acc << " // ";
	//cout << "Loss : " << loss << endl;

	//vector<vector<pair<int, int>>> fwdMap;
	//vector<vector<pair<int, int>>> dxMap;
	//vector<vector<pair<int, int>>> dwMap;
	//LayerSize inputSize(5, 5, 3);
	//LayerSize filterSize(3, 3, 3);
	//LayerSize doutSize(5, 5, 10);

	//NumUtil::create_forward_map(inputSize, filterSize, 1, 1, fwdMap);
	//NumUtil::create_dx_map(inputSize, doutSize, filterSize, 1, 1, dxMap);
	//NumUtil::create_dw_map(inputSize, doutSize, 1, 1, dwMap);
	//int aaa;
	//aaa = 1;
	return 0;
}