#include <iostream>
#include "MultiLayerNetwork.h"
#include "../BatchLayer/BatchLayerSet.h"
#include "../Core/NumUtil.h"

MultiLayerNetwork::MultiLayerNetwork() {

}

MultiLayerNetwork::~MultiLayerNetwork() {

}

void MultiLayerNetwork::init(vector<int>& layer_size, int batch_size, Optimizer& optimizer,
							 float learning_rate, float weight_factor) {
#ifdef _OPENMP
	NumUtil::OMP_THREAD_NUM = 4;
#endif
	int i = 0, totalLayerSize = static_cast<int>(layer_size.size());
	for (i = 0; i < totalLayerSize; ++i) {
		layerSize.push_back(layer_size[i]);
	}
	batchSize = batch_size;
	learningRate = learning_rate;
	
	LayerSize inSize, outSize;
	LayerInfo info(batchSize);

	for (i = 1; i < totalLayerSize; ++i) {
		AffineBatchLayer *affine = new AffineBatchLayer();
		inSize.init(layerSize[i - 1], 1);
		outSize.init(layerSize[i], 1);
		
		info.setInputSize(inSize);
		info.setOutputSize(outSize);
		info.setWeightInfo(WeightInitType::He, weight_factor);
		
		affine->init(info);
		network.push_back(affine);

		info.setInputSize(outSize);
		info.setOutputSize(outSize);

		if (i < totalLayerSize - 1) {
			BatchNormLayer *bn = new BatchNormLayer();
			info.setMomentum(0.9f);
			bn->init(info);

			ActivateBatchLayer *act = new ActivateBatchLayer();
			info.setActivateType(ActivateType::Relu);
			act->init(info);


			DropOutLayer *drop = new DropOutLayer();
			info.setDropOutInfo(0.5f);
			drop->init(info);

			network.push_back(bn);
			network.push_back(act);
			network.push_back(drop);
		}
		else {
			SoftmaxWithLossBatchLayer *softmax = new SoftmaxWithLossBatchLayer();
			softmax->init(info);
			network.push_back(softmax);
		}
	}

	optimizer.initLayer(network);
}

void MultiLayerNetwork::computeGradient(vector<Matrix*>& x, vector<Matrix*>& t) {
	float loss = getLoss(x, t);
	int nSize = static_cast<int>(network.size());

	BatchLayer *nowLayer, *prevLayer;
	nowLayer = dynamic_cast<BatchLayer*>(network[nSize - 1]);
	nowLayer->batchBackward(t);
	for (int i = nSize - 2; i >= 0; --i) {
		prevLayer = nowLayer;
		nowLayer = dynamic_cast<BatchLayer*>(network[i]);
		nowLayer->batchBackward(prevLayer->getBatchDoutPtr());
	}
}

float MultiLayerNetwork::getLoss(vector<Matrix*>& x, vector<Matrix*>& t) {
	int nSize = static_cast<int>(network.size());
	BatchLayer *nowLayer, *prevLayer;
	nowLayer = dynamic_cast<BatchLayer*>(network[0]);
	nowLayer->batchForward(x);
	for (int i = 1; i < nSize; ++i) {
		prevLayer = nowLayer;
		nowLayer = dynamic_cast<BatchLayer*>(network[i]);
		nowLayer->batchForward(prevLayer->getBatchOutPtr());
	}
	SoftmaxWithLossBatchLayer *lastLayer = dynamic_cast<SoftmaxWithLossBatchLayer*>(network[nSize - 1]);
	lastLayer->computeLoss(t);
	return lastLayer->getLoss();
}

float MultiLayerNetwork::getAccuracy(vector<Matrix>& x, vector<Matrix>& t) {
	int nSize = static_cast<int>(network.size());
	int lastId = nSize - 1;
	float accSum = 0.0f;
	int xSize = static_cast<int>(x.size());

	for (int i = 0; i < xSize; ++i) {
		network[0]->forward(x[i]);
		for (int j = 1; j < nSize; j++) {
			network[j]->forward(network[j - 1]->getOut());
		}
		accSum += (NumUtil::isCorrect1D(network[lastId]->getOut(), t[i]) ? 1.0f : 0.0f);
	}

	return accSum / static_cast<float>(xSize);
}

void MultiLayerNetwork::updateNetwork(Optimizer& optimizer) {
	optimizer.update(network);
}