#include <iostream>
#include "ConvolutionNetwork.h"
#include "../BatchLayer/BatchLayerSet.h"
#include "../Core/NumUtil.h"
#include "../Core/DataUtil.h"

ConvolutionNetwork::ConvolutionNetwork() {

}

ConvolutionNetwork::~ConvolutionNetwork() {

}

void ConvolutionNetwork::init(vector<LayerInfo>& info, Optimizer& optimizer, 
	float learning_rate, float weight_factor, float weight_decay) {
#ifdef _OPENMP
	NumUtil::OMP_THREAD_NUM = 8;
#endif
	int i = 0, totalLayerSize = static_cast<int>(info.size());
	
	batchSize = info[0].getBatchSize();
	learningRate = learning_rate;
	weightDecayLambda = weight_decay;

	for (int i = 0; i < totalLayerSize; ++i) {
		Layer *layer;
		DataUtil::create_layer(info[i], &layer);
		layer->init(info[i]);
		if (static_cast<int>(layer->getLayerType()) & UPDATABLE_LAYER) {
			UpdateInterface *ui = dynamic_cast<UpdateInterface*>(layer);
			ui->setWeightDecayLambda(weightDecayLambda);
		}
		network.push_back(layer);
	}

	optimizer.initLayer(network);
}

void ConvolutionNetwork::computeGradient(vector<Matrix*>& x, vector<Matrix*>& t) {
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

float ConvolutionNetwork::getLoss(vector<Matrix*>& x, vector<Matrix*>& t) {
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

	float weightDecay = 0.0f;
	for (int i = 0; i < nSize; ++i) {
		LayerType type = network[i]->getLayerType();
		if (static_cast<int>(type) & UPDATABLE_LAYER) {
			UpdateInterface *ui = dynamic_cast<UpdateInterface*>(network[i]);
			weightDecay += ui->weightSquaredSum();
		}
	}
	return lastLayer->getLoss() + (weightDecay * 0.5f);
}

float ConvolutionNetwork::getAccuracy(vector<Matrix>& x, vector<Matrix>& t) {
	int nSize = static_cast<int>(network.size());
	int lastId = nSize - 1;
	float accSum = 0.0f;
	int xSize = static_cast<int>(x.size());

	for (int i = 0; i < xSize; ++i) {
		network[0]->forward(x[i]);
		for (int j = 1; j < nSize; ++j) {
			network[j]->forward(network[j - 1]->getOut());
		}
		accSum += (NumUtil::isCorrect1D(network[lastId]->getOut(), t[i]) ? 1.0f : 0.0f);
	}

	return accSum / static_cast<float>(xSize);
}

void ConvolutionNetwork::updateNetwork(Optimizer& optimizer) {
	optimizer.update(network);
}