#pragma once

#include "../Layer/Layer.h"
#include "../Optimizer/Optimizer.h"

class MultiLayerNetwork
{
public:
	MultiLayerNetwork();
	virtual ~MultiLayerNetwork();

	void init(vector<int>& layer_size, int batch_size, 
		Optimizer& optimizer, float learning_rate, float weight_factor);
	void computeGradient(vector<Matrix*>& x, vector<Matrix*>& t);
	float getLoss(vector<Matrix*>& x, vector<Matrix*>& t);
	float getAccuracy(vector<Matrix>& x, vector<Matrix>& t);

	void updateNetwork(Optimizer& optimizer);

private:
	vector<Layer*> network;
	vector<int> layerSize;
	float learningRate;
	int batchSize;
};