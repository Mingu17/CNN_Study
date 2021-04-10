#pragma once

#include "../Layer/Layer.h"
#include "../Optimizer/Optimizer.h"

class ConvolutionNetwork
{
public:
	ConvolutionNetwork();
	virtual ~ConvolutionNetwork();

	void init(vector<LayerInfo>& info, Optimizer& optimizer, 
		float learning_rate, float weight_factor,
		float weight_decay = 0.0f);

	void computeGradient(vector<Matrix*>& x, vector<Matrix*>& t);
	float getLoss(vector<Matrix*>& x, vector<Matrix*>& t);
	float getAccuracy(vector<Matrix>& x, vector<Matrix>& t);

	void updateNetwork(Optimizer& optimizer);
	
private:
	vector<Layer*> network;
	vector<int> layerSize;
	float learningRate;
	int batchSize;
	float weightDecayLambda;
};