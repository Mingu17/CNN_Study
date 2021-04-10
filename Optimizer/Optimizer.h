#pragma once

#include "../Layer/Layer.h"
#include "../Core/Matrix.h"
#include <map>
#include <vector>
using std::map;
using std::vector;

class Optimizer 
{
public:
	Optimizer() {}

	virtual void initLayer(vector<Layer*>& network) = 0;
	virtual void initParams(float learning_rate, float, float) = 0;
	virtual void update(vector<Layer*>& network) = 0;

	void updateEpoch(int epochCount) {
		decayLearningRate = std::pow(0.95f, epochCount) * learningRate;
	}

protected:
	float learningRate;
	float decayLearningRate;
	map<int, vector<Matrix>> params;
};