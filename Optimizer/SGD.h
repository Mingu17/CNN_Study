#pragma once

#include "Optimizer.h"

class SGD : public Optimizer
{
public:
	SGD();
	virtual ~SGD();

	virtual void initLayer(vector<Layer*>& network);
	virtual void initParams(float learning_rate = 0.01f, float not_use1 = 0.0f, float not_use2 = 0.0f);
	virtual void update(vector<Layer*>& network);
};