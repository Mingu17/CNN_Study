#pragma once

#include "Optimizer.h"

class RMSProp : public Optimizer
{
public:
	RMSProp();
	virtual ~RMSProp();
	virtual void initLayer(vector<Layer*>& network);
	virtual void initParams(float learning_rate, float decay_rate = 0.99f, float not_use1 = 0.0f);
	virtual void update(vector<Layer*>& network);

protected:
	float decayRate;
};