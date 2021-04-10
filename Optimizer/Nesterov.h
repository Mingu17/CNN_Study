#pragma once

#include "Optimizer.h"

class Nesterov : public Optimizer
{
public:
	Nesterov();
	virtual ~Nesterov();
	virtual void initLayer(vector<Layer*>& network);
	virtual void initParams(float learning_rate = 0.01f, float _momentum = 0.9f, float not_use1 = 0.0f);
	virtual void update(vector<Layer*>& network);
protected:
	float momentum;
};