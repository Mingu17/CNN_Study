#pragma once

#include "Optimizer.h"

class Adam : public Optimizer
{
public:
	Adam();
	virtual ~Adam();
	virtual void initLayer(vector<Layer*>& network);
	virtual void initParams(float learning_rate = 0.001f, float beta_1 = 0.9f, float beta_2 = 0.999f);
	virtual void update(vector<Layer*>& network);
	
protected:
	float beta1;
	float beta2;
	float iteration;
	map<int, vector<Matrix>> params2;

	float realLR;
};