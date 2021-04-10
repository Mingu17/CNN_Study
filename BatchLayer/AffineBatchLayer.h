#pragma once

#include "../Layer/BatchLayer.h"
#include "../Layer/UpdateInterface.h"

class AffineBatchLayer : public BatchLayer, public UpdateInterface
{
public:
	AffineBatchLayer();
	virtual ~AffineBatchLayer();
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& d);
	virtual float weightSquaredSum();

protected:
	Matrix weight;
	Matrix bias;
	Matrix dWeight;
	Matrix dBias;

	vector<Matrix> input;
};