#pragma once

#include "../Layer/BatchLayer.h"

class SoftmaxWithLossBatchLayer : public BatchLayer
{
public:
	SoftmaxWithLossBatchLayer();
	virtual ~SoftmaxWithLossBatchLayer();
	//virtual void init(LayerSize& input_size, LayerSize& out_size, LayerInfo& info);
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& d);

	void computeLoss(vector<Matrix*>& t);
	float getLoss();

protected:
	float loss;
};