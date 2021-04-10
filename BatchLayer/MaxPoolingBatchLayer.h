#pragma once

#include "../Layer/BatchLayer.h"

class MaxPoolingBatchLayer : public BatchLayer
{
public:
	MaxPoolingBatchLayer();
	virtual ~MaxPoolingBatchLayer();
	//virtual void init(LayerSize& input_size, LayerSize& pooling_size, LayerInfo& info);
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& dout);

protected:
	LayerSize poolingSize;
	vector<vector<int>> fwdComputeMap; //per plane cell
	vector<vector<int>> fwdMaxLoc; //per batch
	int stride;
};