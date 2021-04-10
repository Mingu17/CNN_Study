#pragma once

#include "../Layer/BatchLayer.h"

class ActivateBatchLayer : public BatchLayer
{
public:
	ActivateBatchLayer();
	virtual ~ActivateBatchLayer();
	// void init(LayerSize& input_size, LayerSize& out_size, LayerInfo& info);
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& d);

protected:
	void setActivateType(ActivateType _actType);

protected:
	ActivateType actType;
};