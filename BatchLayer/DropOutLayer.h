#pragma once

#include "../Layer/BatchLayer.h"

class DropOutLayer : public BatchLayer
{
public:
	DropOutLayer();
	virtual ~DropOutLayer();
	//virtual void init(LayerSize& input_size, LayerSize& out_size, LayerInfo& info);
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& d);

protected:
	void setDropOption(float drop_ratio, bool isFwdMultiple = true);
	void setRandomFlag();

protected:
	float dropRatio;
	//float revDropRatio;
	bool fwdMultiple;
	vector<Matrix> mask;
	vector<float> realRatio;
};