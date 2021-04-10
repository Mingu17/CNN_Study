#pragma once

#include "Layer.h"
#include <vector>
using std::vector;

class BatchLayer : public Layer
{
public:
	BatchLayer(LayerType type) : Layer(type) {}
	virtual void batchForward(vector<Matrix*>& x) = 0;
	virtual void batchBackward(vector<Matrix*>& d) = 0;

	vector<Matrix*>& getBatchOutPtr() { return batchOutPtr; }
	vector<Matrix*>& getBatchDoutPtr() { return batchDoutPtr; }

protected:
	vector<Matrix> batchOut;
	vector<Matrix> batchDout;
	vector<Matrix*> batchOutPtr;
	vector<Matrix*> batchDoutPtr;

	int batchSize;
};