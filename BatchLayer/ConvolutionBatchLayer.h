#pragma once

#include "../Layer/BatchLayer.h"
#include "../Layer/UpdateInterface.h"
#include <utility>
using std::pair;

class ConvolutionBatchLayer : public BatchLayer, public UpdateInterface
{
public:
	ConvolutionBatchLayer();
	virtual ~ConvolutionBatchLayer();
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& dout);
	virtual float weightSquaredSum();

protected:
	void setConvInfo(LayerInfo& info);
	void setComputeMap();

protected:
	//Per filter : width x height x input depth (channel)
	vector<Matrix> filter;
	vector<Matrix> dFilter;
	//bias : count of filters x 1
	Matrix bias;
	Matrix dBias;
	
	//order : input, filter (index)
	vector<vector<pair<int, int>>> fwdComputeMap;
	//order : dout_rev_pad, w_R
	vector<vector<pair<int, int>>> doutComputeMap;
	//order : input_pad, dout
	vector<vector<pair<int, int>>> dwComputeMap;

	vector<Matrix> input;

	LayerSize filterSize;
	int filterCount;
	int stride;
	int padding;
};