#pragma once

#include "../Layer/BatchLayer.h"
#include "../Layer/UpdateInterface.h"

class BatchNormLayer : public BatchLayer, public UpdateInterface
{
public:
	BatchNormLayer();
	virtual ~BatchNormLayer();
	virtual void init(LayerInfo& info);
	virtual void forward(Matrix& x);
	virtual void batchForward(vector<Matrix*>& x);
	virtual void batchBackward(vector<Matrix*>& d);

protected:
	void setMomentum(float ratio);

protected:
	Matrix gamma;
	Matrix beta;
	Matrix runningMean;
	Matrix runningVar;
	Matrix invRunningStd; //1.0f / runningVar.sqrt(1e-7f)
	vector<Matrix> xc;
	vector<Matrix> dxc;
	vector<Matrix> xn;
	Matrix std;
	Matrix dGamma;
	Matrix dBeta;

	float momentum;

	//Matrix runningGamma;
	//Matrix runningBeta;
};