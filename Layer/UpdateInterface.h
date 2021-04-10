#pragma once

#include "../Core/Matrix.h"
#include <vector>
using std::vector;

class UpdateInterface
{
public:
	UpdateInterface() {}
	virtual ~UpdateInterface() {}
	virtual vector<Matrix*>& getParams() { return params; }
	virtual vector<Matrix*>& getGrads() { return grads; }
	virtual float weightSquaredSum() { return 0.0f; }

	virtual void setWeightDecayLambda(float lambda) {
		weightDecayLambda = lambda;
	}

	//함수는 남겨두되 필요한 경우 사용
	virtual void updateEpoch(int epochSize) {

	}

protected:
	vector<Matrix*> params;
	vector<Matrix*> grads;
	float weightDecayLambda;
};
