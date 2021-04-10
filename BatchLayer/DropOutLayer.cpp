#include <iostream>
#include "DropOutLayer.h"
#include "../Core/NumUtil.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>

DropOutLayer::DropOutLayer() : BatchLayer(LayerType::DropOut), 
							   dropRatio(0.0f), fwdMultiple(false) {
	
}

DropOutLayer::~DropOutLayer() {

}

void DropOutLayer::init(LayerInfo& info) {
	int i = 0;
	outSize = info.getOutputSize();// out_size;
	inSize = info.getInputSize();// input_size;
	batchSize = info.getBatchSize();// batch_size;

	for (i = 0; i < batchSize; ++i) {
		Matrix _out(outSize);
		Matrix _dout(outSize);
		Matrix _mask(outSize);
		batchOut.push_back(_out);
		batchDout.push_back(_dout);
		mask.push_back(_mask);
	}

	for (i = 0; i < batchSize; ++i) {
		batchOutPtr.push_back(&batchOut[i]);
		batchDoutPtr.push_back(&batchDout[i]);
	}

	out.init(outSize);
	setDropOption(info.getDropRatio(), info.getIsFwdMultiple());
}

void DropOutLayer::setDropOption(float drop_ratio, bool isFwdMultiple) {
	dropRatio = drop_ratio;
	fwdMultiple = isFwdMultiple;
	//revDropRatio = 1.0f / (1.0f - dropRatio);
}

void DropOutLayer::setRandomFlag() {
	int length = outSize.getLen();// row * col * depth;
	int i = 0, j = 0;
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (i = 0; i < batchSize; ++i) {
		std::random_device engine;
		std::mt19937_64 generator(engine());
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		float *mData = mask[i].getData();
		for (j = 0; j < length; ++j) {
			mData[j] = dist(generator) > dropRatio ? 1.0f : 0.0f;
		}
	}
}

void DropOutLayer::forward(Matrix& x) {
	if (fwdMultiple) {
		out = x * (1.0f - dropRatio);
	}
	else {
		out = x;
	}
}

void DropOutLayer::batchForward(vector<Matrix*>& x) {
	setRandomFlag();
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		batchOut[i] = *x[i] * mask[i];
		//batchOut[i] *= revDropRatio;
	}
}

void DropOutLayer::batchBackward(vector<Matrix*>& d) {
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		batchDout[i] = *d[i] * mask[i];
		//batchDout[i] *= revDropRatio;
	}
}