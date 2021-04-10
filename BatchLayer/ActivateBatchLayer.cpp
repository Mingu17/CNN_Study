#include <iostream>
#include "ActivateBatchLayer.h"
#include "../Core/NumUtil.h"
#ifdef _OPENMP
#include <omp.h>
#endif

ActivateBatchLayer::ActivateBatchLayer() : BatchLayer(LayerType::Relu) {

}

ActivateBatchLayer::~ActivateBatchLayer() {

}

void ActivateBatchLayer::init(LayerInfo& info) {
	int i = 0;
	
	inSize = info.getInputSize();// input_size;
	outSize = info.getOutputSize();// out_size;
	batchSize = info.getBatchSize();// batch_size;

	for (i = 0; i < batchSize; ++i) {
		Matrix _out(outSize);
		Matrix _dout(inSize);
		batchOut.push_back(_out);
		batchDout.push_back(_dout);
	}

	for (i = 0; i < batchSize; ++i) {
		batchOutPtr.push_back(&batchOut[i]);
		batchDoutPtr.push_back(&batchDout[i]);
	}
	out.init(outSize);
	setActivateType(info.getActivateType());
}

void ActivateBatchLayer::setActivateType(ActivateType _actType) {
	if (_actType == ActivateType::None) {
		actType = ActivateType::Relu;
	}
	else {
		actType = _actType;
	}
}

void ActivateBatchLayer::forward(Matrix& x) {
	NumUtil::activate(x, out, actType);
}

void ActivateBatchLayer::batchForward(vector<Matrix*>& x) {

#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		NumUtil::activate(*x[i], batchOut[i], actType);
	}
}

void ActivateBatchLayer::batchBackward(vector<Matrix*>& d) {
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		NumUtil::deactivate(*d[i], batchOut[i], batchDout[i], actType);
	}
}