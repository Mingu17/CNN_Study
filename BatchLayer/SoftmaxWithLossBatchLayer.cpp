#include <iostream>
#include "SoftmaxWithLossBatchLayer.h"
#include "../Core/NumUtil.h"
#ifdef _OPENMP
#include <omp.h>
#endif

SoftmaxWithLossBatchLayer::SoftmaxWithLossBatchLayer() : BatchLayer(LayerType::SoftmaxWithLoss) {

}

SoftmaxWithLossBatchLayer::~SoftmaxWithLossBatchLayer() {

}

void SoftmaxWithLossBatchLayer::init(LayerInfo& info) {
	int i = 0;
	outSize = info.getOutputSize();// out_size;
	inSize = info.getInputSize();// input_size;
	batchSize = info.getBatchSize();// batch_size;

	for (int i = 0; i < batchSize; ++i) {
		Matrix _out(outSize);
		Matrix _dout(outSize);
		batchOut.push_back(_out);
		batchDout.push_back(_dout);
	}

	for (int i = 0; i < batchSize; ++i) {
		batchOutPtr.push_back(&batchOut[i]);
		batchDoutPtr.push_back(&batchDout[i]);
	}

	out.init(outSize);
	loss = 0.0f;
}

void SoftmaxWithLossBatchLayer::forward(Matrix& x) {
	NumUtil::softmax(x, out);
}

void SoftmaxWithLossBatchLayer::batchForward(vector<Matrix*>& x) {
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		NumUtil::softmax(*x[i], batchOut[i]);
	}
}

void SoftmaxWithLossBatchLayer::batchBackward(vector<Matrix*>& d) {
	float revBatchCount = 1.0f / static_cast<float>(d.size());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		batchDout[i] = (batchOut[i] - (*d[i])) * revBatchCount;
	}
}

void SoftmaxWithLossBatchLayer::computeLoss(vector<Matrix*>& t) {
	loss = NumUtil::cross_entropy_error(batchOutPtr, t);
}

float SoftmaxWithLossBatchLayer::getLoss() {
	return loss;
}