#include <iostream>
#include "AffineBatchLayer.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../Core/NumUtil.h"

AffineBatchLayer::AffineBatchLayer() : BatchLayer(LayerType::Affine), UpdateInterface() {

}

AffineBatchLayer::~AffineBatchLayer() {

}

void AffineBatchLayer::init(LayerInfo& info) {
	int i = 0, inputLen = 0;
	outSize = info.getOutputSize();// out_size;

	if (outSize.getCol() != 1 || outSize.getDepth() != 1) {
		throw std::invalid_argument("[ERROR] column and depth value is incorrect (value is 1)");
	}

	inSize = info.getInputSize();// input_size;

	batchSize = info.getBatchSize();// batch_size;
	inputLen = inSize.getLen();

	for (i = 0; i < batchSize; ++i) {
		Matrix _out(outSize);
		Matrix _dout(inSize);
		Matrix _input(inputLen, 1, 1);
		batchOut.push_back(_out);
		batchDout.push_back(_dout);
		input.push_back(_input);
	}

	for (i = 0; i < batchSize; ++i) {
		batchOutPtr.push_back(&batchOut[i]);
		batchDoutPtr.push_back(&batchDout[i]);
	}

	//(r, c, d) 입력이 들어오면 (r', 1, 1) 입력으로 바꿔야한다 r' = r * c * d
	//즉 affine layer 의 weight는 (r, r', 1) 행렬이다
	int row = outSize.getRow();
	weight.init(row, inputLen, 1);
	dWeight.init(row, inputLen, 1);
	bias.init(outSize);
	dBias.init(outSize);
	out.init(outSize);

	params.push_back(&weight);
	params.push_back(&bias);
	grads.push_back(&dWeight);
	grads.push_back(&dBias);

	NumUtil::weight_init_affine(info.getWeightInitType(), info.getWeightFactor(), weight);
}

void AffineBatchLayer::forward(Matrix& x) {
	bool isReshape = false;
	if (x.getColLen() != 1 || x.getDepthLen() != 1) {
		int xLen = x.getTotalLen();
		x.reshape(xLen, 1, 1);
		isReshape = true;
	}
	out = weight.dot(x);// +bias;

	if (isReshape) x.reshape(inSize);
}

void AffineBatchLayer::batchForward(vector<Matrix*>& x) {
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		bool isReshape = false;
		if (x[i]->getColLen() != 1 || x[i]->getDepthLen() != 1) {
			int xLen = x[i]->getTotalLen();
			x[i]->reshape(xLen, 1, 1);
			isReshape = true;
		}
		input[i] = *x[i]; // r,c,d = N,1,1
		batchOut[i] = weight.dot(*x[i]);// +bias;
		if (isReshape) x[i]->reshape(inSize);
	}
}

void AffineBatchLayer::batchBackward(vector<Matrix*>& d) {
	Matrix tw = Matrix(weight, MatConsType::Transpose);
	dWeight.setZero();
	dBias.setZero();

#ifdef _OPENMP
	vector<Matrix> dWeightSet;
	vector<Matrix> dBiasSet;
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		Matrix dw(dWeight.getTotalSize(), 0.0f);
		Matrix db(dBias.getTotalSize(), 0.0f);
		dWeightSet.push_back(dw);
		dBiasSet.push_back(db);
	}
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		Matrix tinput = Matrix(input[i], MatConsType::Transpose);
		Matrix dout = tw.dot(*d[i]);
		dout.reshape(inSize);
		batchDout[i] = dout;
		Matrix dw = d[i]->dot(tinput);
#ifdef _OPENMP
		int id = omp_get_thread_num();
		dWeightSet[id] += dw;
		dBiasSet[id] += (*d[i]);
#else
		dWeight += dw;
		dBias += (*d[i]);
#endif
	}

#ifdef _OPENMP
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		dWeight += dWeightSet[i];
		dBias += dBiasSet[i];
	}
#endif
	//Weight decay
	if (weightDecayLambda != 0.0f) {
		dWeight += (weight * weightDecayLambda);
	}
}

float AffineBatchLayer::weightSquaredSum() {
	return weight.squared().getSum() * weightDecayLambda;
}