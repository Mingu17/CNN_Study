#include <iostream>
#include "BatchNormLayer.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../Core/NumUtil.h"

BatchNormLayer::BatchNormLayer() : BatchLayer(LayerType::BatchNorm), UpdateInterface() {

}

BatchNormLayer::~BatchNormLayer() {

}

void BatchNormLayer::init(LayerInfo& info) {
	int i = 0;
	outSize = info.getOutputSize();// out_size;
	inSize = info.getInputSize();// input_size;
	batchSize = info.getBatchSize();// batch_size;
	
	for (i = 0; i < batchSize; ++i) {
		Matrix _out(outSize);
		Matrix _dout(outSize);
		Matrix _xc(outSize);
		Matrix _xn(outSize);
		Matrix _dxc(outSize);

		batchOut.push_back(_out);
		batchDout.push_back(_dout);
		xc.push_back(_xc);
		xn.push_back(_xn);
		dxc.push_back(_dxc);
	}

	for (i = 0; i < batchSize; i++) {
		batchOutPtr.push_back(&batchOut[i]);
		batchDoutPtr.push_back(&batchDout[i]);
	}

	out.init(outSize);
	gamma.init(outSize, 1.0f);
	beta.init(outSize);
	runningMean.init(outSize);
	runningVar.init(outSize);
	invRunningStd.init(outSize);
	std.init(outSize);
	dGamma.init(outSize);
	dBeta.init(outSize);

	params.push_back(&gamma);
	params.push_back(&beta);
	grads.push_back(&dGamma);
	grads.push_back(&dBeta);

	setMomentum(info.getMomentum());

	//runningGamma.init(outSize, 1.0);
	//runningBeta.init(outSize);
}

void BatchNormLayer::setMomentum(float ratio) {
	momentum = ratio;
}

void BatchNormLayer::forward(Matrix& x) {
	Matrix _xc = x - runningMean;
	//Matrix _xn = _xc / runningVar.sqrt(10e-7f);
	Matrix _xn = _xc * invRunningStd;
	out = _xn * gamma + beta;
	//out = x * runningGamma + runningBeta;
}

void BatchNormLayer::batchForward(vector<Matrix*>& x) {
	//int i = 0;
	float revSize = 1.0f / static_cast<float>(batchSize);
	Matrix mu(outSize), var(outSize);
#ifdef _OPENMP
	NumUtil::get_mean_omp(x, mu);
#else
	NumUtil::get_mean(x, mu);
#endif

#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		xc[i] = *x[i] - mu;
	}
#ifdef _OPENMP
	vector<Matrix> varSet;
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		Matrix v(outSize, 0.0f);
		varSet.push_back(v);
	}
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		Matrix _var = xc[i].squared();
#ifdef _OPENMP
		int id = omp_get_thread_num();
		varSet[id] += _var;
#else
		var += _var;
#endif
	}

#ifdef _OPENMP
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		var += varSet[i];
	}
#endif

	var *= revSize;
	std = var.sqrt(10e-7f);
	Matrix invStd = std.inverse();

#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		//xn[i] = xc[i] / std;
		xn[i] = xc[i] * invStd;
	}

	runningMean *= momentum;
	runningMean += (mu * (1.0f - momentum));
	runningVar *= momentum;
	runningVar += (var * (1.0f - momentum));
	invRunningStd = runningVar.sqrt(10e-7f).inverse();

	//runningMean += mu;
	//runningVar += var;

#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		batchOut[i] = xn[i] * gamma + beta;
	}
}

void BatchNormLayer::batchBackward(vector<Matrix*>& d) {
	//int i = 0;
#ifdef _OPENMP
	NumUtil::get_sum_omp(d, dBeta);
#else
	NumUtil::get_sum(d, dBeta);
#endif
	//vector<Matrix> dxn;
	//vector<Matrix> dxc;
	Matrix invStd = std.inverse();
	Matrix squaredStd = std.squared();
	Matrix invSquaredStd = squaredStd.inverse();
	Matrix dstd(outSize), dmu(outSize), dvar(outSize);
	float fSizeConst = 2.0f / static_cast<float>(batchSize);
	float revSizeConst = 1.0f / static_cast<float>(batchSize);

	dGamma.setZero();
#ifdef _OPENMP
	vector<Matrix> dstdSet;
	vector<Matrix> dGammaSet;

	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		Matrix ds(outSize, 0.0f);
		Matrix dg(outSize, 0.0f);
		dstdSet.push_back(ds);
		dGammaSet.push_back(dg);
	}
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		Matrix _dxn = gamma * (*d[i]);
		//Matrix _dxc = _dxn / std;
		//dstd += (_dxn * xc[i]) / squaredStd;
		//dstd += (_dxn * xc[i]) * invSquaredStd;
		//dxn.push_back(_dxn);
		//dxc.push_back(_dxc);
		dxc[i] = _dxn * invStd;// / std; //_dxc;
#ifdef _OPENMP
		int id = omp_get_thread_num();
		dGammaSet[id] += (xn[i] * (*d[i]));
		dstdSet[id] += (_dxn * xc[i]) * invSquaredStd;
#else
		dGamma += (xn[i] * (*d[i]));
		dstd += (_dxn * xc[i]) * invSquaredStd;
#endif
		
	}
#ifdef _OPENMP
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		dGamma += dGammaSet[i];
		dstd += dstdSet[i];
	}
#endif
	
	dstd *= (-1.0f);
	dvar = (dstd * 0.5f) * invStd;// / std;

#ifdef _OPENMP
	vector<Matrix> dmuSet;
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		Matrix dm(outSize, 0.0f);
		dmuSet.push_back(dm);
	}
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		//Matrix _dmu = dxc[i] + ((xc[i] * dvar) * fSizeConst);
		dxc[i] += ((xc[i] * dvar) * fSizeConst);
#ifdef _OPENMP
		int id = omp_get_thread_num();
		dmuSet[id] += dxc[i];// _dmu;
#else
		dmu += dxc[i];// _dmu; // dxc[i] + ((xc[i] * dvar) * fSizeConst);
#endif
	}

#ifdef _OPENMP
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		dmu += dmuSet[i];
	}
#endif

	dmu *= revSizeConst;
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < batchSize; ++i) {
		batchDout[i] = dxc[i] - dmu;
	}
}

//void BatchNormLayer::updatePerEpoch(int epoch) {
//	float count = static_cast<float>(epoch);
//	float revCount = 1.0f / count;
//	float varRevCount = revCount * (count / (count - 1.0f));
//	int i = 0;
//
//	runningMean *= revCount;
//	runningVar *= varRevCount;
//
//	runningGamma = runningVar.sqrt(1e-07f).inverse() * gamma;
//	runningBeta = beta - (runningGamma * runningMean);
//
//	runningMean.setZero();
//	runningVar.setZero();
//}