#include <iostream>
#include "ConvolutionBatchLayer.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../Core/NumUtil.h"

ConvolutionBatchLayer::ConvolutionBatchLayer() : BatchLayer(LayerType::Convolution), UpdateInterface() {
	stride = 1;
	padding = 0;
}

ConvolutionBatchLayer::~ConvolutionBatchLayer() {

}

void ConvolutionBatchLayer::init(LayerInfo& info) {
	int i = 0;
	inSize = info.getInputSize();// input_size;
	filterSize = info.getFilterSize();// filter_size;

	if (inSize.getDepth() != filterSize.getDepth()) {
		throw::std::invalid_argument("[ERROR] Depth size mismatch");
	}

	batchSize = info.getBatchSize();
	setConvInfo(info);

	//filter initialize
	for (i = 0; i < filterCount; ++i) {
		Matrix _filter(filterSize.getRow(), filterSize.getCol(), inSize.getDepth());
		Matrix _dFilter(filterSize.getRow(), filterSize.getCol(), inSize.getDepth());
		int in = filterSize.getRow() * filterSize.getCol() * inSize.getDepth();
		int out = filterSize.getRow() * filterSize.getCol() * filterCount;
		NumUtil::weight_init_conv(info.getWeightInitType(), info.getWeightFactor(), _filter, in, out);
		filter.push_back(_filter);
		dFilter.push_back(_dFilter);
	}

	int out_h = 1 + ((inSize.getRow() + 2 * padding - filterSize.getRow()) / stride);
	int out_w = 1 + ((inSize.getCol() + 2 * padding - filterSize.getCol()) / stride);
	outSize = LayerSize(out_h, out_w, filterCount);

	for (i = 0; i < batchSize; i++) {
		Matrix _out(outSize);
		Matrix _dout(inSize);
		Matrix _input(inSize);
		batchOut.push_back(_out);
		batchDout.push_back(_dout);
		input.push_back(_input);
	}

	for (i = 0; i < batchSize; ++i) {
		batchOutPtr.push_back(&batchOut[i]);
		batchDoutPtr.push_back(&batchDout[i]);
	}

	out.init(outSize);
	bias.init(filterCount, 1, 1, 0.0f);
	dBias.init(filterCount, 1, 1, 0.0f);
	setComputeMap();

	for (i = 0; i < filterCount; ++i) {
		params.push_back(&filter[i]);
		grads.push_back(&dFilter[i]);
	}
	params.push_back(&bias);
	grads.push_back(&dBias);
}

void ConvolutionBatchLayer::setConvInfo(LayerInfo& info) {
	filterCount = info.getFilterCount();
	stride = info.getStride();
	padding = info.getPadding();
}

void ConvolutionBatchLayer::setComputeMap() {
	//NumUtil::create_forward_map(inSize, filterSize, stride, padding, fwdComputeMap);
	//NumUtil::create_dx_map(inSize, outSize, filterSize, stride, padding, doutComputeMap);
	//NumUtil::create_dw_map(inSize, outSize, stride, padding, dwComputeMap);
	NumUtil::create_conv_map(inSize, filterSize, stride, padding,
		fwdComputeMap, doutComputeMap, dwComputeMap);
}

void ConvolutionBatchLayer::forward(Matrix& x) {
	float *fX = x.getData();
	int planeLen = out.getLenPerLayer();

#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int d = 0; d < out.getDepthLen(); ++d) { //filter count
		float *fOutPlane = out.getPlaneData(d);
		float *pFilter = filter[d].getData();

		for (int i = 0; i < planeLen; ++i) {
			int cmdLen = static_cast<int>(fwdComputeMap[i].size());
			float fVal = 0.0f;
			for (int j = 0; j < cmdLen; ++j) {
				int first = fwdComputeMap[i][j].first;
				int second = fwdComputeMap[i][j].second;
				fVal += (fX[first] * pFilter[second]);
			}
			fOutPlane[i] = fVal;// +bias.getElement(d);
		}
	}
}

void ConvolutionBatchLayer::batchForward(vector<Matrix*>& x) {
	for (int b = 0; b < batchSize; ++b) {
		float *fX = x[b]->getData();
		int planeLen = batchOut[b].getLenPerLayer();

#ifdef _OPENMP
		#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
		for (int d = 0; d < batchOut[b].getDepthLen(); ++d) {
			float *fOutPlane = batchOut[b].getPlaneData(d);
			float *pFilter = filter[d].getData();

			for (int i = 0; i < planeLen; ++i) {
				int cmdLen = static_cast<int>(fwdComputeMap[i].size());
				float fVal = 0.0f;

				for (int j = 0; j < cmdLen; ++j) {
					int first = fwdComputeMap[i][j].first;
					int second = fwdComputeMap[i][j].second;
					fVal += (fX[first] * pFilter[second]);
				}
				fOutPlane[i] = fVal;// +bias.getElement(d);
			}
		}
	}

#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int b = 0; b < batchSize; ++b) {
		input[b] = *x[b];
	}
}

void ConvolutionBatchLayer::batchBackward(vector<Matrix*>& dout) {
	//int n, b, d, i, j;
	//1. bias
	float revBatchSize = static_cast<float>(batchSize);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int n = 0; n < filterCount; ++n) {
		float sum = 0.0f;
		for (int b = 0; b < batchSize; ++b) {
			sum += dout[b]->getSum(n);
		}
		dBias.setElement(sum, n);
	}

	//2. dx 검증 완료
	int dxPlaneLen = batchDout[0].getLenPerLayer();
	for (int b = 0; b < batchSize; ++b) {
		batchDout[b].setZero();
		float *pDx = batchDout[b].getData();


		for (int d = 0; d < filterCount; ++d) {
			float *pDout = dout[b]->getPlaneData(d);
			float *pFilter = filter[d].getData();
			int dataLen = static_cast<int>(doutComputeMap.size());
#ifdef _OPENMP
			#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
			for (int i = 0; i < dataLen; ++i) {
				float fVal = 0.0f;
				int cmdLen = static_cast<int>(doutComputeMap[i].size());

				for (int j = 0; j < cmdLen; ++j) {
					int first = doutComputeMap[i][j].first;
					int second = doutComputeMap[i][j].second;
					fVal += (pDout[first] * pFilter[second]);
				}
				pDx[i] += fVal;
			}
		}
	}

	//3. dw
	for (int n = 0; n < filterCount; ++n) {
		dFilter[n].setZero();
		float *pDfilter = dFilter[n].getData();

		for (int b = 0; b < batchSize; ++b) {
			float *pInput = input[b].getData();
			float *pDout = dout[b]->getPlaneData(n);
			int dataLen = static_cast<int>(dwComputeMap.size());

#ifdef _OPENMP
			#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
			for (int i = 0; i < dataLen; ++i) {
				float fVal = 0.0f;
				int cmdLen = static_cast<int>(dwComputeMap[i].size());
				for (int j = 0; j < cmdLen; ++j) {
					int first = dwComputeMap[i][j].first;
					int second = dwComputeMap[i][j].second;
					fVal += (pInput[first] * pDout[second]);
				}
				pDfilter[i] += fVal;
			}
		}
		if (weightDecayLambda != 0.0f) {
			dFilter[n] += (filter[n] * weightDecayLambda);
		}
	}
}

float ConvolutionBatchLayer::weightSquaredSum() {
	float squaredSum = 0.0f;
	for (int i = 0; i < static_cast<int>(filter.size()); ++i) {
		squaredSum += filter[i].squared().getSum() * weightDecayLambda;
	}
	return squaredSum;
}