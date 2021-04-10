#include <iostream>
#include "MaxPoolingBatchLayer.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../Core/NumUtil.h"

MaxPoolingBatchLayer::MaxPoolingBatchLayer() : BatchLayer(LayerType::Pooling) {

}

MaxPoolingBatchLayer::~MaxPoolingBatchLayer() {

}

void MaxPoolingBatchLayer::init(LayerInfo& info) {
	int i = 0;
	inSize = info.getInputSize();// input_size;
	poolingSize = info.getPoolingSize();// pooling_size;
	stride = poolingSize.getCol(); //юс╫ц

	int out_h = 1 + (inSize.getRow() - poolingSize.getRow()) / stride;
	int out_w = 1 + (inSize.getCol() - poolingSize.getCol()) / stride;
	
	bool isExtend = false;

	if (inSize.getRow() % poolingSize.getRow() != 0) {
		out_h++;
		isExtend = true;
	}
	if (inSize.getCol() % poolingSize.getCol() != 0) out_w++;

	outSize.init(out_h, out_w, inSize.getDepth());
	batchSize = info.getBatchSize();

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
	NumUtil::create_pooling_map(inSize, poolingSize, stride, fwdComputeMap, isExtend);

	for (i = 0; i < batchSize; ++i) {
		int len = batchOut[i].getTotalLen();
		vector<int> maxLoc;
		maxLoc.resize(len);
		fwdMaxLoc.push_back(maxLoc);
	}
}

void MaxPoolingBatchLayer::forward(Matrix& x) {
	int depthLen = x.getDepthLen();
#ifdef _OPENMP
	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
	for (int i = 0; i < depthLen; ++i) {
		float *pInput = x.getPlaneData(i);
		float *pOut = out.getPlaneData(i);
		int len = static_cast<int>(fwdComputeMap.size());

		for (int j = 0; j < len; j++) {
			int cmdLen = static_cast<int>(fwdComputeMap[j].size());
			vector<int>& cmd = ref(fwdComputeMap[j]);
			float maxVal = -FLT_MAX;

			for (int k = 0; k < cmdLen; ++k) {
				if (maxVal < pInput[cmd[k]]) {
					maxVal = pInput[cmd[k]];
				}
			}
			pOut[j] = maxVal;
		}
	}
}

void MaxPoolingBatchLayer::batchForward(vector<Matrix*>& x) {
	int len = static_cast<int>(fwdComputeMap.size());

	for (int b = 0; b < batchSize; ++b) {
		int depthLen = x[b]->getDepthLen();
		int outLen = batchOut[b].getTotalLen();
		//int inIdx = 0;
		//int outIdx = 0;
		vector<int>& maxLoc = ref(fwdMaxLoc[b]);

#ifdef _OPENMP
		#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
		for (int i = 0; i < depthLen; ++i) {
			float *pInput = x[b]->getPlaneData(i);
			float *pOut = batchOut[b].getPlaneData(i);
			int inIdx = i * x[b]->getLenPerLayer();
			int outIdx = i * batchOut[b].getLenPerLayer();

			for (int j = 0; j < len; ++j) {
				int cmdLen = static_cast<int>(fwdComputeMap[j].size());
				vector<int>& cmd = ref(fwdComputeMap[j]);
				float maxVal = -FLT_MAX;
				int maxIdx = -1;
				for (int k = 0; k < cmdLen; ++k) {
					if (maxVal < pInput[cmd[k]]) {
						maxVal = pInput[cmd[k]];
						maxIdx = cmd[k];
					}
				}
				pOut[j] = maxVal;
				maxLoc[outIdx] = inIdx + maxIdx;
				outIdx++;
			}
		}
	}
}

void MaxPoolingBatchLayer::batchBackward(vector<Matrix*>& dout) {
	for (int b = 0; b < batchSize; ++b) {
		float *pDout = dout[b]->getData();
		batchDout[b].setZero();
		float *pDx = batchDout[b].getData();

		int len = static_cast<int>(fwdMaxLoc[b].size()); //length of dout
		vector<int>& maxLoc = ref(fwdMaxLoc[b]);

#ifdef _OPENMP
		#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
		for (int i = 0; i < len; ++i) {
			pDx[maxLoc[i]] = pDout[i];
		}
	}
}