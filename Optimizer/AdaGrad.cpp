#include <iostream>
#include "AdaGrad.h"
#include "../Layer/UpdateInterface.h"
#include <utility>
#include "../Core/NumUtil.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using std::pair;

AdaGrad::AdaGrad() : Optimizer() {

}

AdaGrad::~AdaGrad() {

}

void AdaGrad::initLayer(vector<Layer*>& network) {
	int size = static_cast<int>(network.size());

	for (int i = 0; i < size; ++i) {
		LayerType type = network[i]->getLayerType();
		if (static_cast<int>(type) & UPDATABLE_LAYER) {
			UpdateInterface *ui = dynamic_cast<UpdateInterface*>(network[i]);
			vector<Matrix> v;
			vector<Matrix*>& param = ui->getParams();
			int pSize = static_cast<int>(param.size());

			for (int j = 0; j < pSize; ++j) {
				Matrix mat(param[j]->getRowLen(), param[j]->getColLen(), param[j]->getDepthLen());
				v.push_back(mat);
			}
			params.insert(std::pair<int, vector<Matrix>>(i, v));
		}
	}
}

void AdaGrad::initParams(float learning_rate, float not_use1, float not_use2) {
	learningRate = learning_rate;
	decayLearningRate = learning_rate;
}

void AdaGrad::update(vector<Layer*>& network) {
	map<int, vector<Matrix>>::iterator iter;

	for (iter = params.begin(); iter != params.end(); iter++) {
		int id = iter->first;
		UpdateInterface *ui = dynamic_cast<UpdateInterface*>(network[id]);
		vector<Matrix>& v = ref(iter->second);
		vector<Matrix*>& param = ui->getParams();
		vector<Matrix*>& grad = ui->getGrads();
		int size = static_cast<int>(v.size());
#ifdef _OPENMP
		#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
		for (int i = 0; i < size; ++i) {
			v[i] += grad[i]->squared();
			Matrix lr_grad = (*grad[i]) * decayLearningRate;// learningRate;
			Matrix sqrt_v = v[i].sqrt() + 1e-7f;
			*param[i] -= lr_grad / sqrt_v;
		}
	}
}
