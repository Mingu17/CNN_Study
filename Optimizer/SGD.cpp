#include <iostream>
#include "SGD.h"
#include "../Layer/UpdateInterface.h"
#include "../Core/NumUtil.h"
#ifdef _OPENMP
#include <omp.h>
#endif

SGD::SGD() : Optimizer() {

}

SGD::~SGD() {

}

void SGD::initLayer(vector<Layer*>& network) {
	//not use
}

void SGD::initParams(float learning_rate, float not_use1, float not_use2) {
	learningRate = learning_rate;
	decayLearningRate = learning_rate;
}

void SGD::update(vector<Layer*>& network) {
	int size = static_cast<int>(network.size());
	for (int i = 0; i < size; ++i) {
		LayerType type = network[i]->getLayerType();
		if (static_cast<int>(type) & UPDATABLE_LAYER) {
			UpdateInterface *ui = dynamic_cast<UpdateInterface*>(network[i]);
			vector<Matrix*> param = ui->getParams();
			vector<Matrix*> grad = ui->getGrads();
			int pSize = static_cast<int>(param.size());
#ifdef _OPENMP
			#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
			for (int j = 0; j < pSize; j++) {
				(*param[j]) -= (*grad[j]) * decayLearningRate;// learningRate;
			}
		}
	}
}