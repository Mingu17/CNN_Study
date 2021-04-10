#include <iostream>
#include "Adam.h"
#include "../Layer/UpdateInterface.h"
#include <utility>
#include "../Core/NumUtil.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using std::pair;

Adam::Adam() : Optimizer() {

}

Adam::~Adam() {

}

void Adam::initLayer(vector<Layer*>& network) {
	int size = static_cast<int>(network.size());

	for (int i = 0; i < size; ++i) {
		LayerType type = network[i]->getLayerType();
		if(static_cast<int>(type) & UPDATABLE_LAYER) {
			UpdateInterface *ui = dynamic_cast<UpdateInterface*>(network[i]);
			vector<Matrix> v;
			vector<Matrix> v2;
			vector<Matrix*>& param = ui->getParams();
			int pSize = static_cast<int>(param.size());

			for (int j = 0; j < pSize; ++j) {
				Matrix mat(param[j]->getRowLen(), param[j]->getColLen(), param[j]->getDepthLen());
				Matrix mat2(param[j]->getRowLen(), param[j]->getColLen(), param[j]->getDepthLen());
				v.push_back(mat);
				v2.push_back(mat2);
			}
			params.insert(std::pair<int, vector<Matrix>>(i, v));
			params2.insert(std::pair<int, vector<Matrix>>(i, v2));
		}
	}
}

void Adam::initParams(float learning_rate, float beta_1, float beta_2) {
	learningRate = learning_rate;
	decayLearningRate = learning_rate;
	beta1 = beta_1;
	beta2 = beta_2;
	iteration = 0.0f;
}

void Adam::update(vector<Layer*>& network) {
	map<int, vector<Matrix>>::iterator m_iter, v_iter;
	iteration += 1.0f;
	float lr = decayLearningRate;
	float lr_t = lr * sqrtf(1.0f - powf(beta2, iteration)) / (1.0f - powf(beta1, iteration));
	
	for (m_iter = params.begin(), v_iter = params2.begin();
		m_iter != params.end(); m_iter++, v_iter++) {
		int id = m_iter->first;
		UpdateInterface *ui = dynamic_cast<UpdateInterface*>(network[id]);
		vector<Matrix>& m = ref(m_iter->second);
		vector<Matrix>& v = ref(v_iter->second);
		vector<Matrix*>& param = ui->getParams();
		vector<Matrix*>& grad = ui->getGrads();
		int size = static_cast<int>(m.size());
#ifdef _OPENMP
		#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
#endif
		for (int i = 0; i < size; ++i) {
			m[i] += (*grad[i] - m[i]) * (1.0f - beta1);
			v[i] += (grad[i]->squared() - v[i]) * (1.0f - beta2);
			Matrix sqrt_v = v[i].sqrt() + 1e-7f;
			*param[i] -= (m[i] * lr_t) / sqrt_v;
		}
	}
}
