#include <iostream>
#include "NumUtil.h"
#include <random>
#include <intrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int NumUtil::OMP_THREAD_NUM = 4;

void NumUtil::op_default(float *dst, const float *src1, const float *src2,
	const int length, const MatCommType commType) {
	int i = 0;
	int etc = length % 8;
	int lengthRight = length - etc;
	switch (commType)
	{
	case MatCommType::Plus: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc1 = src1 + i;
			const float *tsrc2 = src2 + i;
			float *tdst = dst + i;
			__m256 vsrc1 = _mm256_loadu_ps(tsrc1);
			__m256 vsrc2 = _mm256_loadu_ps(tsrc2);
			_mm256_storeu_ps(tdst, _mm256_add_ps(vsrc1, vsrc2));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src1[i] + src2[i];
		break;
	}
	case MatCommType::Minus: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc1 = src1 + i;
			const float *tsrc2 = src2 + i;
			float *tdst = dst + i;
			__m256 vsrc1 = _mm256_loadu_ps(tsrc1);
			__m256 vsrc2 = _mm256_loadu_ps(tsrc2);
			_mm256_storeu_ps(tdst, _mm256_sub_ps(vsrc1, vsrc2));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src1[i] - src2[i];
		break;
	}
	case MatCommType::Multiple: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc1 = src1 + i;
			const float *tsrc2 = src2 + i;
			float *tdst = dst + i;
			__m256 vsrc1 = _mm256_loadu_ps(tsrc1);
			__m256 vsrc2 = _mm256_loadu_ps(tsrc2);
			_mm256_storeu_ps(tdst, _mm256_mul_ps(vsrc1, vsrc2));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src1[i] * src2[i];
		break;
	}
	case MatCommType::Divide: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc1 = src1 + i;
			const float *tsrc2 = src2 + i;
			float *tdst = dst + i;
			__m256 vsrc1 = _mm256_loadu_ps(tsrc1);
			__m256 vsrc2 = _mm256_loadu_ps(tsrc2);
			_mm256_storeu_ps(tdst, _mm256_div_ps(vsrc1, vsrc2));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src1[i] / src2[i];
		break;
	}
	default: {
		break;
	}
	}
}

void NumUtil::op_default_one(float *dst, const float *src, const float f,
	const int length, const NumCommType commType) {
	int i = 0;
	__m256 var = _mm256_set1_ps(f);
	int etc = length % 8;
	int lengthRight = length - etc;

	switch (commType)
	{
	case NumCommType::Plus: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			_mm256_storeu_ps(tdst, _mm256_add_ps(tvec, var));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src[i] + f;
		break;
	}
	case NumCommType::Minus: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			_mm256_storeu_ps(tdst, _mm256_sub_ps(tvec, var));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src[i] - f;
		break;
	}
	case NumCommType::Multiple: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			_mm256_storeu_ps(tdst, _mm256_mul_ps(tvec, var));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src[i] * f;
		break;
	}
	case NumCommType::Divide: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			_mm256_storeu_ps(tdst, _mm256_div_ps(tvec, var));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src[i] / f;
		break;
	}
	case NumCommType::Square: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			_mm256_storeu_ps(tdst, _mm256_mul_ps(tvec, tvec));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = src[i] * src[i];
		break;
	}
	case NumCommType::Sqrt: {
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			__m256 tadd = _mm256_add_ps(tvec, var);
			_mm256_storeu_ps(tdst, _mm256_sqrt_ps(tadd));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = sqrtf(src[i] + f);
		break;
	}
	case NumCommType::Inverse: {
		__m256 invOne = _mm256_set1_ps(1.0f);
		for (i = 0; i < lengthRight; i += 8) {
			const float *tsrc = src + i;
			float *tdst = dst + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			_mm256_storeu_ps(tdst, _mm256_div_ps(invOne, tvec));
		}
		for (i = lengthRight; i < length; ++i) dst[i] = 1.0f / src[i];
		break;
	}
	default:
		break;
	}
}

void NumUtil::endian_swap_32(int& x) {
	x = (x >> 24) |
		((x << 8) & 0x00FF0000) |
		((x >> 8) & 0x0000FF00) |
		(x << 24);
}

void NumUtil::set_gaussian_random(Matrix& mat) {
	float *matData = mat.getData();
	int len = mat.getTotalLen();
	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<float> distribution(0.0f, 1.0f);

	for (int i = 0; i < len; i++) {
		matData[i] = distribution(generator);
	}
}

void NumUtil::set_xavier_weight(Matrix& mat) {
	int inSize = mat.getRowLen();
	int outSize = mat.getColLen();
	int len = mat.getTotalLen();
	float revSqrtIn = 1.0f / std::sqrtf(static_cast<float>(inSize));
	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<float> distribution(0.0f, 1.0f);
	float* matData = mat.getData();

	for (int i = 0; i < len; i++) {
		matData[i] = distribution(generator) * revSqrtIn;
	}
}

void NumUtil::set_he_weight(Matrix& mat) {
	int inSize = mat.getRowLen();
	int outSize = mat.getColLen();
	int len = mat.getTotalLen();
	float revSqrtIn = std::sqrtf(2.0f / static_cast<float>(inSize));
	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<float> distribution(0.0f, 1.0f);
	float* matData = mat.getData();

	for (int i = 0; i < len; i++) {
		matData[i] = distribution(generator) * revSqrtIn;
	}
}

void NumUtil::get_mean(vector<Matrix*>& x, Matrix& mean) {
	int row = x[0]->getRowLen();
	int col = x[0]->getColLen();
	int depth = x[0]->getDepthLen();

	if (mean.getTotalLen() > 0) {
		if (row != mean.getRowLen() || col != mean.getColLen() || depth != mean.getDepthLen()) {
			throw std::invalid_argument("[Error] size mismatch");
		}
	}
	else {
		mean.init(row, col, depth);
	}

	mean.setZero();
	int xSize = static_cast<int>(x.size());
	float revSize = 1.0f / static_cast<float>(xSize);

	for (int i = 0; i < xSize; i++) {
		mean += *x[i];
	}
	mean *= revSize;
}

void NumUtil::get_mean_omp(vector<Matrix*>& x, Matrix& mean) {
	int row = x[0]->getRowLen();
	int col = x[0]->getColLen();
	int depth = x[0]->getDepthLen();

	if (mean.getTotalLen() > 0) {
		if (row != mean.getRowLen() || col != mean.getColLen() || depth != mean.getDepthLen()) {
			throw std::invalid_argument("[Error] size mismatch");
		}
	}
	else {
		mean.init(row, col, depth);
	}

	mean.setZero();
	int xSize = static_cast<int>(x.size());
	float revSize = 1.0f / static_cast<float>(xSize);

	vector<Matrix> meanSet;
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		Matrix _mean(mean.getTotalSize(), 0.0f);
		meanSet.push_back(_mean);
	}

	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
	for (int i = 0; i < xSize; ++i) {
		int id = omp_get_thread_num();
		meanSet[id] += *x[i];
	}

	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		mean += meanSet[i];
	}

	mean *= revSize;
}

void NumUtil::get_sum(vector<Matrix*>& x, Matrix& sum) {
	int row = x[0]->getRowLen();
	int col = x[0]->getColLen();
	int depth = x[0]->getDepthLen();

	if (sum.getTotalLen() > 0) {
		if (row != sum.getRowLen() || col != sum.getColLen() || depth != sum.getDepthLen()) {
			throw std::invalid_argument("[Error] size mismatch");
		}
	}
	else {
		sum.init(row, col, depth);
	}

	sum.setZero();
	int xSize = static_cast<int>(x.size());
	for (int i = 0; i < xSize; ++i) {
		sum += *x[i];
	}
}

void NumUtil::get_sum_omp(vector<Matrix*>& x, Matrix& sum) {
	int row = x[0]->getRowLen();
	int col = x[0]->getColLen();
	int depth = x[0]->getDepthLen();

	if (sum.getTotalLen() > 0) {
		if (row != sum.getRowLen() || col != sum.getColLen() || depth != sum.getDepthLen()) {
			throw std::invalid_argument("[Error] size mismatch");
		}
	}
	else {
		sum.init(row, col, depth);
	}

	sum.setZero();
	int xSize = static_cast<int>(x.size());

	vector<Matrix> sumSet;
	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		Matrix _sum(sum.getTotalSize(), 0.0f);
		sumSet.push_back(_sum);
	}

	#pragma omp parallel for num_threads(NumUtil::OMP_THREAD_NUM)
	for (int i = 0; i < xSize; ++i) {
		int id = omp_get_thread_num();
		sumSet[id] += *x[i];
	}

	for (int i = 0; i < NumUtil::OMP_THREAD_NUM; ++i) {
		sum += sumSet[i];
	}
}

float NumUtil::sigmoid(float f) {
	return 1.0f / (1.0f + expf(-f));
}

float NumUtil::sigmoid_grad(float f) {
	float sf = sigmoid(f);
	return (1.0f - sf) * sf;
}

float NumUtil::relu(float f) {
	return (f > 0.0f) ? f : 0.0f;
}

float NumUtil::relu_grad(float f) {
	return (f > 0.0f) ? 1.0f : 0.0f;
}

int NumUtil::softmax(Matrix& src, Matrix& dst) {
	if (dst.getTotalLen() != 0) {
		if (src.getRowLen() != dst.getRowLen() || src.getColLen() != dst.getColLen()
			|| src.getDepthLen() != dst.getDepthLen()) {
			throw std::invalid_argument("[Error] size mismatch");
		}
	}
	else {
		dst.init(src.getRowLen(), src.getColLen(), src.getDepthLen());
	}
	int len = src.getTotalLen(), i;
	float sum = 0.0f;
	float max = src.getMax();
	Matrix in = src - max;

	float *srcData = in.getData();
	float *dstData = dst.getData();

	for (i = 0; i < len; i++) {
		dstData[i] = expf(srcData[i]);
		sum += dstData[i];
	}
	sum = 1.0f / sum;
	if (sum == 0.0f) {
		return -2;
	}

	for (i = 0; i < len; i++) {
		dstData[i] *= sum;
	}
	return 1;
}

float NumUtil::mean_squared_error(Matrix& y, Matrix& t) {
	Matrix sub = y - t;
	float squaredSum = sub.squared().getSum();
	return squaredSum * 0.5f;
}

float NumUtil::cross_entropy_error(Matrix& y, Matrix& t) {
	int row = y.getRowLen();
	int col = y.getColLen();
	int depth = y.getDepthLen();
	if (row != t.getRowLen() || col != t.getColLen() || depth != t.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float delta = 1e-07f;
	float sum = 0.0f;
	float *yData = y.getData();
	float *tData = t.getData();
	int len = y.getTotalLen();

	for (int i = 0; i < len; i++) {
		sum += (tData[i] * logf(yData[i] + delta));
	}
	return -sum;
}

float NumUtil::cross_entropy_error(vector<Matrix*>& y, vector<Matrix*>& t) {
	int batchSize = static_cast<int>(y.size());
	if (y.size() != t.size()) {
		throw std::invalid_argument("[Error] batch size mismatch");
	}

	float sum = 0.0f;
	for (int i = 0; i < batchSize; i++) {
		sum += cross_entropy_error(*y[i], *t[i]);
	}
	return sum / (float)batchSize;
}

bool NumUtil::isCorrect1D(Matrix& y, Matrix& t) {
	if (y.getRowLen() != t.getRowLen() || y.getColLen() != t.getColLen()
		|| y.getDepthLen() != t.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float max_y = -FLT_MAX, max_t = -FLT_MAX;
	int maxId_y = -1, maxId_t = -1;
	int len = y.getTotalLen();
	float *yData = y.getData();
	float *tData = t.getData();

	for (int i = 0; i < len; i++) {
		if (max_y < yData[i]) {
			max_y = yData[i];
			maxId_y = i;
		}
		if (max_t < tData[i]) {
			max_t = tData[i];
			maxId_t = i;
		}
	}

	return (maxId_y == maxId_t);
}

void NumUtil::weight_init_affine(WeightInitType type, float factor, Matrix& weight) {
	switch (type)
	{
	case WeightInitType::None:
	case WeightInitType::Normal: {
		//NumUtil::set_gaussian_random(weight);
		float *matData = weight.getData();
		int len = weight.getTotalLen();
		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<float> distribution(0.0f, 1.0f);

		for (int i = 0; i < len; i++) {
			matData[i] = distribution(generator);
		}
		weight *= factor;
		break;
	}
	case WeightInitType::Xavier: {
		//NumUtil::set_xavier_weight(weight);
		int inSize = weight.getRowLen();
		int outSize = weight.getColLen();
		int len = weight.getTotalLen();
		float revSqrtIn = 1.0f / std::sqrtf(static_cast<float>(inSize));
		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<float> distribution(0.0f, revSqrtIn);
		float* matData = weight.getData();

		for (int i = 0; i < len; i++) {
			matData[i] = distribution(generator);// *revSqrtIn;
		}
		break;
	}
	case WeightInitType::He: {
		////NumUtil::set_he_weight(weight);
		int inSize = weight.getRowLen();
		int outSize = weight.getColLen();

		int len = weight.getTotalLen();
		float revSqrtIn = std::sqrtf(2.0f / static_cast<float>(inSize));
		float max = std::sqrt(6.0f / static_cast<float>(inSize + outSize));

		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<float> distribution(0.0f, revSqrtIn);// 1.0f);
		float* matData = weight.getData();

		for (int i = 0; i < len; i++) {
			while (true) {
				float w = distribution(generator);
				if (w >= -max && w <= max) {
					matData[i] = w;
					break;
				}
			}
			//matData[i] = distribution(generator);// *revSqrtIn;
		}

		break;
	}
	default:
		break;
	}
}

void NumUtil::weight_init_conv(WeightInitType type, float factor, Matrix& weight, int in, int out) {
	switch (type)
	{
	case WeightInitType::None:
	case WeightInitType::Normal: {
		//NumUtil::set_gaussian_random(weight);
		float *matData = weight.getData();
		int len = weight.getTotalLen();
		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<float> distribution(0.0f, 1.0f);

		for (int i = 0; i < len; i++) {
			matData[i] = distribution(generator);
		}
		weight *= factor;
		break;
	}
	case WeightInitType::Xavier: {
		//NumUtil::set_xavier_weight(weight);
		int inSize = in;// weight.getRowLen();
		int outSize = out;// weight.getColLen();
		int len = weight.getTotalLen();
		float revSqrtIn = std::sqrtf(1.0f / static_cast<float>(inSize));
		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<float> distribution(0.0f, 1.0f);
		float* matData = weight.getData();

		for (int i = 0; i < len; i++) {
			matData[i] = distribution(generator) * revSqrtIn;
		}
		break;
	}
	case WeightInitType::He: {
		//NumUtil::set_he_weight(weight);
		int inSize = in;// weight.getRowLen();
		int outSize = out;// weight.getColLen();
		int len = weight.getTotalLen();
		float revSqrtIn = std::sqrtf(2.0f / static_cast<float>(inSize));
		float max = std::sqrt(6.0f / static_cast<float>(inSize + outSize));

		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<float> distribution(0.0f, revSqrtIn);// 1.0f);
		float* matData = weight.getData();

		for (int i = 0; i < len; i++) {
			while (true) {
				float w = distribution(generator);
				if (w >= -max && w <= max) {
					matData[i] = w;
					break;
				}
			}
			//matData[i] = distribution(generator);// *revSqrtIn;
		}
		
		break;
	}
	default:
		break;
	}
}

void NumUtil::create_conv_map(const LayerSize& inputSize, const LayerSize& filterSize,
	const int stride, const int padding,
	vector<vector<pair<int, int>>>& fwdMap,
	vector<vector<pair<int, int>>>& dxMap,
	vector<vector<pair<int, int>>>& dwMap) {
	
	//1. padding 을 적용한 LayerSize 및 Matrix 생성
	LayerSize padSize(inputSize.getRow() + padding * 2, inputSize.getCol() + padding * 2, inputSize.getDepth());
	Matrix pMat(padSize, -1.0f);
	float *pData = pMat.getData();
	int idx = 0;
	int planeSize = pMat.getLenPerLayer();

	//2. setup index
	for (int d = 0; d < padSize.getDepth(); ++d) {
		int dSize = d * planeSize;
		for (int r = padding; r < padSize.getRow() - padding; ++r) {
			int pSize = dSize + (r * padSize.getCol());
			for (int c = padding; c < padSize.getCol() - padding; ++c) {
				int cSize = pSize + c;
				pData[cSize] = static_cast<float>(idx);
				idx++;
			}
		}
	}

	//3. create convolution computing map
	for (int inR = 0; inR < padSize.getRow() - filterSize.getRow() + 1; inR += stride) {
		int inRLoc = inR * padSize.getCol();

		for (int inC = 0; inC < padSize.getCol() - filterSize.getCol() + 1; inC += stride) {
			int startLoc = inRLoc + inC;
			vector<pair<int, int>> cmdMap;
			int fLoc = 0;

			for (int fD = 0; fD < padSize.getDepth(); ++fD) {
				int sD = startLoc + (padSize.getLenPerPlane() * fD);

				for (int fR = 0; fR < filterSize.getRow(); ++fR) {
					if (inR + fR < padSize.getRow()) {
						int sR = sD + (fR * padSize.getCol());
						for (int fC = 0; fC < filterSize.getCol(); ++fC) {
							if (inC + fC < padSize.getCol()) {
								int sC = sR + fC;
								if (pData[sC] >= 0.0f) {
									int pIdx = static_cast<int>(pData[sC]);
									cmdMap.push_back(pair<int, int>(pIdx, fLoc));//input, filter
								}
							}
							fLoc++;
						}
					}
					else fLoc += filterSize.getCol();
				}
			}
			fwdMap.push_back(cmdMap);
		}
	}

	//4. create dx / dw computing map
	int inputLen = inputSize.getLen();
	int filterLen = filterSize.getLen();
	int i, j;

	for (i = 0; i < inputLen; ++i) {
		vector<pair<int, int>> cmdMap;
		dxMap.push_back(cmdMap);
	}

	for (i = 0; i < filterLen; ++i) {
		vector<pair<int, int>> cmdMap;
		dwMap.push_back(cmdMap);
	}

	for (i = 0; i < static_cast<int>(fwdMap.size()); ++i) {
		int cmdLen = static_cast<int>(fwdMap[i].size());
		for (j = 0; j < cmdLen; ++j) {
			int i_idx = fwdMap[i][j].first;
			int f_idx = fwdMap[i][j].second;
			dxMap[i_idx].push_back(pair<int, int>(i, f_idx));//dx, out, filter
			dwMap[f_idx].push_back(pair<int, int>(i_idx, i));//dw, input, out
		}
	}
}

void NumUtil::create_forward_map(const LayerSize& inputSize, const LayerSize& filterSize,
	const int stride, const int padding, vector<vector<pair<int, int>>>& computeMap) {

	//1. padding 을 적용한 LayerSize 및 Matrix 생성
	LayerSize padSize(inputSize.getRow() + padding * 2, inputSize.getCol() + padding * 2, inputSize.getDepth());
	Matrix pMat(padSize, -1.0f);
	float *pData = pMat.getData();
	int idx = 0;
	int planeSize = pMat.getLenPerLayer();

	//2. setup index
	for (int d = 0; d < padSize.getDepth(); ++d) {
		int dSize = d * planeSize;
		for (int r = padding; r < padSize.getRow() - padding; ++r) {
			int pSize = dSize + (r * padSize.getCol());
			for (int c = padding; c < padSize.getCol() - padding; ++c) {
				int cSize = pSize + c;
				pData[cSize] = static_cast<float>(idx);
				idx++;
			}
		}
	}

	//pMat.printData();

	//3. create convolution computing map
	for (int inR = 0; inR < padSize.getRow() - filterSize.getRow() + 1; inR += stride) {
		int inRLoc = inR * padSize.getCol();

		for (int inC = 0; inC < padSize.getCol() - filterSize.getCol() + 1; inC += stride) {
			int startLoc = inRLoc + inC;
			vector<pair<int, int>> cmdMap;
			int fLoc = 0;

			for (int fD = 0; fD < padSize.getDepth(); ++fD) {
				int sD = startLoc + (padSize.getLenPerPlane() * fD);

				for (int fR = 0; fR < filterSize.getRow(); ++fR) {
					if (inR + fR < padSize.getRow()) {
						int sR = sD + (fR * padSize.getCol());
						for (int fC = 0; fC < filterSize.getCol(); ++fC) {
							if (inC + fC < padSize.getCol()) {
								int sC = sR + fC;
								if (pData[sC] >= 0.0f) {
									int pIdx = static_cast<int>(pData[sC]);
									cmdMap.push_back(pair<int, int>(pIdx, fLoc));
								}
							}
							fLoc++;
						}
					}
					else fLoc += filterSize.getCol();
				}
			}
			computeMap.push_back(cmdMap);
		}
	}
}

void NumUtil::create_dx_map(const LayerSize& inputSize, const LayerSize& doutSize, const LayerSize& filterSize,
	const int stride, const int padding, vector<vector<pair<int, int>>>& computeMap) {
	//1. dout 사이즈 설정
	//dout_size + ((dout_size - 1) * (stride - 1)) + (filter_size - 1 - padding)         CONV filter_reverse
	int termStride = stride - 1;
	int termPadRow = filterSize.getRow() - 1 - padding;
	int termPadCol = filterSize.getCol() - 1 - padding;
	int doutReRow = doutSize.getRow() + ((doutSize.getRow() - 1) * termStride) + (termPadRow * 2);
	int doutReCol = doutSize.getCol() + ((doutSize.getCol() - 1) * termStride) + (termPadCol * 2);

	LayerSize doutResize(doutReRow, doutReCol, 1);
	Matrix dout(doutResize, -1.0f);
	float *pDout = dout.getData();
	int fIdx = 0;

	for (int r = termPadRow; r < doutReRow - termPadRow; r += stride) {
		int sR = r * doutReCol;
		for (int c = termPadCol; c < doutReCol - termPadCol; c += stride) {
			int sC = sR + c;
			pDout[sC] = static_cast<float>(fIdx);
			fIdx++;
		}
	}
	//dout.printData();

	//2. filter setting
	Matrix filter(filterSize);
	float *pFilter = filter.getData();
	for (int f = 0; f < static_cast<int>(filter.getTotalLen()); ++f) {
		pFilter[f] = static_cast<float>(f);
	}
	Matrix revFilter = Matrix(filter, MatConsType::Reverse);
	//revFilter.printData();

	//3. convolution
	float *pRevF = revFilter.getData();
	for (int fD = 0; fD < filterSize.getDepth(); ++fD) {
		float *pDout = dout.getData();

		for (int sR = 0; sR < doutReRow - filterSize.getRow() + 1; ++sR) {
			int sRLoc = sR * doutReCol;
			for (int sC = 0; sC < doutReCol - filterSize.getCol() + 1; ++sC) {
				int sCLoc = sRLoc + sC;
				int fLoc = 0;
				//convolution
				float *pF = pRevF + (fD * filterSize.getLenPerPlane()); //현재 feature plane
				vector<pair<int, int>> cmdMap;

				for (int fR = 0; fR < filterSize.getRow(); ++fR) {
					int srcRLoc = sCLoc + (fR * doutReCol);
					for (int fC = 0; fC < filterSize.getCol(); ++fC) {
						int loc = srcRLoc + fC;
						if (pDout[loc] >= 0.0f) {
							int sIdx = static_cast<int>(pDout[loc]);
							int fIdx = static_cast<int>(pF[fLoc]);
							cmdMap.push_back(pair<int, int>(sIdx, fIdx));
						}
						fLoc++;
					}
				}
				computeMap.push_back(cmdMap);
			}
		}
	}
}

void NumUtil::create_dw_map(const LayerSize& inputSize, const LayerSize& doutSize,
	const int stride, const int padding, vector<vector<pair<int, int>>>& computeMap) {
	//1. 사이즈 설정
	//input_size + padding_size      CONV dout
	LayerSize inputReSize(inputSize.getRow() + padding * 2, inputSize.getCol() + padding * 2, inputSize.getDepth());
	Matrix input(inputReSize, -1.0f);
	float *pInput = input.getData();
	int idx = 0;

	for (int d = 0; d < inputReSize.getDepth(); ++d) {
		int sD = d * inputReSize.getLenPerPlane();
		for (int r = padding; r < inputReSize.getRow() - padding; ++r) {
			int sR = sD + (r * inputReSize.getCol());
			for (int c = padding; c < inputReSize.getCol() - padding; ++c) {
				int sC = sR + c;
				pInput[sC] = static_cast<float>(idx);
				idx++;
			}
		}
	}

	//2. dout 설정
	int doutReRow = doutSize.getRow() + ((doutSize.getRow() - 1) * (stride - 1));
	int doutReCol = doutSize.getCol() + ((doutSize.getCol() - 1) * (stride - 1));
	LayerSize doutReSize(doutReRow, doutReCol, 1);
	Matrix dout(doutReSize, -1.0f);
	float *pDout = dout.getData();
	int dIdx = 0;

	for (int dr = 0; dr < doutReRow; dr += stride) {
		int dR = dr * doutReCol;
		for (int dc = 0; dc < doutReCol; dc += stride) {
			int dC = dR + dc;
			pDout[dC] = static_cast<float>(dIdx);
			dIdx++;
		}
	}

	//3. Convolution
	pInput = input.getData();
	for (int iD = 0; iD < inputReSize.getDepth(); ++iD) {
		//float *pInputPlane = pInput + (iD * inputReSize.getLenPerPlane());
		int iDLoc = iD * inputReSize.getLenPerPlane();

		for (int iR = 0; iR < inputReSize.getRow() - doutReSize.getRow() + 1; ++iR) {
			int iRLoc = iDLoc + (iR * inputReSize.getCol());
			for (int iC = 0; iC < inputReSize.getCol() - doutReSize.getCol() + 1; ++iC) {
				int iCLoc = iRLoc + iC;
				pDout = dout.getData();

				vector<pair<int, int>> cmdMap;
				int dIdx = 0;

				for (int dR = 0; dR < doutReSize.getRow(); ++dR) {
					int inRLoc = iCLoc + (dR * inputReSize.getCol());
					for (int dC = 0; dC < doutReSize.getCol(); ++dC) {
						int inCLoc = inRLoc + dC;
						if (pInput[inCLoc] >= 0.0f && pDout[dIdx] >= 0.0f) {
							int inIdx = static_cast<int>(pInput[inCLoc]);
							int doIdx = static_cast<int>(pDout[dIdx]);
							cmdMap.push_back(pair<int, int>(inIdx, doIdx));
						}
						dIdx++;
					}
				}

				computeMap.push_back(cmdMap);
			}
		}
	}
}

void NumUtil::create_pooling_map(const LayerSize& inputSize, const LayerSize& poolingSize,
	const int stride, vector<vector<int>>& computeMap, bool isExtend) {
	
	int inRow = inputSize.getRow();
	int inCol = inputSize.getCol();
	int poRow = poolingSize.getRow();
	int poCol = poolingSize.getCol();

	if (isExtend) {
		for (int i = 0; i < inRow; i += stride) {
			int iR = i * inCol;
			for (int j = 0; j < inCol; j += stride) {
				int iC = iR + j;
				vector<int> cmdMap;

				for (int y = 0; y < poRow; ++y) {
					if (i + y < inRow) {
						int pR = iC + (y * inCol);
						for (int x = 0; x < poCol; ++x) {
							if (j + x < inCol) {
								int pC = pR + x;
								cmdMap.push_back(pC);
							}
						}
					}
				}
				computeMap.push_back(cmdMap);
			}
		}
	}
	else {
		for (int i = 0; i < inRow; i += stride) {
			if (i + poRow - 1 < inRow) {
				int iR = i * inCol;
				for (int j = 0; j < inCol; j += stride) {
					if (j + poCol - 1 < inCol) {
						int iC = iR + j;
						vector<int> cmdMap;

						for (int y = 0; y < poRow; ++y) {
							int pR = iC + (y * inCol);
							for (int x = 0; x < poCol; ++x) {
								int pC = pR + x;
								cmdMap.push_back(pC);
							}
						}
						computeMap.push_back(cmdMap);
					}
				}
			}
		}
	}
}

void NumUtil::activate(Matrix& src, Matrix& dst, ActivateType actType) {
	float *srcData = src.getData();
	float *dstData = dst.getData();
	int length = src.getTotalLen();
	int etc = length % 8;
	int lengthRight = length - etc;

	switch (actType)
	{
	case ActivateType::Relu: {
		__m256 zero = _mm256_setzero_ps();
		for (int i = 0; i < lengthRight; i += 8) {
			float *tsrc = srcData + i;
			float *tdst = dstData + i;
			__m256 tvec = _mm256_loadu_ps(tsrc);
			__m256 tand = _mm256_and_ps(tvec, _mm256_cmp_ps(tvec, zero, _CMP_GT_OS));
			_mm256_storeu_ps(tdst, tand);
		}
		for (int i = lengthRight; i < length; ++i) {
			dstData[i] = relu(srcData[i]);
		}
		break;
	}
	case ActivateType::Sigmoid: {
		for (int i = 0; i < length; ++i) {
			dstData[i] = sigmoid(srcData[i]);
		}
		break;
	}
	default:
		break;
	}
}

void NumUtil::deactivate(Matrix& d, Matrix& out, Matrix& dout, ActivateType actType) {
	float *dData = d.getData();
	float *outData = out.getData();
	float *doutData = dout.getData();
	int length = d.getTotalLen();
	int etc = length % 8;
	int lengthRight = length - etc;

	switch (actType)
	{
	case ActivateType::Relu: {
		__m256 zero = _mm256_setzero_ps();
		for (int i = 0; i < lengthRight; i += 8) {
			float *td = dData + i;
			float *tout = outData + i;
			float *tdout = doutData + i;
			__m256 tc = _mm256_loadu_ps(td);
			__m256 tvec = _mm256_loadu_ps(tout);
			__m256 tand = _mm256_and_ps(tc, _mm256_cmp_ps(tvec, zero, _CMP_GT_OS));
			_mm256_storeu_ps(tdout, tand);
		}
		for (int i = lengthRight; i < length; ++i) {
			doutData[i] = (outData[i] > 0.0f) ? dData[i] : 0.0f;
		}
		break;
	}
	case ActivateType::Sigmoid: {
		__m256 one = _mm256_set1_ps(1.0f);
		for (int i = 0; i < lengthRight; ++i) {
			float *fd = dData + i;
			float *fout = outData + i;
			float *fdout = doutData + i;
			__m256 td = _mm256_loadu_ps(fd);
			__m256 tout = _mm256_loadu_ps(fout);
			__m256 upper = _mm256_mul_ps(td, _mm256_sub_ps(one, tout));
			_mm256_storeu_ps(fdout, _mm256_mul_ps(upper, tout));
		}
		for (int j = lengthRight; j < length; ++j) {
			doutData[j] = dData[j] * (1.0f - outData[j]) * outData[j];
		}
		break;
	}
	default:
		break;
	}
}