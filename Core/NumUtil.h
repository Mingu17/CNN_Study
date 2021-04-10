#pragma once

#include "Matrix.h"
#include <vector>

using std::vector;
using std::pair;

class NumUtil
{
//function
public:
	static void op_default(float *dst, const float *src1, const float *src2,
		const int length, const MatCommType commType);

	static void op_default_one(float *dst, const float *src, const float f,
		const int length, const NumCommType commType);

	static void endian_swap_32(int& x);
	static void set_gaussian_random(Matrix& mat);
	static void set_xavier_weight(Matrix& mat);
	static void set_he_weight(Matrix& mat);
	static void get_mean(vector<Matrix*>& x, Matrix& mean);
	static void get_mean_omp(vector<Matrix*>& x, Matrix& mean);
	static void get_sum(vector<Matrix*>& x, Matrix& sum);
	static void get_sum_omp(vector<Matrix*>& x, Matrix& sum);

	static float sigmoid(float f);
	static float sigmoid_grad(float f);
	static float relu(float f);
	static float relu_grad(float f);
	static int softmax(Matrix& src, Matrix& dst);
	static float mean_squared_error(Matrix& y, Matrix& t);
	static float cross_entropy_error(Matrix& y, Matrix& t);
	static float cross_entropy_error(vector<Matrix*>& y, vector<Matrix*>& t);

	static bool isCorrect1D(Matrix& y, Matrix& t);
	static void weight_init_affine(WeightInitType type, float factor, Matrix& weight);
	static void weight_init_conv(WeightInitType type, float factor, Matrix& weight, int in, int out);

	static void create_conv_map(const LayerSize& inputSize, const LayerSize& filterSize,
		const int stride, const int padding,
		vector<vector<pair<int, int>>>& fwdMap,
		vector<vector<pair<int, int>>>& dxMap,
		vector<vector<pair<int, int>>>& dwMap);

	static void create_forward_map(const LayerSize& inputSize, const LayerSize& filterSize,
		const int stride, const int padding, vector<vector<pair<int, int>>>& computeMap);

	static void create_dx_map(const LayerSize& inputSize, const LayerSize& doutSize, const LayerSize& filterSize,
		const int stride, const int padding, vector<vector<pair<int, int>>>& computeMap);

	static void create_dw_map(const LayerSize& inputSize, const LayerSize& doutSize,
		const int stride, const int padding, vector<vector<pair<int, int>>>& computeMap);

	static void create_pooling_map(const LayerSize& inputSize, const LayerSize& poolingSize,
		const int stride, vector<vector<int>>& computeMap, bool isExtend = false);

	static void activate(Matrix& src, Matrix& dst, ActivateType actType);
	static void deactivate(Matrix& d, Matrix& out, Matrix& dout, ActivateType actType);
//variable
public:
	static int OMP_THREAD_NUM;
};