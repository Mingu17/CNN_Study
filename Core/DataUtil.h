#pragma once

#include "Matrix.h"
#include <vector>
#include "../Core/LayerInfo.h"
#include "../Layer/Layer.h"

using std::vector;

class DataUtil
{
public:
	static int load_mnist(const char *dataPath, const char *labelPath,
		vector<Matrix>& dataSet, vector<Matrix>& labelSet, bool is1D);
	static void random_select(vector<Matrix>& data, vector<Matrix>& label,
		vector<Matrix*>& selectData, vector<Matrix*>& selectLabel, int selectNum);
	static void ordered_select(vector<int>& order, int step,
		vector<Matrix>& data, vector<Matrix>& label,
		vector<Matrix*>& selectData, vector<Matrix*>& selectLabel, int selectNum);

	static void create_cnn_info(vector<LayerInfo>& info, int batch_size);// user define
	static void create_cnn_info2(vector<LayerInfo>& info, int batch_size); // user define2
	static void create_cnn_info3(vector<LayerInfo>& info, int batch_size); // user define3
	static void create_layer(LayerInfo& info, Layer **layer);
};