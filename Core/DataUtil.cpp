#include <iostream>
#include "DataUtil.h"
#include "NumUtil.h"
#include <random>
#include <iterator>
#include <algorithm>
#include "../Core/LayerSize.h"
#include "../BatchLayer/BatchLayerSet.h"

int DataUtil::load_mnist(const char *dataPath, const char *labelPath,
	vector<Matrix>& dataSet, vector<Matrix>& labelSet, bool is1D) {
	FILE *data_fp = 0, *label_fp = 0;
	int magicNum[2], itemNum[2], hei, wid;

	errno_t err = fopen_s(&data_fp, dataPath, "rb");
	if (err != 0) {
		throw std::invalid_argument("[Error] cannot file open");
	}
	err = fopen_s(&label_fp, labelPath, "rb");
	if (err != 0) {
		throw std::invalid_argument("[Error] cannot file open");
	}

	fread_s(&magicNum[0], 4, 4, 1, label_fp);
	NumUtil::endian_swap_32(magicNum[0]);
	fread_s(&itemNum[0], 4, 4, 1, label_fp);
	NumUtil::endian_swap_32(itemNum[0]);

	fread_s(&magicNum[1], 4, 4, 1, data_fp);
	NumUtil::endian_swap_32(magicNum[1]);
	fread_s(&itemNum[1], 4, 4, 1, data_fp);
	NumUtil::endian_swap_32(itemNum[1]);
	fread_s(&hei, 4, 4, 1, data_fp);
	NumUtil::endian_swap_32(hei);
	fread_s(&wid, 4, 4, 1, data_fp);
	NumUtil::endian_swap_32(wid);

	unsigned char *buffer = new unsigned char[wid*hei];
	int bufferSize = wid * hei * sizeof(unsigned char);
	int length = wid * hei;
	unsigned char id;
	int nId;

	for (int i = 0; i < itemNum[0]; i++) {
		fread_s(&id, 1, 1, 1, label_fp);
		fread_s(buffer, bufferSize, bufferSize, 1, data_fp);
		nId = static_cast<int>(id);

		Matrix img, label(10, 1, 1);
		if (is1D) img.init(hei * wid, 1, 1);
		else img.init(hei, wid, 1);

		float *imgData = img.getData();

		for (int x = 0; x < length; x++) {
			imgData[x] = static_cast<float>(buffer[x]) / 255.0f;
		}
		label.setElement(1.0f, nId);

		dataSet.push_back(img);
		labelSet.push_back(label);
	}

	fclose(data_fp);
	fclose(label_fp);
	return itemNum[0];
}

void DataUtil::random_select(vector<Matrix>& data, vector<Matrix>& label,
	vector<Matrix*>& selectData, vector<Matrix*>& selectLabel, int selectNum) {
	if (selectData.size() != 0) {
		selectData.clear();
		selectLabel.clear();
	}
	vector<int> result;
	int oriSize = static_cast<int>(data.size());
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
	std::uniform_int_distribution<int> dist(0, oriSize - 1);

	for (int i = 0; i < selectNum; i++) {
		int randNum = dist(engine);
		selectData.push_back(&data[randNum]);
		selectLabel.push_back(&label[randNum]);
	}
}

void DataUtil::ordered_select(vector<int>& order, int step,
	vector<Matrix>& data, vector<Matrix>& label,
	vector<Matrix*>& selectData, vector<Matrix*>& selectLabel, int selectNum) {
	if (selectData.size() != 0) {
		selectData.clear();
		selectLabel.clear();
	}
	int startIdx = step * selectNum;

	for (int i = 0; i < selectNum; ++i) {
		selectData.push_back(&data[order[startIdx + i]]);
		selectLabel.push_back(&label[order[startIdx + i]]);
	}
}

void DataUtil::create_cnn_info(vector<LayerInfo>& info, int batch_size) {
	//part 1 : Conv - BN - Act - Conv - BN - Act - Pooling
	LayerInfo conv00(batch_size);
	conv00.setLayerType(LayerType::Convolution);
	conv00.setInputSize(LayerSize(28, 28, 1));
	conv00.setFilterSize(LayerSize(3, 3, 1));
	conv00.setConvolutionInfo(16, 1, 1);
	conv00.setWeightInfo(WeightInitType::He, 0.01f);
	conv00.computeConvOutputSize();
	info.push_back(conv00);

	LayerInfo bn00(batch_size);
	bn00.setLayerType(LayerType::BatchNorm);
	bn00.setInputSize(conv00.getOutputSize());
	bn00.setOutputSize(conv00.getOutputSize());
	bn00.setMomentum(0.9f);
	info.push_back(bn00);

	LayerInfo act00(batch_size);
	act00.setLayerType(LayerType::Relu);
	act00.setInputSize(bn00.getOutputSize());
	act00.setOutputSize(bn00.getOutputSize());
	act00.setActivateType(ActivateType::Relu);
	info.push_back(act00);

	LayerInfo conv01(batch_size);
	conv01.setLayerType(LayerType::Convolution);
	conv01.setInputSize(act00.getOutputSize());
	conv01.setFilterSize(LayerSize(3, 3, act00.getOutputSize().getDepth()));
	conv01.setConvolutionInfo(16, 1, 1);
	conv01.setWeightInfo(WeightInitType::He, 0.01f);
	conv01.computeConvOutputSize();
	info.push_back(conv01);

	LayerInfo bn01(batch_size);
	bn01.setLayerType(LayerType::BatchNorm);
	bn01.setInputSize(conv01.getOutputSize());
	bn01.setOutputSize(conv01.getOutputSize());
	bn01.setMomentum(0.9f);
	info.push_back(bn01);

	LayerInfo act01(batch_size);
	act01.setLayerType(LayerType::Relu);
	act01.setInputSize(bn01.getOutputSize());
	act01.setOutputSize(bn01.getOutputSize());
	act01.setActivateType(ActivateType::Relu);
	info.push_back(act01);

	LayerInfo pooling0(batch_size);
	pooling0.setLayerType(LayerType::Pooling);
	pooling0.setInputSize(act01.getOutputSize());
	pooling0.setPoolingSize(LayerSize(2, 2, 1));
	pooling0.computePoolOutputSize();
	info.push_back(pooling0);

	//Part 2 : Conv - BN - Act - Conv(padding 2) - BN - Act - Pooling
	LayerInfo conv10(batch_size);
	conv10.setLayerType(LayerType::Convolution);
	conv10.setInputSize(pooling0.getOutputSize());
	conv10.setFilterSize(LayerSize(3, 3, pooling0.getOutputSize().getDepth()));
	conv10.setConvolutionInfo(32, 1, 1);
	conv10.setWeightInfo(WeightInitType::He, 0.01f);
	conv10.computeConvOutputSize();
	info.push_back(conv10);

	LayerInfo bn10(batch_size);
	bn10.setLayerType(LayerType::BatchNorm);
	bn10.setInputSize(conv10.getOutputSize());
	bn10.setOutputSize(conv10.getOutputSize());
	bn10.setMomentum(0.9f);
	info.push_back(bn10);

	LayerInfo act10(batch_size);
	act10.setLayerType(LayerType::Relu);
	act10.setInputSize(bn10.getOutputSize());
	act10.setOutputSize(bn10.getOutputSize());
	act10.setActivateType(ActivateType::Relu);
	info.push_back(act10);

	LayerInfo conv11(batch_size);
	conv11.setLayerType(LayerType::Convolution);
	conv11.setInputSize(act10.getOutputSize());
	conv11.setFilterSize(LayerSize(3, 3, act10.getOutputSize().getDepth()));
	conv11.setConvolutionInfo(32, 1, 2);
	conv11.setWeightInfo(WeightInitType::He, 0.01f);
	conv11.computeConvOutputSize();
	info.push_back(conv11);

	LayerInfo bn11(batch_size);
	bn11.setLayerType(LayerType::BatchNorm);
	bn11.setInputSize(conv11.getOutputSize());
	bn11.setOutputSize(conv11.getOutputSize());
	bn11.setMomentum(0.9f);
	info.push_back(bn11);

	LayerInfo act11(batch_size);
	act11.setLayerType(LayerType::Relu);
	act11.setInputSize(bn11.getOutputSize());
	act11.setOutputSize(bn11.getOutputSize());
	act11.setActivateType(ActivateType::Relu);
	info.push_back(act11);

	LayerInfo pooling1(batch_size);
	pooling1.setLayerType(LayerType::Pooling);
	pooling1.setInputSize(act11.getOutputSize());
	pooling1.setPoolingSize(LayerSize(2, 2, 1));
	pooling1.computePoolOutputSize();
	info.push_back(pooling1);

	//Part 3 : Conv - BN - Act - Conv - BN - Act - Pooling
	LayerInfo conv20(batch_size);
	conv20.setLayerType(LayerType::Convolution);
	conv20.setInputSize(pooling1.getOutputSize());
	conv20.setFilterSize(LayerSize(3, 3, pooling1.getOutputSize().getDepth()));
	conv20.setConvolutionInfo(64, 1, 1);
	conv20.setWeightInfo(WeightInitType::He, 0.01f);
	conv20.computeConvOutputSize();
	info.push_back(conv20);

	LayerInfo bn20(batch_size);
	bn20.setLayerType(LayerType::BatchNorm);
	bn20.setInputSize(conv20.getOutputSize());
	bn20.setOutputSize(conv20.getOutputSize());
	bn20.setMomentum(0.9f);
	info.push_back(bn20);

	LayerInfo act20(batch_size);
	act20.setLayerType(LayerType::Relu);
	act20.setInputSize(bn20.getOutputSize());
	act20.setOutputSize(bn20.getOutputSize());
	act20.setActivateType(ActivateType::Relu);
	info.push_back(act20);

	LayerInfo conv21(batch_size);
	conv21.setLayerType(LayerType::Convolution);
	conv21.setInputSize(act20.getOutputSize());
	conv21.setFilterSize(LayerSize(3, 3, act20.getOutputSize().getDepth()));
	conv21.setConvolutionInfo(64, 1, 1);
	conv21.setWeightInfo(WeightInitType::He, 0.01f);
	conv21.computeConvOutputSize();
	info.push_back(conv21);

	LayerInfo bn21(batch_size);
	bn21.setLayerType(LayerType::BatchNorm);
	bn21.setInputSize(conv21.getOutputSize());
	bn21.setOutputSize(conv21.getOutputSize());
	bn21.setMomentum(0.9f);
	info.push_back(bn21);

	LayerInfo act21(batch_size);
	act21.setLayerType(LayerType::Relu);
	act21.setInputSize(bn21.getOutputSize());
	act21.setOutputSize(bn21.getOutputSize());
	act21.setActivateType(ActivateType::Relu);
	info.push_back(act21);

	LayerInfo pooling2(batch_size);
	pooling2.setLayerType(LayerType::Pooling);
	pooling2.setInputSize(act21.getOutputSize());
	pooling2.setPoolingSize(LayerSize(2, 2, 1));
	pooling2.computePoolOutputSize();
	info.push_back(pooling2);

	//Part 4 : Affine - BN - Act - Dropout
	LayerInfo affine3(batch_size);
	affine3.setLayerType(LayerType::Affine);
	affine3.setInputSize(pooling2.getOutputSize());
	affine3.setOutputSize(LayerSize(256, 1, 1));
	affine3.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine3);

	LayerInfo bn3(batch_size);
	bn3.setLayerType(LayerType::BatchNorm);
	bn3.setInputSize(affine3.getOutputSize());
	bn3.setOutputSize(affine3.getOutputSize());
	bn3.setMomentum(0.9f);
	info.push_back(bn3);

	LayerInfo act3(batch_size);
	act3.setLayerType(LayerType::Relu);
	act3.setInputSize(bn3.getOutputSize());
	act3.setOutputSize(bn3.getOutputSize());
	act3.setActivateType(ActivateType::Relu);
	info.push_back(act3);

	LayerInfo dropout3(batch_size);
	dropout3.setLayerType(LayerType::DropOut);
	dropout3.setInputSize(act3.getOutputSize());
	dropout3.setOutputSize(act3.getOutputSize());
	dropout3.setDropOutInfo(0.3f, true);
	info.push_back(dropout3);

	//Part 5 : Affine - dropout - Softmax
	LayerInfo affine4(batch_size);
	affine4.setLayerType(LayerType::Affine);
	affine4.setInputSize(dropout3.getOutputSize());
	affine4.setOutputSize(LayerSize(10, 1, 1));
	affine4.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine4);

	LayerInfo dropout4(batch_size);
	dropout4.setLayerType(LayerType::DropOut);
	dropout4.setInputSize(affine4.getOutputSize());
	dropout4.setOutputSize(affine4.getOutputSize());
	dropout4.setDropOutInfo(0.3f, true);
	info.push_back(dropout4);

	LayerInfo softmax4(batch_size);
	softmax4.setLayerType(LayerType::SoftmaxWithLoss);
	softmax4.setInputSize(affine4.getOutputSize());
	softmax4.setOutputSize(affine4.getOutputSize());
	info.push_back(softmax4);
}

void DataUtil::create_cnn_info2(vector<LayerInfo>& info, int batch_size) {
	//part 1 : Conv - BN - Act - Pooling
	float momentum = 0.9f;

	LayerInfo conv00(batch_size);
	conv00.setLayerType(LayerType::Convolution);
	conv00.setInputSize(LayerSize(28, 28, 1));
	conv00.setFilterSize(LayerSize(3, 3, 1));
	conv00.setConvolutionInfo(32, 1, 1);
	conv00.setWeightInfo(WeightInitType::He, 0.01f);
	conv00.computeConvOutputSize();
	info.push_back(conv00);

	LayerInfo bn00(batch_size);
	bn00.setLayerType(LayerType::BatchNorm);
	bn00.setInputSize(conv00.getOutputSize());
	bn00.setOutputSize(conv00.getOutputSize());
	bn00.setMomentum(momentum);
	info.push_back(bn00);

	LayerInfo act00(batch_size);
	act00.setLayerType(LayerType::Relu);
	act00.setInputSize(bn00.getOutputSize());
	act00.setOutputSize(bn00.getOutputSize());
	act00.setActivateType(ActivateType::Relu);
	info.push_back(act00);

	///////////////////////////////////////////////////////
	LayerInfo dropout0(batch_size);
	dropout0.setLayerType(LayerType::DropOut);
	dropout0.setInputSize(act00.getOutputSize());
	dropout0.setOutputSize(act00.getOutputSize());
	dropout0.setDropOutInfo(0.1f, true);
	info.push_back(dropout0);
	///////////////////////////////////////////////////////

	LayerInfo pooling0(batch_size);
	pooling0.setLayerType(LayerType::Pooling);
	pooling0.setInputSize(act00.getOutputSize());
	pooling0.setPoolingSize(LayerSize(2, 2, 1));
	pooling0.computePoolOutputSize();
	info.push_back(pooling0);

	//Part 2 : Conv(padding 2) - BN - Act - Pooling
	LayerInfo conv10(batch_size);
	conv10.setLayerType(LayerType::Convolution);
	conv10.setInputSize(pooling0.getOutputSize());
	conv10.setFilterSize(LayerSize(3, 3, pooling0.getOutputSize().getDepth()));
	conv10.setConvolutionInfo(32, 1, 1);
	conv10.setWeightInfo(WeightInitType::He, 0.01f);
	conv10.computeConvOutputSize();
	info.push_back(conv10);

	LayerInfo bn10(batch_size);
	bn10.setLayerType(LayerType::BatchNorm);
	bn10.setInputSize(conv10.getOutputSize());
	bn10.setOutputSize(conv10.getOutputSize());
	bn10.setMomentum(momentum);
	info.push_back(bn10);

	LayerInfo act10(batch_size);
	act10.setLayerType(LayerType::Relu);
	act10.setInputSize(bn10.getOutputSize());
	act10.setOutputSize(bn10.getOutputSize());
	act10.setActivateType(ActivateType::Relu);
	info.push_back(act10);

	///////////////////////////////////////////////////////
	LayerInfo dropout1(batch_size);
	dropout1.setLayerType(LayerType::DropOut);
	dropout1.setInputSize(act10.getOutputSize());
	dropout1.setOutputSize(act10.getOutputSize());
	dropout1.setDropOutInfo(0.1f, true);
	info.push_back(dropout1);
	///////////////////////////////////////////////////////

	LayerInfo pooling1(batch_size);
	pooling1.setLayerType(LayerType::Pooling);
	pooling1.setInputSize(act10.getOutputSize());
	pooling1.setPoolingSize(LayerSize(2, 2, 1));
	pooling1.computePoolOutputSize();
	info.push_back(pooling1);

	//Part 3 : Conv - BN - Act - Pooling
	LayerInfo conv20(batch_size);
	conv20.setLayerType(LayerType::Convolution);
	conv20.setInputSize(pooling1.getOutputSize());
	conv20.setFilterSize(LayerSize(3, 3, pooling1.getOutputSize().getDepth()));
	conv20.setConvolutionInfo(64, 1, 1);
	conv20.setWeightInfo(WeightInitType::He, 0.01f);
	conv20.computeConvOutputSize();
	info.push_back(conv20);

	LayerInfo bn20(batch_size);
	bn20.setLayerType(LayerType::BatchNorm);
	bn20.setInputSize(conv20.getOutputSize());
	bn20.setOutputSize(conv20.getOutputSize());
	bn20.setMomentum(momentum);
	info.push_back(bn20);

	LayerInfo act20(batch_size);
	act20.setLayerType(LayerType::Relu);
	act20.setInputSize(bn20.getOutputSize());
	act20.setOutputSize(bn20.getOutputSize());
	act20.setActivateType(ActivateType::Relu);
	info.push_back(act20);

	///////////////////////////////////////////////////////
	LayerInfo dropout2(batch_size);
	dropout2.setLayerType(LayerType::DropOut);
	dropout2.setInputSize(act20.getOutputSize());
	dropout2.setOutputSize(act20.getOutputSize());
	dropout2.setDropOutInfo(0.1f, true);
	info.push_back(dropout2);
	///////////////////////////////////////////////////////

	LayerInfo pooling2(batch_size);
	pooling2.setLayerType(LayerType::Pooling);
	pooling2.setInputSize(act20.getOutputSize());
	pooling2.setPoolingSize(LayerSize(2, 2, 1));
	pooling2.computePoolOutputSize();
	info.push_back(pooling2);

	//Part 4 : Affine - BN - Act - Dropout
	LayerInfo affine3(batch_size);
	affine3.setLayerType(LayerType::Affine);
	affine3.setInputSize(pooling2.getOutputSize());
	affine3.setOutputSize(LayerSize(256, 1, 1));
	affine3.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine3);

	LayerInfo bn3(batch_size);
	bn3.setLayerType(LayerType::BatchNorm);
	bn3.setInputSize(affine3.getOutputSize());
	bn3.setOutputSize(affine3.getOutputSize());
	bn3.setMomentum(momentum);
	info.push_back(bn3);

	LayerInfo act3(batch_size);
	act3.setLayerType(LayerType::Relu);
	act3.setInputSize(bn3.getOutputSize());
	act3.setOutputSize(bn3.getOutputSize());
	act3.setActivateType(ActivateType::Relu);
	info.push_back(act3);

	LayerInfo dropout3(batch_size);
	dropout3.setLayerType(LayerType::DropOut);
	dropout3.setInputSize(act3.getOutputSize());
	dropout3.setOutputSize(act3.getOutputSize());
	dropout3.setDropOutInfo(0.5f, true);
	info.push_back(dropout3);

	//Part 5 : Affine - BN - Act - Dropout
	LayerInfo affine4(batch_size);
	affine4.setLayerType(LayerType::Affine);
	affine4.setInputSize(dropout3.getOutputSize());
	affine4.setOutputSize(LayerSize(64, 1, 1));
	affine4.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine4);

	LayerInfo bn4(batch_size);
	bn4.setLayerType(LayerType::BatchNorm);
	bn4.setInputSize(affine4.getOutputSize());
	bn4.setOutputSize(affine4.getOutputSize());
	bn4.setMomentum(momentum);
	info.push_back(bn4);

	LayerInfo act4(batch_size);
	act4.setLayerType(LayerType::Relu);
	act4.setInputSize(bn4.getOutputSize());
	act4.setOutputSize(bn4.getOutputSize());
	act4.setActivateType(ActivateType::Relu);
	info.push_back(act4);

	LayerInfo dropout4(batch_size);
	dropout4.setLayerType(LayerType::DropOut);
	dropout4.setInputSize(act4.getOutputSize());
	dropout4.setOutputSize(act4.getOutputSize());
	dropout4.setDropOutInfo(0.5f, true);
	info.push_back(dropout4);

	//Part 6 : Affine - dropout - Softmax
	LayerInfo affine5(batch_size);
	affine5.setLayerType(LayerType::Affine);
	affine5.setInputSize(dropout4.getOutputSize());
	affine5.setOutputSize(LayerSize(10, 1, 1));
	affine5.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine5);

	LayerInfo bn5(batch_size);
	bn5.setLayerType(LayerType::BatchNorm);
	bn5.setInputSize(affine5.getOutputSize());
	bn5.setOutputSize(affine5.getOutputSize());
	bn5.setMomentum(momentum);
	info.push_back(bn5);

	LayerInfo dropout5(batch_size);
	dropout5.setLayerType(LayerType::DropOut);
	dropout5.setInputSize(bn5.getOutputSize());
	dropout5.setOutputSize(bn5.getOutputSize());
	dropout5.setDropOutInfo(0.5f, false);
	//info.push_back(dropout5);

	LayerInfo softmax5(batch_size);
	softmax5.setLayerType(LayerType::SoftmaxWithLoss);
	softmax5.setInputSize(dropout5.getOutputSize());
	softmax5.setOutputSize(dropout5.getOutputSize());
	info.push_back(softmax5);
}

void DataUtil::create_cnn_info3(vector<LayerInfo>& info, int batch_size) {
	//part 1 : Conv - BN - Act - Pooling
	float momentum = 0.99f;

	LayerInfo conv00(batch_size);
	conv00.setLayerType(LayerType::Convolution);
	conv00.setInputSize(LayerSize(28, 28, 1));
	conv00.setFilterSize(LayerSize(3, 3, 1));
	conv00.setConvolutionInfo(32, 1, 1);
	conv00.setWeightInfo(WeightInitType::He, 0.01f);
	conv00.computeConvOutputSize();
	info.push_back(conv00);

	LayerInfo bn00(batch_size);
	bn00.setLayerType(LayerType::BatchNorm);
	bn00.setInputSize(conv00.getOutputSize());
	bn00.setOutputSize(conv00.getOutputSize());
	bn00.setMomentum(momentum);
	info.push_back(bn00);

	LayerInfo act00(batch_size);
	act00.setLayerType(LayerType::Relu);
	act00.setInputSize(bn00.getOutputSize());
	act00.setOutputSize(bn00.getOutputSize());
	act00.setActivateType(ActivateType::Relu);
	info.push_back(act00);

	///////////////////////////////////////////////////////
	LayerInfo dropout0(batch_size);
	dropout0.setLayerType(LayerType::DropOut);
	dropout0.setInputSize(act00.getOutputSize());
	dropout0.setOutputSize(act00.getOutputSize());
	dropout0.setDropOutInfo(0.1f, true);
	info.push_back(dropout0);
	///////////////////////////////////////////////////////

	LayerInfo pooling0(batch_size);
	pooling0.setLayerType(LayerType::Pooling);
	pooling0.setInputSize(act00.getOutputSize());
	pooling0.setPoolingSize(LayerSize(2, 2, 1));
	pooling0.computePoolOutputSize();
	info.push_back(pooling0);

	//14 x 14
	//Part 2 : Conv(padding 2) - BN - Act - Pooling
	LayerInfo conv10(batch_size);
	conv10.setLayerType(LayerType::Convolution);
	conv10.setInputSize(pooling0.getOutputSize());
	conv10.setFilterSize(LayerSize(3, 3, pooling0.getOutputSize().getDepth()));
	conv10.setConvolutionInfo(64, 1, 1);
	conv10.setWeightInfo(WeightInitType::He, 0.01f);
	conv10.computeConvOutputSize();
	info.push_back(conv10);

	LayerInfo bn10(batch_size);
	bn10.setLayerType(LayerType::BatchNorm);
	bn10.setInputSize(conv10.getOutputSize());
	bn10.setOutputSize(conv10.getOutputSize());
	bn10.setMomentum(momentum);
	info.push_back(bn10);

	LayerInfo act10(batch_size);
	act10.setLayerType(LayerType::Relu);
	act10.setInputSize(bn10.getOutputSize());
	act10.setOutputSize(bn10.getOutputSize());
	act10.setActivateType(ActivateType::Relu);
	info.push_back(act10);

	///////////////////////////////////////////////////////
	LayerInfo dropout1(batch_size);
	dropout1.setLayerType(LayerType::DropOut);
	dropout1.setInputSize(act10.getOutputSize());
	dropout1.setOutputSize(act10.getOutputSize());
	dropout1.setDropOutInfo(0.1f, true);
	info.push_back(dropout1);
	///////////////////////////////////////////////////////

	LayerInfo pooling1(batch_size);
	pooling1.setLayerType(LayerType::Pooling);
	pooling1.setInputSize(act10.getOutputSize());
	pooling1.setPoolingSize(LayerSize(2, 2, 1));
	pooling1.computePoolOutputSize();
	info.push_back(pooling1);

	//7 x 7
	//Part 3 : Conv - BN - Act - Pooling
	LayerInfo conv20(batch_size);
	conv20.setLayerType(LayerType::Convolution);
	conv20.setInputSize(pooling1.getOutputSize());
	conv20.setFilterSize(LayerSize(4, 4, pooling1.getOutputSize().getDepth()));
	conv20.setConvolutionInfo(128, 1, 1);
	conv20.setWeightInfo(WeightInitType::He, 0.01f);
	conv20.computeConvOutputSize();
	info.push_back(conv20);

	LayerInfo bn20(batch_size);
	bn20.setLayerType(LayerType::BatchNorm);
	bn20.setInputSize(conv20.getOutputSize());
	bn20.setOutputSize(conv20.getOutputSize());
	bn20.setMomentum(momentum);
	info.push_back(bn20);

	LayerInfo act20(batch_size);
	act20.setLayerType(LayerType::Relu);
	act20.setInputSize(bn20.getOutputSize());
	act20.setOutputSize(bn20.getOutputSize());
	act20.setActivateType(ActivateType::Relu);
	info.push_back(act20);

	///////////////////////////////////////////////////////
	LayerInfo dropout2(batch_size);
	dropout2.setLayerType(LayerType::DropOut);
	dropout2.setInputSize(act20.getOutputSize());
	dropout2.setOutputSize(act20.getOutputSize());
	dropout2.setDropOutInfo(0.1f, true);
	info.push_back(dropout2);
	///////////////////////////////////////////////////////

	LayerInfo pooling2(batch_size);
	pooling2.setLayerType(LayerType::Pooling);
	pooling2.setInputSize(act20.getOutputSize());
	pooling2.setPoolingSize(LayerSize(2, 2, 1));
	pooling2.computePoolOutputSize();
	info.push_back(pooling2);

	//Part 4 : Affine - BN - Act - Dropout
	LayerInfo affine3(batch_size);
	affine3.setLayerType(LayerType::Affine);
	affine3.setInputSize(pooling2.getOutputSize());
	affine3.setOutputSize(LayerSize(256, 1, 1));
	affine3.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine3);

	LayerInfo bn3(batch_size);
	bn3.setLayerType(LayerType::BatchNorm);
	bn3.setInputSize(affine3.getOutputSize());
	bn3.setOutputSize(affine3.getOutputSize());
	bn3.setMomentum(momentum);
	info.push_back(bn3);

	LayerInfo act3(batch_size);
	act3.setLayerType(LayerType::Relu);
	act3.setInputSize(bn3.getOutputSize());
	act3.setOutputSize(bn3.getOutputSize());
	act3.setActivateType(ActivateType::Relu);
	info.push_back(act3);

	LayerInfo dropout3(batch_size);
	dropout3.setLayerType(LayerType::DropOut);
	dropout3.setInputSize(act3.getOutputSize());
	dropout3.setOutputSize(act3.getOutputSize());
	dropout3.setDropOutInfo(0.5f, true);
	info.push_back(dropout3);

	//Part 5 : Affine - BN - Act - Dropout
	LayerInfo affine4(batch_size);
	affine4.setLayerType(LayerType::Affine);
	affine4.setInputSize(dropout3.getOutputSize());
	affine4.setOutputSize(LayerSize(64, 1, 1));
	affine4.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine4);

	LayerInfo bn4(batch_size);
	bn4.setLayerType(LayerType::BatchNorm);
	bn4.setInputSize(affine4.getOutputSize());
	bn4.setOutputSize(affine4.getOutputSize());
	bn4.setMomentum(momentum);
	info.push_back(bn4);

	//LayerInfo softmax4(batch_size);
	//softmax4.setLayerType(LayerType::SoftmaxWithLoss);
	//softmax4.setInputSize(bn4.getOutputSize());
	//softmax4.setOutputSize(bn4.getOutputSize());
	//info.push_back(softmax4);

	LayerInfo act4(batch_size);
	act4.setLayerType(LayerType::Relu);
	act4.setInputSize(bn4.getOutputSize());
	act4.setOutputSize(bn4.getOutputSize());
	act4.setActivateType(ActivateType::Relu);
	info.push_back(act4);

	LayerInfo dropout4(batch_size);
	dropout4.setLayerType(LayerType::DropOut);
	dropout4.setInputSize(act4.getOutputSize());
	dropout4.setOutputSize(act4.getOutputSize());
	dropout4.setDropOutInfo(0.5f, true);
	info.push_back(dropout4);

	//Part 6 : Affine - dropout - Softmax
	LayerInfo affine5(batch_size);
	affine5.setLayerType(LayerType::Affine);
	affine5.setInputSize(dropout4.getOutputSize());
	affine5.setOutputSize(LayerSize(10, 1, 1));
	affine5.setWeightInfo(WeightInitType::He, 0.01f);
	info.push_back(affine5);

	LayerInfo bn5(batch_size);
	bn5.setLayerType(LayerType::BatchNorm);
	bn5.setInputSize(affine5.getOutputSize());
	bn5.setOutputSize(affine5.getOutputSize());
	bn5.setMomentum(momentum);
	info.push_back(bn5);

	LayerInfo softmax5(batch_size);
	softmax5.setLayerType(LayerType::SoftmaxWithLoss);
	softmax5.setInputSize(bn5.getOutputSize());
	softmax5.setOutputSize(bn5.getOutputSize());
	info.push_back(softmax5);
}


void DataUtil::create_layer(LayerInfo& info, Layer **layer) {
	LayerType type = info.getLayerType();
	switch (type)
	{
	case LayerType::Affine: {
		*layer = new AffineBatchLayer();
		break;
	}
	case LayerType::BatchNorm: {
		*layer = new BatchNormLayer();
		break;
	}
	case LayerType::Convolution: {
		*layer = new ConvolutionBatchLayer();
		break;
	}
	case LayerType::DropOut: {
		*layer = new DropOutLayer();
		break;
	}
	case LayerType::Pooling: {
		*layer = new MaxPoolingBatchLayer();
		break;
	}
	case LayerType::Relu: {
	}
	case LayerType::Sigmoid: {
		*layer = new ActivateBatchLayer();
		break;
	}
	case LayerType::SoftmaxWithLoss: {
		*layer = new SoftmaxWithLossBatchLayer();
		break;
	}
	default:
		break;
	}
}