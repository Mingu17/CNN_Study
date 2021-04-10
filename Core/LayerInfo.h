#pragma once

#include "Common.h"

class LayerInfo
{
public:
	LayerInfo() {}
	LayerInfo(int batch_size) {
		batchSize = batch_size;
	}

	virtual ~LayerInfo() {}

	//setter
	void setLayerType(LayerType type) {
		layerType = type;
	}

	void setInputSize(LayerSize size) {
		inputSize = size;
	}

	void setOutputSize(LayerSize size) {
		outputSize = size;
	}

	void setFilterSize(LayerSize size) {
		filterSize = size;
	}

	void setPoolingSize(LayerSize size) {
		poolingSize = size;
	}

	void setBatchSize(int batch_size) {
		batchSize = batch_size;
	}
	
	void setWeightInfo(WeightInitType wType, float weight_factor) {
		weightType = wType;
		weightFactor = weight_factor;
	}

	void setDropOutInfo(float drop_ratio, bool isFwdMultiple = false) {
		dropRatio = drop_ratio;
		fwdMultiple = isFwdMultiple;
	}

	void setActivateType(ActivateType aType) {
		actType = aType;
	}

	void setMomentum(float _momentum) {
		momentum = _momentum;
	}

	void setConvolutionInfo(int filter_count, int _stride, int _padding) {
		filterCount = filter_count;
		stride = _stride;
		padding = _padding;
	}

	void computeConvOutputSize() {
		int out_h = 1 + ((inputSize.getRow() + 2 * padding - filterSize.getRow()) / stride);
		int out_w = 1 + ((inputSize.getCol() + 2 * padding - filterSize.getCol()) / stride);
		outputSize = LayerSize(out_h, out_w, filterCount);
	}

	void computePoolOutputSize() {
		stride = poolingSize.getCol(); //юс╫ц
		int out_h = 1 + (inputSize.getRow() - poolingSize.getRow()) / stride;
		int out_w = 1 + (inputSize.getCol() - poolingSize.getCol()) / stride;
		outputSize = LayerSize(out_h, out_w, inputSize.getDepth());
	}
	//getter
	LayerType getLayerType() { return layerType; }
	LayerSize& getInputSize() { return inputSize; }
	LayerSize& getOutputSize() { return outputSize; }
	LayerSize& getFilterSize() { return filterSize; }
	LayerSize& getPoolingSize() { return poolingSize; }

	int getBatchSize() { return batchSize; }

	WeightInitType getWeightInitType() { return weightType; }
	float getWeightFactor() { return weightFactor; }

	float getDropRatio() { return dropRatio; }
	bool getIsFwdMultiple() { return fwdMultiple; }

	ActivateType getActivateType() { return actType; }
	float getMomentum() { return momentum; }

	int getFilterCount() { return filterCount; }
	int getStride() { return stride; }
	int getPadding() { return padding; }

protected:
	LayerType layerType;

	LayerSize inputSize;
	LayerSize outputSize;
	LayerSize filterSize;
	LayerSize poolingSize;

	int batchSize;

	WeightInitType weightType;
	float weightFactor;

	float dropRatio;
	bool fwdMultiple;

	ActivateType actType;
	float momentum;

	int filterCount;
	int stride;
	int padding;
};