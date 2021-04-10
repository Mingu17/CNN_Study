#pragma once

#include "../Core/Matrix.h"
#include "../Core/LayerSize.h"
#include "../Core/LayerInfo.h"

class Layer
{
public:
	Layer(LayerType type) {
		layerType = type;
	}

	//virtual void init(LayerSize& input_size, LayerSize& out_size, LayerInfo& info) = 0;
	virtual void init(LayerInfo& info) = 0;
	virtual void forward(Matrix& x) = 0;

	LayerType getLayerType() { return layerType; }
	Matrix& getOut() { return out; }

	LayerSize& getOutSize() { return outSize; }
	LayerSize& getInSize() { return inSize; }

protected:
	LayerType layerType;
	Matrix out;

	LayerSize outSize;
	LayerSize inSize;
};