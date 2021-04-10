#pragma once

#include "Layer.h"

class SingleLayer : public Layer
{
public:
	SingleLayer(LayerType type) : Layer(type) {}
	virtual void backward(Matrix& d) = 0;

	Matrix& getDout() { return dout; }
protected:
	Matrix dout;
};