#pragma once

typedef enum class _MatConsType {
	OnlyCopy,
	Transpose,
	Reverse
}MatConsType;

typedef enum class _MatCommType {
	Dot,
	Plus,
	Minus,
	Multiple,
	Divide
}MatCommType;

typedef enum class _NumCommType {
	Plus,
	Minus,
	Multiple,
	Divide,
	Square,
	Sqrt,
	Inverse
}NumCommType;

const int ACTIVATE_LAYER = 0x00010000;
const int NON_UPDATABLE_LAYER = 0x00100000;
const int UPDATABLE_LAYER = 0x01000000;

typedef enum class _LayerType {
	Relu = 0x00010001,
	Sigmoid = 0x00010002,
	SoftmaxWithLoss = 0x00010004,
	DropOut = 0x00100001,
	Pooling = 0x00100002,
	Affine = 0x01000001,
	BatchNorm = 0x01000002,
	Convolution = 0x01000004
}LayerType;

typedef enum class _WeightInitType {
	None,
	Normal,
	Xavier,
	He
}WeightInitType;

typedef enum class _ActivateType {
	None,
	Sigmoid,
	Relu,
}ActivateType;
