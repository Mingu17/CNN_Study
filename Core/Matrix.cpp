#include <iostream>
#include "Matrix.h"
#include "NumUtil.h"
#include <intrin.h>

Matrix::Matrix() : row(0), col(0), depth(0), length(0), lengthPerLayer(0), data(0) {

}

Matrix::Matrix(const int _row, const int _col, const int _depth, const float f) {
	init(_row, _col, _depth, f);
}

Matrix::Matrix(const int _row, const int _col, const int _depth, const float *f_arr) {
	init(_row, _col, _depth, f_arr);
}

Matrix::Matrix(const LayerSize& size, const float f) {
	init(size.getRow(), size.getCol(), size.getDepth(), f);
}

Matrix::Matrix(const LayerSize& size, const float *f_arr) {
	init(size.getRow(), size.getCol(), size.getDepth(), f_arr);
}

Matrix::Matrix(const Matrix& src, MatConsType consType) {
	switch (consType)
	{
	case MatConsType::OnlyCopy: {
		init(src.getRowLen(), src.getColLen(), src.getDepthLen(), src.getData());
		break;
	}
	case MatConsType::Transpose: { //per plane
		init(src.getColLen(), src.getRowLen(), src.getDepthLen());

		for (int d = 0; d < depth; ++d) {
			float *srcData = src.getPlaneData();
			float *dstData = getPlaneData();
			int sz = 0;

			for (int i = 0; i < col; ++i) {
				int _col = i;
				for (int j = 0; j < row; ++j) {
					dstData[sz++] = srcData[_col];
					_col += col;
				}
			}
		}
		break;
	}
	case MatConsType::Reverse: { //per plane	
		init(src.getRowLen(), src.getColLen(), src.getDepthLen());

		for (int d = 0; d < depth; ++d) {
			float *srcData = src.getPlaneData(d);
			float *dstData = getPlaneData(d);

			for (int i = 0, j = lengthPerLayer - 1; i < lengthPerLayer; ++i, --j) {
				//dstData[i] = srcData[lengthPerLayer - i - 1];
				dstData[i] = srcData[j];
			}
		}
		break;
	}
	default:
		break;
	}
}

Matrix::~Matrix() {
	release();
}

void Matrix::init(const int _row, const int _col, const int _depth, const float f) {
	row = _row;
	col = _col;
	depth = _depth;
	lengthPerLayer = row * col;
	length = lengthPerLayer * depth;
	data = new float[length];

	int etc = length % 8;
	int lengthRight = length - etc;

	if (f == 0.0f) {
		//memset(data, 0, sizeof(float)*length);
		for (int i = 0; i < lengthRight; i += 8) {
			float *tdata = data + i;
			_mm256_storeu_ps(tdata, _mm256_setzero_ps());
		}
	}
	else {
		__m256 var = _mm256_set1_ps(f);
		for (int i = 0; i < lengthRight; i += 8) {
			float *tdata = data + i;
			_mm256_storeu_ps(tdata, var);
		}
	}
	for (int i = lengthRight; i < length; ++i) data[i] = f;
}

void Matrix::init(const int _row, const int _col, const int _depth, const float *f) {
	if (f == 0) {
		throw std::invalid_argument("[ERROR] data is null");
	}
	row = _row;
	col = _col;
	depth = _depth;
	lengthPerLayer = row * col;
	length = lengthPerLayer * depth;
	data = new float[length];

	memcpy_s(data, sizeof(float)*length, f, sizeof(float)*length);
}

void Matrix::init(const LayerSize& size, const float f) {
	init(size.getRow(), size.getCol(), size.getDepth(), f);
}

void Matrix::init(const LayerSize& size, const float *f_arr) {
	init(size.getRow(), size.getCol(), size.getDepth(), f_arr);
}

void Matrix::setZero() {
	if (data) {
		//memset(data, 0, sizeof(float)*length);
		__m256 zero = _mm256_setzero_ps();
		int etc = length % 8;
		int lengthRight = length - etc;

		for (int i = 0; i < lengthRight; i += 8) {
			float *tdata = data + i;
			_mm256_storeu_ps(tdata, zero);
		}
		for (int i = lengthRight; i < length; ++i) data[i] = 0.0f;
	}
}

void Matrix::release() {
	delete[] data;
	row = 0;
	col = 0;
	depth = 0;
	length = 0;
	lengthPerLayer = 0;
}

float Matrix::getElement(const int _row, const int _col, const int _depth) const {
	int loc = (lengthPerLayer * _depth) + (_row * col) + _col;
	return data[loc];
}

float Matrix::getElement(const int idx) const {
	return data[idx];
}

void Matrix::setElement(const float f, const int _row, const int _col, const int _depth) {
	int loc = (lengthPerLayer * _depth) + (_row * col) + _col;
	data[loc] = f;
}

void Matrix::setElement(const float f, const int idx) {
	data[idx] = f;
}

float* Matrix::getData() const {
	return data;
}

float* Matrix::getPlaneData(const int _depth) const {
	int loc = lengthPerLayer * _depth;
	return data + loc;
}

float* Matrix::getRowData(const int _row, const int _depth) const {
	int loc = lengthPerLayer * _depth + _row * col;
	return data + loc;
}

int Matrix::getRowLen() const {
	return row;
}

int Matrix::getColLen() const {
	return col;
}

int Matrix::getDepthLen() const {
	return depth;
}

int Matrix::getTotalLen() const {
	return length;
}

int Matrix::getLenPerLayer() const {
	return lengthPerLayer;
}

LayerSize Matrix::getTotalSize() const {
	return LayerSize(row, col, depth);
}

void Matrix::printData() {
	std::cout << "====================== Data Start =======================" << std::endl;
	for (int d = 0; d < depth; ++d) {
		std::cout << "****************** Plane : " << d << "******************" << std::endl;
		float *p = getPlaneData(d);
		for (int r = 0; r < row; ++r) {
			int loc = r * col;
			for (int c = 0; c < col; ++c) {
				std::cout << p[loc + c] << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << "====================== Data End =======================" << std::endl;
}

float Matrix::getMax(int _depth) const {
	float max = -FLT_MAX;
	if (_depth == -1) { //all
		for (int i = 0; i < length; ++i) {
			if (max < data[i]) max = data[i];
		}
	}
	else {
		//float *ref = dataRef[_depth].ref[0];
		float *ref = getPlaneData(_depth);
		for (int i = 0; i < lengthPerLayer; ++i) {
			if (max < ref[i]) max = ref[i];
		}
	}
	return max;
}

float Matrix::getMin(int _depth) const {
	float min = FLT_MAX;
	if (_depth == -1) { //all
		for (int i = 0; i < length; ++i) {
			if (min > data[i]) min = data[i];
		}
	}
	else {
		//float *ref = dataRef[_depth].ref[0];
		float *ref = getPlaneData(_depth);
		for (int i = 0; i < lengthPerLayer; ++i) {
			if (min > ref[i]) min = ref[i];
		}
	}
	return min;
}

float Matrix::getSum(int _depth) const {
	float sum = 0.0f;
	if (_depth == -1) { //all
		for (int i = 0; i < length; ++i) {
			sum += data[i];
		}
	}
	else {
		//float *ref = dataRef[_depth].ref[0];
		float *ref = getPlaneData(_depth);
		for (int i = 0; i < lengthPerLayer; ++i) {
			sum += ref[i];
		}
	}
	return sum;
}

//operator
Matrix& Matrix::operator=(const Matrix& src) {
	if (row != src.getRowLen() || col != src.getColLen() || depth != src.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float *srcData = src.getData();
	int totalSize = sizeof(float) * length;
	memcpy_s(data, totalSize, srcData, totalSize);
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& src) {
	if (row != src.getRowLen() || col != src.getColLen() || depth != src.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float *srcData = src.getData();
	//for (int i = 0; i < length; ++i) data[i] += srcData[i];
	NumUtil::op_default(data, data, srcData, length, MatCommType::Plus);
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& src) {
	if (row != src.getRowLen() || col != src.getColLen() || depth != src.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float *srcData = src.getData();
	//for (int i = 0; i < length; ++i) data[i] -= srcData[i];
	NumUtil::op_default(data, data, srcData, length, MatCommType::Minus);
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& src) {
	if (row != src.getRowLen() || col != src.getColLen() || depth != src.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float *srcData = src.getData();
	//for (int i = 0; i < length; ++i) data[i] -= srcData[i];
	NumUtil::op_default(data, data, srcData, length, MatCommType::Multiple);
	return *this;
}

Matrix& Matrix::operator*=(const float f) {
	if (row == 0 || col == 0 || depth == 0) {
		throw std::invalid_argument("[Error] size is zero");
	}
	//for (int i = 0; i < length; ++i) data[i] *= f;
	NumUtil::op_default_one(data, data, f, length, NumCommType::Multiple);
	return *this;
}

Matrix& Matrix::operator+=(const float f) {
	if (row == 0 || col == 0 || depth == 0) {
		throw std::invalid_argument("[Error] size is zero");
	}
	//for (int i = 0; i < length; ++i) data[i] += f;
	NumUtil::op_default_one(data, data, f, length, NumCommType::Plus);
	return *this;
}

Matrix& Matrix::operator-=(const float f) {
	if (row == 0 || col == 0 || depth == 0) {
		throw std::invalid_argument("[Error] size is zero");
	}
	//for (int i = 0; i < length; ++i) data[i] -= f;
	NumUtil::op_default_one(data, data, f, length, NumCommType::Minus);
	return *this;
}

Matrix& Matrix::operator/=(const float f) {
	if (row == 0 || col == 0 || depth == 0) {
		throw std::invalid_argument("[Error] size is zero");
	}
	if (f == 0.0f) {
		throw std::invalid_argument("[Error] f is zero");
	}
	//for (int i = 0; i < length; ++i) data[i] /= f;
	NumUtil::op_default_one(data, data, f, length, NumCommType::Divide);
	return *this;
}

const Matrix Matrix::dot(const Matrix& src) const {
	if (col != src.getRowLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	int matDepth = src.getDepthLen();
	if (depth == matDepth || matDepth == 1) {
		return Matrix(*this, src, MatCommType::Dot);
	}
	else {
		throw std::invalid_argument("[Error] depth size mismatch (right value : 1 or src depth)");
	}
}

const Matrix Matrix::operator+(const Matrix& src) const {
	if (row != src.getRowLen() || col != src.getColLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	int matDepth = src.getDepthLen();
	if (depth == matDepth || matDepth == 1) {
		return Matrix(*this, src, MatCommType::Plus);
	}
	else {
		throw std::invalid_argument("[Error] depth size mismatch (right value : 1 or src depth)");
	}
}

const Matrix Matrix::operator-(const Matrix& src) const {
	if (row != src.getRowLen() || col != src.getColLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	int matDepth = src.getDepthLen();
	if (depth == matDepth || matDepth == 1) {
		return Matrix(*this, src, MatCommType::Minus);
	}
	else {
		throw std::invalid_argument("[Error] depth size mismatch (right value : 1 or src depth)");
	}
}

const Matrix Matrix::operator*(const Matrix& src) const {
	if (row != src.getRowLen() || col != src.getColLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	int matDepth = src.getDepthLen();
	if (depth == matDepth || matDepth == 1) {
		return Matrix(*this, src, MatCommType::Multiple);
	}
	else {
		throw std::invalid_argument("[Error] depth size mismatch (right value : 1 or src depth)");
	}
}

const Matrix Matrix::operator/(const Matrix& src) const {
	if (row != src.getRowLen() || col != src.getColLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	int matDepth = src.getDepthLen();
	if (depth == matDepth || matDepth == 1) {
		return Matrix(*this, src, MatCommType::Divide);
	}
	else {
		throw std::invalid_argument("[Error] depth size mismatch (right value : 1 or src depth)");
	}
}

const Matrix Matrix::operator+(const float f) const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Plus, f);
}

const Matrix Matrix::operator-(const float f) const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Minus, f);
}

const Matrix Matrix::operator*(const float f) const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Multiple, f);
}

const Matrix Matrix::operator/(const float f) const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Divide, f);
}

const Matrix Matrix::squared() const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Square, 0.0f);
}

const Matrix Matrix::sqrt(const float add_f) const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Sqrt, add_f);
}

const Matrix Matrix::inverse() const {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	return Matrix(*this, NumCommType::Inverse, 0.0f);
}

void Matrix::elemComputeSelf(const Matrix& src, float(*commFunc)(float)) {
	if (row != src.getRowLen() || col != src.getColLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	if (depth != src.getDepthLen()) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	float *srcData = src.getData();
	for (int i = 0; i < length; ++i) data[i] = commFunc(srcData[i]);
}

void Matrix::elemComputeSelf(float(*commFunc)(float)) {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	for (int i = 0; i < length; ++i) data[i] = commFunc(data[i]);
}

void Matrix::reshape(const int new_row, const int new_col, const int new_depth) {
	if (row <= 0 || col <= 0 || depth < 1) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	int newLen = new_row * new_col * new_depth;
	if (newLen != length) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	row = new_row;
	col = new_col;
	depth = new_depth;
	lengthPerLayer = new_row * new_col;
	length = lengthPerLayer * new_depth;
}

void Matrix::reshape(const LayerSize& size) {
	int newLen = size.getLen();
	if (newLen == 0) {
		throw std::invalid_argument("[Error] Matrix size is zero");
	}
	if (newLen != length) {
		throw std::invalid_argument("[Error] size mismatch");
	}
	row = size.getRow();// new_row;
	col = size.getCol();// new_col;
	depth = size.getDepth();// new_depth;
	lengthPerLayer = size.getLenPerPlane();// new_row * new_col;
	length = size.getLen();// lengthPerLayer * new_depth;
}

/////////////////////////////////////////////////
Matrix::Matrix(const Matrix& src1, const Matrix& src2, const MatCommType commType) {
	if (commType == MatCommType::Dot) {
		int src1Row = src1.getRowLen();
		int src1Col = src1.getColLen();
		int src1Depth = src1.getDepthLen();
		int src2Row = src2.getRowLen();
		int src2Col = src2.getColLen();
		int src2Depth = src2.getDepthLen();

		init(src1Row, src2Col, src1Depth);

		for (int d = 0; d < src1Depth; ++d) {
			float *src1Ptr = src1.getPlaneData(d);
			float *src2Ptr = 0;

			if (src1Depth == src2Depth) src2Ptr = src2.getPlaneData(d);
			else src2Ptr = src2.getPlaneData();

			float *dst = getPlaneData(d);
			int i, j, k, sz = 0;

			for (i = 0; i < src1Row; ++i) {
				for (j = 0; j < src2Col; ++j) {
					float locVal = 0.0f;
					float *s2 = src2Ptr + j;
					for (k = 0; k < src1Col; ++k) {
						locVal += (src1Ptr[k] * s2[0]);
						s2 = s2 + src2Col;
					}
					dst[sz++] = locVal;

				}
				src1Ptr = src1Ptr + src1Col;
			}
		}
	}
	else {
		int src1Row = src1.getRowLen();
		int src1Col = src1.getColLen();
		int src1Depth = src1.getDepthLen();
		int src2Depth = src2.getDepthLen();
		float *src1Ptr, *src2Ptr, *dst;
		init(src1Row, src1Col, src1Depth);

		if (src2Depth == 1) {
			src2Ptr = src2.getData();
			for (int d = 0; d < src1Depth; ++d) {
				src1Ptr = src1.getPlaneData(d);
				dst = getPlaneData(d);
				//int dIdx = (src1Depth == src2Depth) ? d : 0;
				//src2Ptr = src2.getPlaneData(dIdx);
				NumUtil::op_default(dst, src1Ptr, src2Ptr, lengthPerLayer, commType);
			}
		}
		else {
			src1Ptr = src1.getData();
			src2Ptr = src2.getData();
			NumUtil::op_default(data, src1Ptr, src2Ptr, length, commType);
		}
	}
}

Matrix::Matrix(const Matrix& src, const NumCommType commType, const float f) {
	init(src.getRowLen(), src.getColLen(), src.getDepthLen());
	float *srcData = src.getData();
	NumUtil::op_default_one(data, srcData, f, length, commType);
}