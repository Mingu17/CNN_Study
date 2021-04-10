#pragma once

#include "Common.h"
#include "LayerSize.h"

class Matrix
{
public:
	Matrix();
	Matrix(const int _row, const int _col, const int _depth, const float f = 0.0f);
	Matrix(const int _row, const int _col, const int _depth, const float *f_arr);
	Matrix(const LayerSize& size, const float init = 0.0f);
	Matrix(const LayerSize& size, const float *init_arr);
	Matrix(const Matrix& src, MatConsType consType = MatConsType::OnlyCopy);
	virtual ~Matrix();

	void init(const int _row, const int _col, const int _depth, const float f = 0.0f);
	void init(const int _row, const int _col, const int _depth, const float *f);
	void init(const LayerSize& size, const float f = 0.0f);
	void init(const LayerSize& size, const float* f_arr);
	void setZero();
	void release();

	float getElement(const int _row, const int _col, const int _depth = 1) const;
	float getElement(const int idx) const;

	void setElement(const float f, const int _row, const int _col, const int _depth = 1);
	void setElement(const float f, const int idx);

	float* getData() const;
	float* getPlaneData(const int _depth = 0) const;
	float* getRowData(const int _row, const int _depth = 0) const;
	
	int getRowLen() const;
	int getColLen() const;
	int getDepthLen() const;
	int getTotalLen() const;
	int getLenPerLayer() const;
	LayerSize getTotalSize() const;

	void printData();

	float getMax(int _depth = -1) const;
	float getMin(int _depth = -1) const;
	float getSum(int _depth = -1) const;

	//operator
	Matrix& operator=(const Matrix& src);
	Matrix& operator+=(const Matrix& src);
	Matrix& operator-=(const Matrix& src);
	Matrix& operator*=(const Matrix& src);

	Matrix& operator*=(const float f);
	Matrix& operator+=(const float f);
	Matrix& operator-=(const float f);
	Matrix& operator/=(const float f);

	const Matrix dot(const Matrix& src) const;
	const Matrix operator+(const Matrix& src) const;
	const Matrix operator-(const Matrix& src) const;
	const Matrix operator*(const Matrix& src) const;
	const Matrix operator/(const Matrix& src) const;

	const Matrix operator+(const float f) const;
	const Matrix operator-(const float f) const;
	const Matrix operator*(const float f) const;
	const Matrix operator/(const float f) const;
	const Matrix squared() const;
	const Matrix sqrt(const float add_f = 0.0f) const;
	const Matrix inverse() const;

	void elemComputeSelf(const Matrix& src, float(*commFunc)(float));
	void elemComputeSelf(float(*commFunc)(float));

	void reshape(const int new_row, const int new_col, const int new_depth = 1);
	void reshape(const LayerSize& size);

protected:
	Matrix(const Matrix& src1, const Matrix& src2, const MatCommType commType);
	Matrix(const Matrix& src, const NumCommType commType, const float f);
	
protected:
	float *data;
	int row;
	int col;
	int depth;
	int length;
	int lengthPerLayer;
};