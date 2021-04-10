#pragma once

class LayerSize 
{
public:
	LayerSize() : row(0), col(0), depth(0) {}
	LayerSize(int _row, int _col, int _depth = 1) {
		init(_row, _col, _depth);
	}

	virtual ~LayerSize() {

	}
	
	void init(int _row, int _col, int _depth = 1) {
		row = _row;
		col = _col;
		depth = _depth;
		lengthPerPlane = _row * _col;
		length = lengthPerPlane * _depth;
	}

	int getRow() const { return row; }
	int getCol() const { return col; }
	int getDepth() const { return depth; }
	int getLen() const { return length; }
	int getLenPerPlane() const { return lengthPerPlane; }

	LayerSize& operator=(const LayerSize& size) {
		init(size.getRow(), size.getCol(), size.getDepth());
		return *this;
	}

protected:
	int row;
	int col;
	int depth;
	int length;
	int lengthPerPlane;
};