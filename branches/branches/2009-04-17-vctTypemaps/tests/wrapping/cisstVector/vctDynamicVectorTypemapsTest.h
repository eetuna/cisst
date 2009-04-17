
#include <cisstVector.h>
#include <iostream>

// Always include last
//#include "vctTestPythonExport.h"

class vctDynamicVectorTypemapsTest
{

protected:

	vctDynamicVector<int> vector;

public:

	// Resize the input vector by factor
	unsigned int ResizeVector(vctDynamicVector<int> &inputVec, double factor)
	{
		inputVec.resize( (int)(factor * inputVec.size()) );

		return inputVec.size();
	}

    // Change the value at index to newValue in input vector
    void ModifyVectorData(vctDynamicVector<int> &inputVec, int index, int newValue)
    {
        inputVec[index] = newValue;
    }

	void SetByCopyDVRef(vctDynamicVectorRef<int> in)
	{
		vector.SetSize(in.size());
		vector.Assign(in);
	}

	void SetByCopyConstDVRef(vctDynamicConstVectorRef<int> in)
	{
		vector.SetSize(in.size());
		vector.Assign(in);
	}

	// Set vector to the input vector, in, by copy
	void SetByCopy(vctDynamicVector<int> in)
	{
		vector.SetSize(in.size());
		vector.Assign(in);
	}

	// Set vector to the input vector, in, by reference
	void SetByRef(vctDynamicVector<int> &in)
	{
		vector.SetSize(in.size());
		vector = in;
	}

    // Set vector to the input vector, in, by const Ref &
    void SetByConstRefAmp(const vctDynamicVectorRef<int> &in)
    {
        vector.SetSize(in.size());
        vector = in;
    }

    // Set vector to the input vector, in, by const ConstRef &
    void SetByConstConstRefAmp(const vctDynamicConstVectorRef<int> &in)
    {
        vector.SetSize(in.size());
        vector = in;
    }

	// Set vector to the input vector, in, by const reference
	void SetByConstRef(const vctDynamicVector<int> &in)
	{
		vector.SetSize(in.size());
		vector = in;
	}


	// Return vector by copy
	vctDynamicVector<int> GetByCopy() 
	{
		return vector;
	}

	// Return vector by reference
	vctDynamicVector<int> & GetByRef() 
	{
		return vector;
	}

	// Return vector by const reference
	const vctDynamicVector<int> & GetByConstRef()
	{
		return vector;
	}


    int getElement(unsigned int i) 
	{
		return vector[i];
	}

	void setElement(unsigned int i, int to) 
	{
		vector[i] = to;
	}

	int size() 
	{
		return vector.size();
	}

	void add(int value) 
	{
		vector.Add(value);
	}

	std::string ToString(void) 
	{
        std::stringstream outputStream;
        ToStream(outputStream);
        return outputStream.str();
    }

	void ToStream(std::ostream & outputStream)
    {
		vector.ToStream(outputStream);
    }

	inline int __getitem__(unsigned int index) const throw(std::out_of_range) 
	{
        return vector.at(index);
    }

	inline void __setitem__(unsigned int index, int value) throw(std::out_of_range) 
	{
        vector.at(index) = value;
    }

};
