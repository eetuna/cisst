#include <cisstVector/vctMatrixRotation2.h>
#include <cisstVector/vctMatrixRotation3.h>
#include <cisstVector/vctQuaternionRotation3.h>
#include <cisstVector/vctRandom.h>
#include <cisstCommon/cmnConstants.h>
#include <vector>

using std::vector;
using std::cout;
using std::cin;

typedef std::vector<int>::size_type size_type;

//: Create all the possible rotation matrices with exactly three nonzero elements.
// The possible nonzero values are {-1, 1}.  Each row and each column has exactly
// one nonzero elements.  The determinant of the matrix is always 1.
template<class _rotationMatrixType>
void GenerateAll3NonzeroRotationMatrices(vector<_rotationMatrixType> & rotations)
{
    typedef _rotationMatrixType::value_type valType;
    rotations.clear();
    const int n = _rotationMatrixType::DIMENSION;
    valType currentDet = 1;
    _rotationMatrixType rotMatrix;
    rotMatrix.SetAll(0);
    vctFixedSizeVector<int, _rotationMatrixType::DIMENSION> cols;
    enum {NUM_MATRICES = 3 * 2 * 2};
    rotations.reserve(NUM_MATRICES);
    for (cols[0] = 0; cols[0] < n; ++cols[0])
    {
        valType sign1 = 1;
        do {
            rotMatrix.Element(0, cols[0]) = sign1;
            for (cols[1] = 0; cols[1] < n; ++cols[1])
            {
                if (cols[1] == cols[0])
                    continue;
                if (n == 2)
                {
                    rotMatrix.Element(1, cols[1]) = currentDet;
                    rotations.push_back(rotMatrix);
                    rotMatrix.Element(1, cols[1]) = 0;
                    continue;
                }
                valType sign2 = 1;
                do {
                    rotMatrix.Element(1, cols[1]) = sign2;
                    for (cols[2] = 0; cols[2] < n; ++cols[2])
                    {
                        if ((cols[2] == cols[0]) || (cols[2] == cols[1]))
                            continue;
                        rotMatrix.Element(2, cols[2]) = currentDet;
                        rotations.push_back(rotMatrix);
                        rotMatrix.Element(2, cols[2]) = 0;
                    }
                    sign2 = -sign2;
                    currentDet = -currentDet;
                } while (sign2 != 1);
                rotMatrix.Element(1, cols[1]) = 0;
                currentDet = -currentDet;
            }
            sign1 = -sign1;
            currentDet = -currentDet;
        } while (sign1 != 1);
        currentDet = -currentDet;
        rotMatrix.Element(0, cols[0]) = 0;
    }
}

//: Create all rotation quaternions about the main axes for angles {-pi, -pi/2, pi/2, pi}
template<class _rotationQuatType>
void GenerateAllMajorAxisRotationQuaternion(vector<_rotationQuatType> & rotations)
{
    typedef _rotationQuatType::value_type valType;
    rotations.clear();
    enum {NUM_AXES = 3, NUM_ANGLES = 4};
    const valType angles[NUM_ANGLES] = {(valType)-cmnPI, (valType)-cmnPI_2, (valType)cmnPI_2, (valType)cmnPI};
    const valType halfAngles[NUM_ANGLES] = {(valType)-cmnPI_2, (valType)-cmnPI_4, (valType)cmnPI_4, (valType)cmnPI_2};
    valType cosHalfAngles[NUM_ANGLES];
    valType sinHalfAngles[NUM_ANGLES];
    int cAngle;
    for (cAngle = 0; cAngle < NUM_ANGLES; ++cAngle)
    {
        cosHalfAngles[cAngle] = (valType)cos(halfAngles[cAngle]);
        sinHalfAngles[cAngle] = (valType)sin(halfAngles[cAngle]);
    }

    int cAxis;
    _rotationQuatType rotQuaternion;
    rotQuaternion.SetAll(0);
    rotations.reserve(NUM_ANGLES * NUM_AXES);
    for (cAngle = 0; cAngle < NUM_ANGLES; ++cAngle)
    {
        rotQuaternion.R() = cosHalfAngles[cAngle];
        for (cAxis = 0; cAxis < NUM_AXES; ++cAxis)
        {
            rotQuaternion.Element(cAxis) = sinHalfAngles[cAngle];
            rotations.push_back(rotQuaternion);
            rotQuaternion.Element(cAxis) = 0;
        }
    }
}

//: Create all the rotation quaternions about the main axes and a set of random axes.
// The tiny angle is multiples of 1 through 17 times one third of the tolerance of 
// the quatenion element type.  This creates values: (1) under half the tolerance; 
// (2) between half and the tolerance; (3) the tolerance; (4) (5) half angle is less than
// the tolerance; (6) half angle equal to tolerance; (7) half angle more than tolerance.
// The tiny angles are added to and subtracted from multiples of pi/2 so that all near-zero
// sine and cosine values are covered.
template<class _rotationQuatType>
void GenerateTinyAngleRotationQuats(vector<_rotationQuatType> & rotations, int numRandom)
{
    typedef _rotationQuatType::value_type valType;
    typedef vctFixedSizeVector<valType, 3> axisType;
    rotations.clear();
    const valType tinyAngle = cmnTypeTraits<valType>::Tolerance() / 3;
    enum {NUM_AXES = 3, NUM_BASES = 5, NUM_STEPS = 17};
    const valType angleShift[NUM_BASES] = {(valType)-cmnPI, (valType)-cmnPI_2, 0, (valType)cmnPI_2, (valType)cmnPI};
    vector<valType> cosHalfAngles;
    vector<valType> sinHalfAngles;
    cosHalfAngles.reserve(NUM_BASES * NUM_STEPS * 2);
    sinHalfAngles.reserve(NUM_BASES * NUM_STEPS * 2);
    int cAngle;
    for (cAngle = 1; cAngle <= NUM_STEPS; ++cAngle)
    {
        valType multAngle = cAngle * tinyAngle;
        int cShift;
        for (cShift = 0; cShift < NUM_BASES; ++cShift)
        {
            valType shiftMultAngle;
            shiftMultAngle = (angleShift[cShift] + multAngle) / 2;
            cosHalfAngles.push_back((valType)cos(shiftMultAngle));
            sinHalfAngles.push_back((valType)sin(shiftMultAngle));
            shiftMultAngle = (angleShift[cShift] - multAngle) / 2;
            cosHalfAngles.push_back((valType)cos(shiftMultAngle));
            sinHalfAngles.push_back((valType)sin(shiftMultAngle));
        }
    }

    vector<axisType> axisList;
    axisList.reserve(2 * NUM_AXES + numRandom);
    axisType newAxis(0, 0, 0);
    int cAxis;
    for (cAxis = 0; cAxis < NUM_AXES; ++cAxis)
    {
        newAxis[cAxis] = 1;
        axisList.push_back(newAxis);
        newAxis[cAxis] = -1;
        axisList.push_back(newAxis);
        newAxis[cAxis] = 0;
    }
    for (cAxis = 0; cAxis < numRandom; ++cAxis)
    {
        vctRandom(newAxis, (valType)-1, (valType)1);
        newAxis.NormalizedSelf();
        axisList.push_back(newAxis);
    }

    rotations.reserve( axisList.size() * cosHalfAngles.size() );
    for (cAxis = 0; cAxis < (int)axisList.size(); ++cAxis)
    {
        for (cAngle = 0; cAngle < (int)cosHalfAngles.size(); ++cAngle)
        {
            _rotationQuatType newQuat;
            newQuat.R() = cosHalfAngles[cAngle];
            newQuat.XYZ().ProductOf( axisList[cAxis], sinHalfAngles[cAngle] );
            newQuat.NormalizedSelf();
            rotations.push_back(newQuat);
        }
    }

}

template<class _dstRotationType, class _srcRotationType>
void ConvertRotationList(vector<_dstRotationType> & dstRotations, const vector<_srcRotationType> & srcRotations)
{
    const size_type dstLen = dstRotations.size();
    assert(dstLen == srcRotations.size());
    size_type i;
    for (i = 0; i < dstLen; ++i)
        dstRotations[i].From(srcRotations[i]);
}

template<class _rotType1, class _rotType2>
void TestRotationEquivalence(const vector<_rotType1> & rotList1, const vector<_rotType2> & rotList2,
                             vector<size_type> & differences, vector<size_type> & exceptions)
{
    const size_type len = rotList1.size();
    assert(rotList2.size() == len);
    size_type i;
    _rotType1 rot1FromRot2;
    _rotType2 rot2FromRot1;

    differences.clear();
    exceptions.clear();
    typedef typename _rotType1::value_type value_type;
    value_type tolerance = cmnTypeTraits<value_type>::Tolerance();

    for (i = 0; i < len; ++i)
    {
        try {
            rot1FromRot2.From(rotList2[i]);
            rot2FromRot1.From(rotList1[i]);
            if ( (!rot1FromRot2.AlmostEquivalent(rotList1[i], tolerance)) || (!rot2FromRot1.AlmostEquivalent(rotList2[i], tolerance)) )
                differences.push_back(i);
        }
        catch(std::runtime_error e)
        {
            /* DEBUG 
            try {
                rot1FromRot2.From(rotList2[i]);
                rot2FromRot1.From(rotList1[i]);
            }
            catch(std::runtime_error e1)
            {}
            /* END DEBUG */

            exceptions.push_back(i);
        }
    }
}

template<class _rotType>
void ComputeAllProducts(const vector<_rotType> & leftOps, const vector<_rotType> & rightOps, vector<_rotType> & result)
{
    result.clear();
    result.resize( leftOps.size() * rightOps.size() );
    const vector<_rotType>::const_iterator leftEnd = leftOps.end();
    const vector<_rotType>::const_iterator rightEnd = rightOps.end();
    vector<_rotType>::const_iterator leftIt;
    vector<_rotType>::const_iterator rightIt;
    vector<_rotType>::iterator resltIt = result.begin();

    for (leftIt = leftOps.begin(); leftIt != leftEnd; ++leftIt)
    {
        for (rightIt = rightOps.begin(); rightIt != rightEnd; ++rightIt)
        {
            (*resltIt).ProductOf(*leftIt, *rightIt);
            ++resltIt;
        }
    }
}

void main()
{
    typedef float value_type;
    typedef vctMatrixRotation3<value_type> MatRotf;
    typedef vctQuaternionRotation3<value_type> QuatRotf;

    vector<size_type> differences, exceptions;

    vector<MatRotf> matRotList1;
    GenerateAll3NonzeroRotationMatrices(matRotList1);
    vector<QuatRotf> quatRotList1(matRotList1.size());
    ConvertRotationList(quatRotList1, matRotList1);
    cout << "Number of rotation matrices with 3 nonzero elements: " << matRotList1.size() << std::endl;
    TestRotationEquivalence(matRotList1, quatRotList1, differences, exceptions);
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;


    vector<QuatRotf> quatRotList2;
    GenerateAllMajorAxisRotationQuaternion(quatRotList2);
    vector<MatRotf> matRotList2(quatRotList2.size());
    ConvertRotationList(matRotList2, quatRotList2);
    cout << "Number of rotation quaternions about major axes: " << quatRotList2.size() << std::endl;
    TestRotationEquivalence(matRotList2, quatRotList2, differences, exceptions);
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    vector<QuatRotf> quatRotList3;
    GenerateTinyAngleRotationQuats(quatRotList3, 25);
    vector<MatRotf> matRotList3(quatRotList3.size());
    ConvertRotationList(matRotList3, quatRotList3);
    cout << "Number of rotation quatenions with tiny angle (major and random axes): " << quatRotList3.size() << std::endl;
    TestRotationEquivalence(matRotList3, quatRotList3, differences, exceptions);
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    vector<MatRotf> prodMatLists;
    vector<QuatRotf> prodQuatLists;

    ComputeAllProducts(matRotList1, matRotList2, prodMatLists);
    ComputeAllProducts(quatRotList1, quatRotList2, prodQuatLists);
    TestRotationEquivalence(prodMatLists, prodQuatLists, differences, exceptions);
    cout << "Product: list1*list2. Number of tests: " << prodMatLists.size() << "\n";
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    ComputeAllProducts(matRotList2, matRotList1, prodMatLists);
    ComputeAllProducts(quatRotList2, quatRotList1, prodQuatLists);
    TestRotationEquivalence(prodMatLists, prodQuatLists, differences, exceptions);
    cout << "Product: list2*list1. Number of tests: " << prodMatLists.size() << "\n";
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    ComputeAllProducts(matRotList1, matRotList3, prodMatLists);
    ComputeAllProducts(quatRotList1, quatRotList3, prodQuatLists);
    TestRotationEquivalence(prodMatLists, prodQuatLists, differences, exceptions);
    cout << "Product: list1*list3. Number of tests: " << prodMatLists.size() << "\n";
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    ComputeAllProducts(matRotList3, matRotList1, prodMatLists);
    ComputeAllProducts(quatRotList3, quatRotList1, prodQuatLists);
    TestRotationEquivalence(prodMatLists, prodQuatLists, differences, exceptions);
    cout << "Product: list3*list1. Number of tests: " << prodMatLists.size() << "\n";
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    ComputeAllProducts(matRotList2, matRotList3, prodMatLists);
    ComputeAllProducts(quatRotList2, quatRotList3, prodQuatLists);
    TestRotationEquivalence(prodMatLists, prodQuatLists, differences, exceptions);
    cout << "Product: list2*list3. Number of tests: " << prodMatLists.size() << "\n";
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;

    ComputeAllProducts(matRotList3, matRotList2, prodMatLists);
    ComputeAllProducts(quatRotList3, quatRotList2, prodQuatLists);
    TestRotationEquivalence(prodMatLists, prodQuatLists, differences, exceptions);
    cout << "Product: list3*list2. Number of tests: " << prodMatLists.size() << "\n";
    cout << "Number of unequal matrix-quaternion pairs: " << differences.size() << std::endl;
    cout << "Number of conversion exceptions: " << exceptions.size() << std::endl;
    cout << std::endl;
}
