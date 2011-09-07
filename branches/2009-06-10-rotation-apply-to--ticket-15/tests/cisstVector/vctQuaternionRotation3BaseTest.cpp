/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):  Anton Deguet
  Created on: 2007-02-05
  
  (C) Copyright 2007-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include "vctQuaternionRotation3BaseTest.h"
#include "vctGenericRotationTest.h"

#include <cisstCommon/cmnConstants.h>
#include <cisstVector/vctRandom.h>


template <class _elementType>
void vctQuaternionRotation3BaseTest::TestConstructors(void) {

    vctFixedSizeVector<_elementType, 3> x(1.0, 0.0, 0.0);
    vctFixedSizeVector<_elementType, 3> y(0.0, 1.0, 0.0);
    vctFixedSizeVector<_elementType, 3> z(0.0, 0.0, 1.0);
    vctFixedSizeVector<_elementType, 3> difference;

    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType testRotation1;

    _elementType tolerance = cmnTypeTraits<_elementType>::Tolerance();

    CPPUNIT_ASSERT(testRotation1.size() == 4);
    CPPUNIT_ASSERT(testRotation1.X() == 0.0);
    CPPUNIT_ASSERT(testRotation1.Y() == 0.0);
    CPPUNIT_ASSERT(testRotation1.Z() == 0.0);
    CPPUNIT_ASSERT(testRotation1.R() == 1.0);

    CPPUNIT_ASSERT(testRotation1 == QuatRotType::Identity());
    testRotation1.X() = 1.0;
    CPPUNIT_ASSERT(testRotation1 != QuatRotType::Identity());
    testRotation1.Assign(0.0, 0.0, 0.0, 1.0);
    CPPUNIT_ASSERT(testRotation1 == QuatRotType::Identity());

    QuatRotType testRotation2(vctAxisAngleRotation3<_elementType>(x, _elementType(cmnPI_2)));
    difference = testRotation2 * y - z;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);
    difference = -(testRotation2 * z) - y;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);

    testRotation2.From(vctAxisAngleRotation3<_elementType>(y, _elementType(cmnPI_2)));
    difference = testRotation2 * z - x;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);
    difference = -(testRotation2 * x) - z;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);

    testRotation2.From(vctAxisAngleRotation3<_elementType>(z, _elementType(cmnPI_2)));
    difference = testRotation2 * x - y;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);
    difference = -(testRotation2 * y) - x;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);

    QuatRotType testRotation3;
    difference = testRotation3 * x - x;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);
    difference = testRotation3 * y - y;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);
    difference = testRotation3 * y - y;
    CPPUNIT_ASSERT(difference.LinfNorm() < tolerance);
}


void vctQuaternionRotation3BaseTest::TestConstructorsDouble(void) {
    TestConstructors<double>();
}

void vctQuaternionRotation3BaseTest::TestConstructorsFloat(void) {
    TestConstructors<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestApplyTo(void) {

    _elementType tolerance = cmnTypeTraits<_elementType>::Tolerance();

    vctFixedSizeVector<_elementType, 3> x(1.0, 0.0, 0.0);
    vctFixedSizeVector<_elementType, 3> y(0.0, 1.0, 0.0);
    vctFixedSizeVector<_elementType, 3> z(0.0, 0.0, 1.0);
    vctFixedSizeVector<_elementType, 3> result;
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType composed;

    QuatRotType testRotation(vctAxisAngleRotation3<_elementType>(x, _elementType(cmnPI_2)));
    testRotation.ApplyTo(y, result);
    CPPUNIT_ASSERT((z - result).LinfNorm() < tolerance);
    testRotation.ApplyTo(z, result);
    CPPUNIT_ASSERT((y + result).LinfNorm() < tolerance);
    testRotation.ApplyTo(QuatRotType::Identity(), composed);
    composed.ApplyTo(y, result);
    CPPUNIT_ASSERT((z - result).LinfNorm() < tolerance);
    composed.ApplyTo(z, result);
    CPPUNIT_ASSERT((y + result).LinfNorm() < tolerance);
    
    testRotation.From(vctAxisAngleRotation3<_elementType>(y, _elementType(cmnPI_2)));
    testRotation.ApplyTo(z, result);
    CPPUNIT_ASSERT((x - result).LinfNorm() < tolerance);
    testRotation.ApplyTo(x, result);
    CPPUNIT_ASSERT((z + result).LinfNorm() < tolerance);
    testRotation.ApplyTo(QuatRotType::Identity(), composed);
    composed.ApplyTo(z, result);
    CPPUNIT_ASSERT((x - result).LinfNorm() < tolerance);
    composed.ApplyTo(x, result);
    CPPUNIT_ASSERT((z + result).LinfNorm() < tolerance);

    testRotation.From(vctAxisAngleRotation3<_elementType>(z, _elementType(cmnPI_2)));
    testRotation.ApplyTo(x, result);
    CPPUNIT_ASSERT((y - result).LinfNorm() < tolerance);
    testRotation.ApplyTo(y, result);
    CPPUNIT_ASSERT((x + result).LinfNorm() < tolerance);
    testRotation.ApplyTo(QuatRotType::Identity(), composed);
    composed.ApplyTo(x, result);
    CPPUNIT_ASSERT((y - result).LinfNorm() < tolerance);
    composed.ApplyTo(y, result);
    CPPUNIT_ASSERT((x + result).LinfNorm() < tolerance);
}


void vctQuaternionRotation3BaseTest::TestApplyToDouble(void) {
    TestApplyTo<double>();
}

void vctQuaternionRotation3BaseTest::TestApplyToFloat(void) {
    TestApplyTo<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestConversionMatrix(void) {
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType quaternionRotation;
    QuatRotType rotationQuaternion;
    vctRandom(quaternionRotation);
    vctGenericRotationTest::TestConversion(quaternionRotation, rotationQuaternion);
}

void vctQuaternionRotation3BaseTest::TestConversionMatrixDouble(void) {
    TestConversionMatrix<double>();
}

void vctQuaternionRotation3BaseTest::TestConversionMatrixFloat(void) {
    TestConversionMatrix<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestConversionAxisAngle(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > quaternionRotation;
    vctAxisAngleRotation3<_elementType> axisAngleRotation;
    vctRandom(quaternionRotation);
    vctGenericRotationTest::TestConversion(quaternionRotation, axisAngleRotation);
}

void vctQuaternionRotation3BaseTest::TestConversionAxisAngleDouble(void) {
    TestConversionAxisAngle<double>();
}

void vctQuaternionRotation3BaseTest::TestConversionAxisAngleFloat(void) {
    TestConversionAxisAngle<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestConversionRodriguez(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > quaternionRotation;
    vctRodriguezRotation3<_elementType> rodriguezRotation;
    vctRandom(quaternionRotation);
    vctGenericRotationTest::TestConversion(quaternionRotation, rodriguezRotation);
}

void vctQuaternionRotation3BaseTest::TestConversionRodriguezDouble(void) {
    TestConversionRodriguez<double>();
}

void vctQuaternionRotation3BaseTest::TestConversionRodriguezFloat(void) {
    TestConversionRodriguez<float>();
}




template <class _elementType>
void vctQuaternionRotation3BaseTest::TestFromSignaturesMatrix(void) {
    typedef vctMatrixRotation3Base<vctFixedSizeMatrix<_elementType, 3, 3> > MatRotType;
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType toRotation;
    MatRotType fromRotationNormalized, fromRotationNotNormalized;
    vctRandom(fromRotationNormalized);
    vctRandom(fromRotationNotNormalized);
    fromRotationNotNormalized.Add(_elementType(1.0));
    vctGenericRotationTest::TestFromSignatures(toRotation,
                                               fromRotationNormalized,
                                               fromRotationNotNormalized);
}

void vctQuaternionRotation3BaseTest::TestFromSignaturesMatrixDouble(void) {
    TestFromSignaturesMatrix<double>();
}

void vctQuaternionRotation3BaseTest::TestFromSignaturesMatrixFloat(void) {
    TestFromSignaturesMatrix<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestFromSignaturesAxisAngle(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > toRotation;
    vctAxisAngleRotation3<_elementType> fromRotationNormalized, fromRotationNotNormalized;
    vctRandom(fromRotationNormalized);
    vctRandom(fromRotationNotNormalized);
    fromRotationNotNormalized.Axis().Add(_elementType(1.0));
    vctGenericRotationTest::TestFromSignatures(toRotation,
                                               fromRotationNormalized,
                                               fromRotationNotNormalized);
}

void vctQuaternionRotation3BaseTest::TestFromSignaturesAxisAngleDouble(void) {
    TestFromSignaturesAxisAngle<double>();
}

void vctQuaternionRotation3BaseTest::TestFromSignaturesAxisAngleFloat(void) {
    TestFromSignaturesAxisAngle<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestFromSignaturesRodriguez(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > toRotation;
    vctRodriguezRotation3<_elementType> fromRotationNormalized, fromRotationNotNormalized;
    vctRandom(fromRotationNormalized);
    vctRandom(fromRotationNotNormalized);
    fromRotationNotNormalized.Add(_elementType(20.0));
    vctGenericRotationTest::TestFromSignatures(toRotation,
                                               fromRotationNormalized,
                                               fromRotationNotNormalized,
                                               true);
}

void vctQuaternionRotation3BaseTest::TestFromSignaturesRodriguezDouble(void) {
    TestFromSignaturesRodriguez<double>();
}

void vctQuaternionRotation3BaseTest::TestFromSignaturesRodriguezFloat(void) {
    TestFromSignaturesRodriguez<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestIdentity(void) {
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType quaternionRotation;
    vctFixedSizeVector<_elementType, QuatRotType::DIMENSION> inputVector, outputVector;
    vctRandom(quaternionRotation);
    vctRandom(inputVector, _elementType(-1.0), _elementType(1.0));
    vctGenericRotationTest::TestIdentity(quaternionRotation, inputVector, outputVector);
}

void vctQuaternionRotation3BaseTest::TestIdentityDouble(void) {
    TestIdentity<double>();
}

void vctQuaternionRotation3BaseTest::TestIdentityFloat(void) {
    TestIdentity<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestDefaultConstructor(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > quaternionRotation;
    vctGenericRotationTest::TestDefaultConstructor(quaternionRotation);
}

void vctQuaternionRotation3BaseTest::TestDefaultConstructorDouble(void) {
    TestDefaultConstructor<double>();
}

void vctQuaternionRotation3BaseTest::TestDefaultConstructorFloat(void) {
    TestDefaultConstructor<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestInverse(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > quaternionRotation;
    vctRandom(quaternionRotation);
    vctGenericRotationTest::TestInverse(quaternionRotation);
    
}

void vctQuaternionRotation3BaseTest::TestInverseDouble(void) {
    TestInverse<double>();
}

void vctQuaternionRotation3BaseTest::TestInverseFloat(void) {
    TestInverse<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestComposition(void) {
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType quaternionRotation1;
    QuatRotType quaternionRotation2;
    vctFixedSizeVector<_elementType, QuatRotType::DIMENSION> inputVector;
    vctRandom(quaternionRotation1);
    vctRandom(quaternionRotation2);
    vctRandom(inputVector, _elementType(-1.0), _elementType(1.0));
    vctGenericRotationTest::TestComposition(quaternionRotation1, quaternionRotation2, inputVector);
}

void vctQuaternionRotation3BaseTest::TestCompositionDouble(void) {
    TestComposition<double>();
}

void vctQuaternionRotation3BaseTest::TestCompositionFloat(void) {
    TestComposition<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestRandom(void) {
    vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > quaternionRotation;
    vctRandom(quaternionRotation);
    vctGenericRotationTest::TestRandom(quaternionRotation);
}

void vctQuaternionRotation3BaseTest::TestRandomDouble(void) {
    TestRandom<double>();
}

void vctQuaternionRotation3BaseTest::TestRandomFloat(void) {
    TestRandom<float>();
}



template <class _elementType>
void vctQuaternionRotation3BaseTest::TestRigidity(void) {
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<_elementType, 4> > QuatRotType;
    QuatRotType quaternionRotation;
    vctFixedSizeVector<_elementType, QuatRotType::DIMENSION> vector1;
    vctFixedSizeVector<_elementType, QuatRotType::DIMENSION> vector2;
    vctRandom(quaternionRotation);
    vctRandom(vector1, _elementType(-1.0), _elementType(1.0));
    vctRandom(vector2, _elementType(-1.0), _elementType(1.0));
    vctGenericRotationTest::TestRigidity(quaternionRotation, vector1, vector2);
}

void vctQuaternionRotation3BaseTest::TestRigidityDouble(void) {
    TestRigidity<double>();
}

void vctQuaternionRotation3BaseTest::TestRigidityFloat(void) {
    TestRigidity<float>();
}


template <class _elementType>
void vctQuaternionRotation3BaseTest::TestApplyMethodsOperators(void) {
    typedef _elementType value_type;
    typedef vctQuaternionRotation3Base<vctFixedSizeVector<value_type, 4> > QuatRotType;
    typedef vctFixedSizeVector<value_type, 3> VectorType;
    enum {NUM_MATRIX_ROWS = 3, NUM_MATRIX_COLS = 6};
    typedef vctFixedSizeMatrix<value_type, NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_ROW_MAJOR> FixedSizeMatrixRowMajor;
    typedef vctFixedSizeMatrix<value_type, NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_COL_MAJOR> FixedSizeMatrixColMajor;
    typedef vctFixedSizeMatrix<value_type, NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_ROW_MAJOR> FixedSizeMatrixTpsRowMajor;
    typedef vctFixedSizeMatrix<value_type, NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_COL_MAJOR> FixedSizeMatrixTpsColMajor;
    typedef vctDynamicVector<value_type> DynamicVectorType;
    typedef vctDynamicMatrix<value_type> DynamicMatrixType;

    QuatRotType quaternionRotation;
    vctRandom(quaternionRotation);

    VectorType vector;
    vctRandom(vector, _elementType(-1.0), _elementType(1.0));
    vctGenericRotationTest::TestApplyMethodsOperatorsObject(quaternionRotation, vector);

    DynamicVectorType dataDynVector(NUM_MATRIX_ROWS), resultDynVec1(NUM_MATRIX_ROWS), resultDynVec2(NUM_MATRIX_ROWS);
    vctRandom(dataDynVector, static_cast<value_type>(-1.0), static_cast<value_type>(1.0));
    vctGenericRotationTest::TestApplyMethodsOperatorsObject(quaternionRotation, dataDynVector);

    QuatRotType rotation;
    vctRandom(rotation);
    vctGenericRotationTest::TestApplyMethodsOperatorsXform(quaternionRotation, rotation);

    vctDynamicVectorRef<value_type> dataVecRef(dataDynVector), resultVec1Ref(resultDynVec1), resultVec2Ref(resultDynVec2);
    quaternionRotation.ApplyTo(dataVecRef, resultVec1Ref);
    quaternionRotation.ApplyInverseTo(resultVec1Ref, resultVec2Ref);
    CPPUNIT_ASSERT( resultDynVec2.AlmostEqual(dataDynVector) );

    FixedSizeMatrixRowMajor dataMatrixRowMajor;
    vctRandom(dataMatrixRowMajor, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    FixedSizeMatrixRowMajor resultMatrixRowMajor1, resultMatrixRowMajor2;
    quaternionRotation.ApplyTo(dataMatrixRowMajor, resultMatrixRowMajor1);
    quaternionRotation.ApplyInverseTo(resultMatrixRowMajor1, resultMatrixRowMajor2);
    CPPUNIT_ASSERT( resultMatrixRowMajor2.AlmostEqual(dataMatrixRowMajor) );

    // Test the Apply methods on a single column of a fixed-size matrix
    vctRandom(dataMatrixRowMajor.Column(0), static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    quaternionRotation.ApplyTo(dataMatrixRowMajor.Column(0), resultMatrixRowMajor1.Column(0));
    quaternionRotation.ApplyInverseTo(resultMatrixRowMajor1.Column(0), resultMatrixRowMajor2.Column(0));
    CPPUNIT_ASSERT( resultMatrixRowMajor2.Column(0).AlmostEqual(dataMatrixRowMajor.Column(0)) );

    FixedSizeMatrixColMajor dataMatrixColMajor;
    vctRandom(dataMatrixColMajor, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    FixedSizeMatrixColMajor resultMatrixColMajor1, resultMatrixColMajor2;
    quaternionRotation.ApplyTo(dataMatrixColMajor, resultMatrixColMajor1);
    quaternionRotation.ApplyInverseTo(resultMatrixColMajor1, resultMatrixColMajor2);
    CPPUNIT_ASSERT( resultMatrixColMajor2.AlmostEqual(dataMatrixColMajor) );

    FixedSizeMatrixTpsRowMajor dataMatrixTpsRowMajor;
    vctRandom(dataMatrixTpsRowMajor, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    FixedSizeMatrixTpsRowMajor resultMatrixTpsRowMajor1, resultMatrixTpsRowMajor2;
    quaternionRotation.ApplyTo(dataMatrixTpsRowMajor.TransposeRef(), resultMatrixTpsRowMajor1.TransposeRef());
    quaternionRotation.ApplyInverseTo(resultMatrixTpsRowMajor1.TransposeRef(), resultMatrixTpsRowMajor2.TransposeRef());
    CPPUNIT_ASSERT( resultMatrixTpsRowMajor2.AlmostEqual(dataMatrixTpsRowMajor) );

    FixedSizeMatrixTpsColMajor dataMatrixTpsColMajor;
    vctRandom(dataMatrixTpsColMajor, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    FixedSizeMatrixTpsColMajor resultMatrixTpsColMajor1, resultMatrixTpsColMajor2;
    quaternionRotation.ApplyTo(dataMatrixTpsColMajor.TransposeRef(), resultMatrixTpsColMajor1.TransposeRef());
    quaternionRotation.ApplyInverseTo(resultMatrixTpsColMajor1.TransposeRef(), resultMatrixTpsColMajor2.TransposeRef());
    CPPUNIT_ASSERT( resultMatrixTpsColMajor2.AlmostEqual(dataMatrixTpsColMajor) );

    DynamicMatrixType dataMatrixDynamic, resultMatrixDynamic1, resultMatrixDynamic2;
    dataMatrixDynamic.SetSize(NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_ROW_MAJOR);
    resultMatrixDynamic1.SetSize(NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_ROW_MAJOR);
    resultMatrixDynamic2.SetSize(NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_ROW_MAJOR);
    vctRandom(dataMatrixDynamic, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    quaternionRotation.ApplyTo(dataMatrixDynamic, resultMatrixDynamic1);
    quaternionRotation.ApplyInverseTo(resultMatrixDynamic1, resultMatrixDynamic2);
    CPPUNIT_ASSERT( resultMatrixDynamic2.AlmostEqual(dataMatrixDynamic) );

    // Test the Apply methods on a single column of a dynamic matrix
    vctRandom(dataMatrixDynamic.Column(0), static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    quaternionRotation.ApplyTo(dataMatrixDynamic.Column(0), resultMatrixDynamic1.Column(0));
    quaternionRotation.ApplyInverseTo(resultMatrixDynamic1.Column(0), resultMatrixDynamic2.Column(0));
    CPPUNIT_ASSERT( resultMatrixDynamic2.Column(0).AlmostEqual(dataMatrixDynamic.Column(0)) );

    dataMatrixDynamic.SetSize(NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_COL_MAJOR);
    resultMatrixDynamic1.SetSize(NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_COL_MAJOR);
    resultMatrixDynamic2.SetSize(NUM_MATRIX_ROWS, NUM_MATRIX_COLS, VCT_COL_MAJOR);
    vctRandom(dataMatrixDynamic, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    quaternionRotation.ApplyTo(dataMatrixDynamic, resultMatrixDynamic1);
    quaternionRotation.ApplyInverseTo(resultMatrixDynamic1, resultMatrixDynamic2);
    CPPUNIT_ASSERT( resultMatrixDynamic2.AlmostEqual(dataMatrixDynamic) );

    dataMatrixDynamic.SetSize(NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_ROW_MAJOR);
    resultMatrixDynamic1.SetSize(NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_ROW_MAJOR);
    resultMatrixDynamic2.SetSize(NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_ROW_MAJOR);
    // Note that here we also test vctRandom for a reference matrix
    vctRandom(dataMatrixDynamic.TransposeRef(), static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    quaternionRotation.ApplyTo(dataMatrixDynamic.TransposeRef(), resultMatrixDynamic1.TransposeRef());
    quaternionRotation.ApplyInverseTo(resultMatrixDynamic1.TransposeRef(), resultMatrixDynamic2.TransposeRef());
    CPPUNIT_ASSERT( resultMatrixDynamic2.AlmostEqual(dataMatrixDynamic) );

    dataMatrixDynamic.SetSize(NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_COL_MAJOR);
    resultMatrixDynamic1.SetSize(NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_COL_MAJOR);
    // Note that resultMatrixDynamic2 was deliberately left as VCT_ROW_MAJOR to test mixed-order
    // arguments
    resultMatrixDynamic2.SetSize(NUM_MATRIX_COLS, NUM_MATRIX_ROWS, VCT_ROW_MAJOR);
    vctRandom(dataMatrixDynamic, static_cast<value_type>(-2.0), static_cast<value_type>(2.0));
    quaternionRotation.ApplyTo(dataMatrixDynamic.TransposeRef(), resultMatrixDynamic1.TransposeRef());
    quaternionRotation.ApplyInverseTo(resultMatrixDynamic1.TransposeRef(), resultMatrixDynamic2.TransposeRef());
    CPPUNIT_ASSERT( resultMatrixDynamic2.AlmostEqual(dataMatrixDynamic) );


}


void vctQuaternionRotation3BaseTest::TestApplyMethodsOperatorsDouble(void) {
    TestApplyMethodsOperators<double>();
}

void vctQuaternionRotation3BaseTest::TestApplyMethodsOperatorsFloat(void) {
    TestApplyMethodsOperators<float>();
}



CPPUNIT_TEST_SUITE_REGISTRATION(vctQuaternionRotation3BaseTest);

