#include "testlayers.h"
#include <ops/declarable/helpers/householder.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <ops/declarable/helpers/hhSequence.h>
#include <ops/declarable/helpers/svd.h>
#include <ops/declarable/helpers/hhColPivQR.h>


using namespace nd4j;

class HelpersTests1 : public testing::Test {
public:
    
    HelpersTests1() {
        
        std::cout<<std::endl<<std::flush;
    }

};


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test1) {
            
    NDArray<double> x('c', {1,4}, {14,17,3,1});                
    NDArray<double> exp('c', {4,4}, {-0.629253, -0.764093,   -0.13484, -0.0449467, -0.764093,  0.641653, -0.0632377, -0.0210792, -0.13484,-0.0632377,    0.98884,-0.00371987, -0.0449467,-0.0210792,-0.00371987,    0.99876});
    
    NDArray<double> result = ops::helpers::Householder<double>::evalHHmatrix(x);    

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test2) {
            
    NDArray<double> x('c', {1,3}, {14,-4,3});                
    NDArray<double> exp('c', {3,3}, {-0.941742, 0.269069,-0.201802, 0.269069, 0.962715,0.0279639, -0.201802,0.0279639, 0.979027});
    
    NDArray<double> result = ops::helpers::Householder<double>::evalHHmatrix(x);    

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrixData_test1) {
            
    NDArray<double> x('c', {1,4}, {14,17,3,1});            
    NDArray<double> tail('c', {1,3});        
    NDArray<double> expTail('c', {1,3}, {0.468984, 0.0827618, 0.0275873});
    const double normXExpected = -22.2486;
    const double coeffExpected = 1.62925;

    double normX, coeff;
    ops::helpers::Householder<double>::evalHHmatrixData(x, tail, coeff, normX);    

    ASSERT_NEAR(normX, normXExpected, 1e-5);
    ASSERT_NEAR(coeff, coeffExpected, 1e-5);
    ASSERT_TRUE(tail.isSameShapeStrict(&expTail));
    ASSERT_TRUE(tail.equalsTo(&expTail));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulLeft_test1) {
            
    NDArray<double> x('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});            
    NDArray<double> tail('c', {1,3}, {0.5,0.5,0.5});            
    NDArray<double> exp('c', {4,4}, {9.05,15.8,11.4, 0.8, 8.525, 2.4,15.7,17.9, 17.525,16.4, 3.7, 1.9, 4.525, 2.4, 0.7,14.9});
    
    ops::helpers::Householder<double>::mulLeft(x, tail, 0.1);
    // expTail.printShapeInfo();

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}

/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulLeft_test2) {
            
    NDArray<double> x('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});            
    NDArray<double> tail('c', {3,1}, {0.5,0.5,0.5});            
    NDArray<double> exp('c', {4,4}, {9.05,15.8,11.4, 0.8, 8.525, 2.4,15.7,17.9, 17.525,16.4, 3.7, 1.9, 4.525, 2.4, 0.7,14.9});
    
    ops::helpers::Householder<double>::mulLeft(x, tail, 0.1);    

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}

/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulRight_test1) {
            
    NDArray<double> x('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});            
    NDArray<double> tail('c', {1,3}, {0.5,0.5,0.5});            
    NDArray<double> exp('c', {4,4}, {9,17.5,12.5,  1.5, 7, 2.5,15.5, 17.5, 15.8,16.4, 3.4,  1.4, 4.3,3.15,1.15,15.15});
    
    ops::helpers::Householder<double>::mulRight(x, tail, 0.1);    

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test1) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6,13,11,7,6,3,7,4,7,6,6,7,10});      
    NDArray<double> hhMatrixExp('c', {4,4}, {1.524000,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,0, 0.229221,-0.272237,0.938237,0});
    NDArray<double> hhBidiagExp('c', {4,4}, {-17.1756, 24.3869,       0,      0, 0,-8.61985,-3.89823,      0, 0,       0, 4.03047,4.13018, 0,       0,       0,1.21666});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();
    object._HHbidiag.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}
    
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test2) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});      
    NDArray<double> hhMatrixExp('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392,        0, -0.22821, 0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});
    NDArray<double> hhBidiagExp('c', {4,4}, {-17.2916,7.03123,       0,       0, 0, 16.145,-22.9275,       0, 0,      0, -9.9264,-11.5516, 0,      0,       0,-12.8554});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();
    object._HHbidiag.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}
    
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test3) {
            
    NDArray<double> matrix('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12, 0,-15,10,2});      
    NDArray<double> hhMatrixExp('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});
    NDArray<double> hhBidiagExp('c', {4,4}, {-17.2916,7.03123,       0,      0, 0,16.3413,-20.7828,      0, 0,      0,-18.4892,4.13261, 0,      0,       0,-21.323});
    
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    object._HHmatrix.printBuffer();
    object._HHbidiag.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test1) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> vectorsUseqExp('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392, 0, -0.22821,0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});
    NDArray<double> vectorsVseqExp('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392, 0, -0.22821,0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});      
    NDArray<double> coeffsUseqExp('c', {4,1}, {1.52048,1.66025,1.58392,1.99303});      
    NDArray<double> coeffsVseqExp('c', {3,1}, {1.37012,1.66979,0});      
        
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');

    ASSERT_TRUE(uSeq._vectors.isSameShapeStrict(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.isSameShapeStrict(&vectorsVseqExp));
    ASSERT_TRUE(uSeq._vectors.equalsTo(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.equalsTo(&vectorsVseqExp));

    ASSERT_TRUE(vSeq._length == uSeq._length - 1);    
    ASSERT_TRUE(vSeq._shift == 1);    
    ASSERT_TRUE(uSeq._shift == 0);    
        
}
  
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test2) {
            
    NDArray<double> matrix('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12 ,0,-15,10,2});
    NDArray<double> vectorsUseqExp('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});
    NDArray<double> vectorsVseqExp('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});      
    NDArray<double> coeffsUseqExp('c', {4,1}, {1.52048,1.65232,1.35075,1.61136});      
    NDArray<double> coeffsVseqExp('c', {3,1}, {1.37012,1.59666,0});      
        
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');

    ASSERT_TRUE(uSeq._vectors.isSameShapeStrict(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.isSameShapeStrict(&vectorsVseqExp));
    ASSERT_TRUE(uSeq._vectors.equalsTo(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.equalsTo(&vectorsVseqExp));

    ASSERT_TRUE(vSeq._length == uSeq._length - 1);    
    ASSERT_TRUE(vSeq._shift == 1);    
    ASSERT_TRUE(uSeq._shift == 0);    
        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test3) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    NDArray<double> vectorsUseqExp('c', {4,4}, {1.524,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,       0, 0.229221,-0.272237,0.938237, 0});
    NDArray<double> vectorsVseqExp('c', {4,4}, {1.524,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,       0, 0.229221,-0.272237,0.938237, 0});      
    NDArray<double> coeffsUseqExp('c', {4,1}, { 1.524, 1.5655,1.06367,0});      
    NDArray<double> coeffsVseqExp('c', {3,1}, {1.75682,1.02929, 0});
        
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');

    ASSERT_TRUE(uSeq._vectors.isSameShapeStrict(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.isSameShapeStrict(&vectorsVseqExp));
    ASSERT_TRUE(uSeq._vectors.equalsTo(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.equalsTo(&vectorsVseqExp));

    ASSERT_TRUE(vSeq._length == uSeq._length - 1);    
    ASSERT_TRUE(vSeq._shift == 1);    
    ASSERT_TRUE(uSeq._shift == 0);    
        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test4) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    NDArray<double> exp   ('c', {4,4}, {2.49369, 2.62176, 5.88386, 7.69905, -16.0588,-18.7319,-9.15007,-12.6164, 4.7247, 3.46252, 1.02038, -1.4533, 2.9279,-2.29178, 1.90139,-0.66187});
        
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix);
    
    ASSERT_TRUE(matrix.equalsTo(&exp));
        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test5) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> exp   ('c', {5,4}, {4.52891, 8.09473,-2.73704,-13.0302, -11.0752, 7.41549,-3.75125,0.815252, -7.76818,-15.9102,-9.90869,-11.8677, 1.63942,-17.0312,-9.05102,-4.49088, -9.63311,0.540226,-1.52764, 5.79111});
            
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix);
    
    ASSERT_TRUE(matrix.equalsTo(&exp));
        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test6) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,-1,3,9, -4.43019,-15.1713, -3.2854,-7.65743, -9.39162,-7.03599, 8.03827, 9.48453, -2.97785, -16.424, 5.35265,-20.1171, -0.0436177, -13.118,-8.37287,-17.3012, -1.14074, 4.18282,-10.0914,-5.69014});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test7) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    NDArray<double> exp   ('c', {4,4}, {9,13,3,6,-5.90424,-2.30926,-0.447417, 3.05712, -10.504,-9.31339, -8.85493,-10.8886, -8.29494,-10.6737, -5.94895,-7.55591});
        
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix);    
    
    ASSERT_TRUE(matrix.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test8) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> exp   ('c', {5,4}, {9,     -13,        3,       6, 13,      11,        7,      -6, -6.90831,-5.01113, 0.381677,0.440128, -0.80107,0.961605,-0.308019,-1.96153, -0.795985, 18.6538,  12.0731, 16.9988});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix);    

    ASSERT_TRUE(matrix.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test9) {
            
    NDArray<double> matrix('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12 ,0,-15,10,2});
    NDArray<double> exp   ('c', {6,4}, {9,     -13,        3,       6, 13,      11,        7,      -6, 3,       7,        4,       7, 3.77597, 18.6226,-0.674868, 4.61365, 5.02738,-14.1486, -2.22877,-8.98245, -0.683766, 1.73722,  14.9859, 12.0843});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix);    

    ASSERT_TRUE(matrix.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test10) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,      -1,       3,        9, 10,      11,      -7,       -5, 3,       2,       4,        7, 2.58863, 11.0295,-4.17483,-0.641012, -1.21892,-16.3151, 6.12049, -20.0239, -0.901799,-15.0389,-12.4944, -20.2394});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test11) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,      -1,       3,       9, 10,      11,      -7,      -5, 3,       2,       4,       7, 1.14934, 4.40257, 8.70127,-1.18824, 1.5132,0.220419,-11.6285,-11.7549, 2.32148, 24.3838,0.256531, 25.9116});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test12) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,      -1,       3,       9, 10,      11,      -7,      -5, 3,       2,       4,       7, -1,       6,       7,      19, -2.62252,-22.2914, 4.76743,-19.6689, -1.05943,-9.00514,-11.8013,-7.94571});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test13) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9 ,     -1 ,      3 ,      9, -4.65167, 3.44652, 7.83593, 22.6899, -9.48514, -21.902, 5.66559,-13.0533, -0.343184, 15.2895,  7.2888, 14.0489, 0.289638,-1.87752,   3.944,-1.49707, -2.48845, 3.18285,-10.6685,0.406502});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test14) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    NDArray<double> exp   ('c', {5,5}, {1.78958,  8.06962,-6.13687, 4.36267, 1.06472, -14.9578,  -8.1522, 1.30442,-18.3343,-13.2578, 13.5536,  5.50764, 15.7859, 7.60831, 11.7871, -1.3626,-0.634986, 7.60934, -2.1841, 5.62694, -13.0577,  15.1554, -7.6511, 3.76365,-5.87368});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test15) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    NDArray<double> exp   ('c', {5,5}, {9,      -1,       3,       9,      10, 11,      -7,      -5,       3,       2, 4,       7,      -1,       6,       7, -9.26566,-16.4298, 1.64125,-17.3243,-7.70257, -16.7077, 4.80216,-19.1652,-2.42279,-13.0258});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test1) {
            
    NDArray<double> matrix ('c', {5,5}, {-17 ,14 ,9 ,-12 ,-12 ,5 ,-4 ,-19 ,-7 ,-12 ,15 ,16 ,17 ,-6 ,8 ,-10 ,14 ,-15 ,6 ,-10 ,-14 ,12 ,-1 ,-16 ,3});
    NDArray<double> matrix2('c', {5,5}, {18 ,3 ,2 ,7 ,-11 ,7 ,7 ,10 ,-13 ,-8 ,13 ,20 ,-4 ,-16 ,-9 ,-17 ,-5 ,-7 ,-19 ,-8 ,-9 ,9 ,6 ,14 ,-11});
    NDArray<double> expM   ('c', {5,5}, {-17,14,9,-12,-12, 5,-4,    -19, -7,-12, 15,16,17.0294, -6,  8, -10,14,    -15,  6,-10, -14,12,      0,-16,  0});
    NDArray<double> expU   ('c', {5,5}, {18,3, 2,7,-11, 7, 7.75131,10,-12.5665, -8, 13,  20.905,-4,-14.7979, -9, -17,-3.87565,-7,-19.2608, -8, -9,       9, 6,      14,-11});

    ops::helpers::SVD<double> svd(matrix, true, true, true);    
    svd._M = matrix;
    svd._U = matrix2;
    svd.deflation1(1,1,2,2);    

    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test2) {
            
    NDArray<double> matrix ('c', {5,5}, {-17 ,14 ,9 ,-12 ,-12 ,5 ,-4 ,-19 ,-7 ,-12 ,15 ,16 ,17 ,-6 ,8 ,-10 ,14 ,-15 ,6 ,-10 ,-14 ,12 ,-1 ,-16 ,3});
    NDArray<double> matrix2('c', {5,5}, {18 ,3 ,2 ,7 ,-11 ,7 ,7 ,10 ,-13 ,-8 ,13 ,20 ,-4 ,-16 ,-9 ,-17 ,-5 ,-7 ,-19 ,-8 ,-9 ,9 ,6 ,14 ,-11});
    NDArray<double> expM   ('c', {5,5}, {22.6716,14,  9,-12,-12, 5,-4,-19, -7,-12, 0,16,  0, -6,  8, -10,14,-15,  6,-10, -14,12, -1,-16,  3});
    NDArray<double> expU   ('c', {5,5}, {-12.1738, 3, -13.4089,  7,-11, 1.36735, 7, -12.1297,-13, -8, -12.3944,20, -5.60173,-16, -9, -17,-5,-7,-19, -8, -9, 9, 6, 14,-11});

    ops::helpers::SVD<double> svd(matrix, true, true, true);    
    svd._M = matrix;
    svd._U = matrix2;
    svd.deflation1(0,0,2,2);    
        
    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test3) {
            
    NDArray<double> matrix ('c', {5,5}, {-17 ,14 ,9 ,-12 ,-12 ,5 ,-4 ,-19 ,-7 ,-12 ,15 ,16 ,17 ,-6 ,8 ,-10 ,14 ,-15 ,6 ,-10 ,-14 ,12 ,-1 ,-16 ,3});
    NDArray<double> matrix2('c', {2,6}, {18 ,3 ,2 ,7 ,-11 ,7 ,7 ,10 ,-13 ,-8 ,13 ,20});
    NDArray<double> expM   ('c', {5,5}, {-17,14,9,-12,-12, 5,-4,    -19, -7,-12, 15,16,17.0294, -6,  8, -10,14,    -15,  6,-10, -14,12,      0,-16,  0});
    NDArray<double> expU   ('c', {2,6}, {18, 2.58377,   2,  7.16409,-11,  7, 7 ,10.4525 ,-13, -7.39897 ,13 ,20});

    ops::helpers::SVD<double> svd(matrix, false, true, true);    
    svd._M = matrix;
    svd._U = matrix2;
    svd.deflation1(1,1,2,2);    
        
    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test4) {
            
    NDArray<double> matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    NDArray<double> matrix2('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    NDArray<double> expM   ('c', {6,5}, {12, 20,     19,-18, -6, 3,  6,      2, -7, -7, 14,  8,     18,-17, 18, -14,-15,8.06226,  2,  2, -3,-18,      0,-17,  2, 12, 18,      6, -2,-17});
    NDArray<double> expU   ('c', {6,6}, {-10,-16,     -20,     13, 20,-10, -9, -1,-20.7138,4.46525, -4, 20, -11, 19,-18.4812,2.72876, 12,-19, 18,-18,      17,    -10,-19, 14, -2, -7,     -17,    -14, -4,-16, 18, -6,     -18,      1,-15,-12});
    NDArray<double> expV   ('c', {5,5}, {-18,  1,     19,      -7, 1, 2,-18,    -13,      14, 2, -2,-11,2.97683,-7.69015,-6, -3, -8,      8,      -2, 7, 16, 15,     -3,       7, 0});

    ops::helpers::SVD<double> svd(matrix3, true, true, true);    
    svd._M = matrix1;
    svd._U = matrix2;
    svd._V = matrix3;
    svd.deflation2(1, 2, 2, 1, 1, 2, 1);    
        
    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
    ASSERT_TRUE(expV.equalsTo(&svd._V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test5) {
            
    NDArray<double> matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    NDArray<double> matrix2('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    NDArray<double> expM   ('c', {6,5}, {18.4391, 20,     19,-18, -6, 3,  6,      2, -7, -7, 0,  8,18.4391,-17, 18, -14,-15,      1,  2,  2, -3,-18,      8,-17,-19, 12, 18,      6, -2,-17});
    NDArray<double> expU   ('c', {6,6}, {-10,-16,-20,13, 20,-10, -9,-15.8359, -7,-12.2566, -4, 20, -11,-1.30158, -5,-26.1401, 12,-19, 18,-19.3068, 17, 7.15871,-19, 14, -2,      -7,-17,     -14, -4,-16, 18,      -6,-18,       1,-15,-12});
    NDArray<double> expV   ('c', {5,5}, {-18,       1, 19,     -7, 1, 2,-1.08465,-13,22.7777, 2, -2,-5.64019,  8,9.65341,-6, -3,      -8,  8,     -2, 7, 16,      15, -3,      7, 0});

    ops::helpers::SVD<double> svd(matrix3, true, true, true);    
    svd._M = matrix1;
    svd._U = matrix2;
    svd._V = matrix3;
    svd.deflation2(1, 0, 1, 1, 0, 2, 2);    
        
    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
    ASSERT_TRUE(expV.equalsTo(&svd._V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test6) {
            
    NDArray<double> matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    NDArray<double> matrix2('c', {2,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    NDArray<double> expM   ('c', {6,5}, {18.4391, 20,     19,-18, -6, 3,  6,      2, -7, -7, 0,  8,18.4391,-17, 18, -14,-15,      1,  2,  2, -3,-18,      8,-17,-19, 12, 18,      6, -2,-17});
    NDArray<double> expU   ('c', {2,6}, {-10, -0.542326,-20, 20.6084,20,-10, -9,  -15.8359, -7,-12.2566,-4, 20});
    NDArray<double> expV   ('c', {5,5}, {-18,       1, 19,     -7, 1, 2,-1.08465,-13,22.7777, 2, -2,-5.64019,  8,9.65341,-6, -3,      -8,  8,     -2, 7, 16,      15, -3,      7, 0});

    ops::helpers::SVD<double> svd(matrix3, false, true, true);    
    svd._M = matrix1;
    svd._U = matrix2;
    svd._V = matrix3;
    svd.deflation2(1, 0, 1, 1, 0, 2, 2);    
        
    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
    ASSERT_TRUE(expV.equalsTo(&svd._V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test7) {
            
    NDArray<double> matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    NDArray<double> matrix2('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    NDArray<double> expM   ('c', {6,5}, {12, 20,     19,-18, -6, 3,  6,      2, -7, -7, 14,  8,19.6977,-17, 18, -14,-15,      1,  2,  2, -3,-18,      0,-17,  0, 12, 18,      6, -2,-17});
    NDArray<double> expU   ('c', {6,6}, {-10,     -16,-20,      13, 20,-10, -9,-9.03658, -7,-17.8701, -4, 20, -11, 10.0519, -5,-24.1652, 12,-19, 18,  -20.51, 17,-1.82762,-19, 14, -2,-12.0826,-17,-9.95039, -4,-16, 18,      -6,-18,       1,-15,-12});
    NDArray<double> expV   ('c', {5,5}, {-18,  1, 19,-7, 1, 2,-18,-13,14, 2, -2,-11,  8, 2,-6, -3, -8,  8,-2, 7, 16, 15, -3, 7, 0});

    ops::helpers::SVD<double> svd(matrix3, true, true, true);    
    svd._M = matrix1;
    svd._U = matrix2;
    svd._V = matrix3;
    svd.deflation(1, 3, 1, 1, 2, 1);

    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
    ASSERT_TRUE(expV.equalsTo(&svd._V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test8) {
            
    NDArray<double> matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    NDArray<double> matrix2('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    NDArray<double> expM   ('c', {6,5}, {12, 20,19,-18, -6, 3,  6, 2, -7, -7, 14,-15, 2,-17, 18, -14,  8, 1, 18,  2, -3,-18, 8,-17,-19, 12, 18, 6, -2,-17});
    NDArray<double> expU   ('c', {6,6}, {-10,-20,-16, 13, 20,-10, -9, -7, -1,-20, -4, 20, -11, -5, 19,-18, 12,-19, 18, 17,-18,-10,-19, 14, -2, -7,-17,-14, -4,-16, 18, -6,-18,  1,-15,-12});                                        
    NDArray<double> expV   ('c', {5,5}, {-18,  1, 19,-7, 1, 2,-18,-13, 2,14, -2,-11,  8,-6, 2, -3, -8,  8, 7,-2, 16, 15, -3, 7, 0});                                        

    ops::helpers::SVD<double> svd(matrix3, true, true, true);    
    svd._M = matrix1;
    svd._U = matrix2;
    svd._V = matrix3;
    svd.deflation(0, 2, 2, 1, 2, 1);     

    ASSERT_TRUE(expM.equalsTo(&svd._M));        
    ASSERT_TRUE(expU.equalsTo(&svd._U));
    ASSERT_TRUE(expV.equalsTo(&svd._V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test9) {
            
    NDArray<double> col0  ('c', {10,1}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,14});
    NDArray<double> diag  ('c', {10,1}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2});
    NDArray<double> permut('c', {1,10}, {8 ,1 ,4 ,0, 5 ,2 ,9 ,3 ,7 ,6});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    NDArray<double> expSingVals('c', {10,1}, {-2, 12.862, 11.2, -1, 1.73489, -12, -15.3043, -12.862, 5.6, 41.4039});
    NDArray<double> expShifts  ('c', {10,1}, {1, 19, 19, 1, 2, -18, -18, -13, 2, 2});
    NDArray<double> expMus     ('c', {10,1}, {-3, -6.13805, -7.8, -2, -0.265108, 6, 2.69568, 0.138048, 3.6, 39.4039});

    NDArray<double> singVals('c', {10,1});
    NDArray<double> shifts  ('c', {10,1});
    NDArray<double> mus     ('c', {10,1});

    ops::helpers::SVD<double> svd(matrix3, true, true, true);        
    svd.calcSingVals(col0, diag, permut, singVals, shifts, mus);    

    ASSERT_TRUE(expSingVals.equalsTo(&singVals));        
    ASSERT_TRUE(expShifts.equalsTo(&shifts));
    ASSERT_TRUE(expMus.equalsTo(&mus));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test10) {
            
    NDArray<double> singVals('c', {4,1}, {1 ,1 ,1 ,1});
    NDArray<double> col0  ('c', {4,1}, {1 ,1 ,1 ,1});
    NDArray<double> diag  ('c', {4,1}, {5 ,7 ,-13 ,14});
    NDArray<double> permut('c', {1,4}, {0 ,2 ,3 ,1 });    
    NDArray<double> mus   ('c', {4,1}, {4,1,4,6});    
    NDArray<double> shifts('c', {4,1}, {4,2,5,6});    
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    
    NDArray<double> expZhat('c', {4,1}, {0, 0.278208, 72.502, 0});

    NDArray<double> zhat('c', {4,1});    

    ops::helpers::SVD<double> svd(matrix3, true, true, true);        
    svd.perturb(col0, diag, permut, singVals, shifts,  mus, zhat);    

    ASSERT_TRUE(expZhat(1) = zhat(1));        
    ASSERT_TRUE(expZhat(2) = zhat(2));        
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test11) {
            
    NDArray<double> singVals('c', {4,1}, {1 ,1 ,1 ,1});
    NDArray<double> zhat    ('c', {4,1}, {2 ,1 ,2 ,1});
    NDArray<double> diag  ('c', {4,1}, {5 ,7 ,-13 ,14});
    NDArray<double> permut('c', {1,4}, {0 ,2 ,3 ,1 });    
    NDArray<double> mus   ('c', {4,1}, {4,1,4,6});    
    NDArray<double> shifts('c', {4,1}, {4,2,5,6});    
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    
    NDArray<double> expU('c', {5,5}, {-0.662161, 0.980399,-0.791469,-0.748434, 0, -0.744931, 0.183825,-0.593602,-0.392928, 0, 0.0472972, 0.061275,0.0719517, 0.104781, 0, 0.0662161,0.0356509, 0.126635, 0.523904, 0, 0,        0,        0,        0, 1});
    NDArray<double> expV('c', {4,4}, {-0.745259,-0.965209, -0.899497, -0.892319, -0.652102,  0.21114,  -0.39353, -0.156156, -0.0768918,-0.130705,-0.0885868,-0.0773343, 0.115929,0.0818966,  0.167906,  0.416415});
    NDArray<double> U('c', {5,5});
    NDArray<double> V('c', {4,4});
    

    ops::helpers::SVD<double> svd(matrix3, true, true, true);        
    svd.calcSingVecs(zhat, diag,permut, singVals, shifts, mus, U, V);

    ASSERT_TRUE(expU.equalsTo(&U));        
    ASSERT_TRUE(expV.equalsTo(&V));
    
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test12) {
            
    NDArray<double> matrix1('c', {6,5}, {-2 ,-3 ,2 ,1 ,0 ,0 ,-4 ,5 ,-2 ,-3 ,-4 ,0 ,5 ,-1 ,-5 ,-3 ,-5 ,3 ,3 ,3 ,-5 ,5 ,-5 ,0 ,2 ,-2 ,-3 ,-4 ,-5 ,-3});
    NDArray<double> matrix2('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    NDArray<double> matrix4('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

    NDArray<double> expSingVals('c', {4,1}, {8.43282, 5, 2.3, 1.10167});
    NDArray<double> expU   ('c', {5,5}, {0.401972,0, 0.206791, 0.891995,0, 0,1,        0,        0,0, 0.816018,0,-0.522818,-0.246529,0, -0.415371,0,-0.826982, 0.378904,0, 0,0,        0,        0,1});
    NDArray<double> expV   ('c', {4,4}, {-0.951851,0,-0.133555,-0.275939, 0,1,        0,        0, 0.290301,0,-0.681937,-0.671333, -0.098513,0,-0.719114, 0.687873});

    ops::helpers::SVD<double> svd(matrix4, true, true, true);    
    svd._M = matrix1;
    svd._U = matrix2;
    svd._V = matrix3;
    NDArray<double> U, singVals, V;
    svd.calcBlockSVD(1, 4, U, singVals, V);

    ASSERT_TRUE(expSingVals.equalsTo(&singVals));
    ASSERT_TRUE(expU.equalsTo(&U));
    ASSERT_TRUE(expV.equalsTo(&V));

    ASSERT_TRUE(expSingVals.isSameShapeStrict(&singVals));
    ASSERT_TRUE(expU.isSameShapeStrict(&U));
    ASSERT_TRUE(expV.isSameShapeStrict(&V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test13) {
            
    NDArray<double> matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});

    NDArray<double> expQR('c', {6,5}, {-37.054 ,  0.323852 , 8.04231 , -22.9395 ,-13.089, 0.105164,    32.6021,  6.42277, -0.262898,-1.58766, 0.140218,  -0.485058,  29.2073,  -9.92301,-23.7111, -0.262909,-0.00866538, 0.103467,   8.55831,-1.86455, -0.315491,   0.539207,  0.40754,-0.0374124,-7.10401, 0.315491,   0.385363,-0.216459, -0.340008,0.628595});
    NDArray<double> expCoeffs('c', {1,5}, {1.53975, 1.19431, 1.63446, 1.7905, 1.43356});
    NDArray<double> expPermut('c', {5,5}, {0,0,0,1,0, 1,0,0,0,0, 0,0,0,0,1, 0,0,1,0,0, 0,1,0,0,0});

    ops::helpers::HHcolPivQR<double> qr(matrix1);    
    
    // qr._coeffs.printIndexedBuffer();    
    ASSERT_TRUE(expQR.equalsTo(&qr._qr));
    ASSERT_TRUE(expCoeffs.equalsTo(&qr._coeffs));
    ASSERT_TRUE(expPermut.equalsTo(&qr._permut));

    ASSERT_TRUE(expQR.isSameShapeStrict(&qr._qr));
    ASSERT_TRUE(expCoeffs.isSameShapeStrict(&qr._coeffs));
    ASSERT_TRUE(expPermut.isSameShapeStrict(&qr._permut));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test14) {
            
    NDArray<double> matrix1('c', {5,6}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});

    NDArray<double> expQR('c', {5,6}, {-32.665, -4.95944,  -8.26574,  7.22487, 16.5927, 11.7251, -0.135488, -29.0586,   10.9776, -14.6886, 4.18841, 20.7116, 0.348399, 0.323675,   25.5376,  1.64324, 9.63959, -9.0238, -0.0580664,0.0798999,-0.0799029,  19.5281,-4.97736, 16.0969, 0.348399,-0.666783, 0.0252425,0.0159188, 10.6978,-4.69198});
    NDArray<double> expCoeffs('c', {1,5}, {1.58166, 1.28555, 1.98605, 1.99949, 0});
    NDArray<double> expPermut('c', {6,6}, {0,1,0,0,0,0, 0,0,1,0,0,0, 1,0,0,0,0,0, 0,0,0,0,0,1, 0,0,0,0,1,0, 0,0,0,1,0,0});

    ops::helpers::HHcolPivQR<double> qr(matrix1);    
        
    ASSERT_TRUE(expQR.equalsTo(&qr._qr));
    ASSERT_TRUE(expCoeffs.equalsTo(&qr._coeffs));
    ASSERT_TRUE(expPermut.equalsTo(&qr._permut));

    ASSERT_TRUE(expQR.isSameShapeStrict(&qr._qr));
    ASSERT_TRUE(expCoeffs.isSameShapeStrict(&qr._coeffs));
    ASSERT_TRUE(expPermut.isSameShapeStrict(&qr._permut));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test15) {
            
    NDArray<double> matrix1('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});

    NDArray<double> expQR('c', {6,6}, {38.1707, -3.03898, 5.16103,  23.0805, -7.57126, -13.885, -0.41519,  34.3623, 3.77403,  2.62327, -8.17784, 9.10312, 0.394431, 0.509952,-30.2179, -6.78341,  12.8421, 28.5491, -0.290633, 0.111912,0.450367,  28.1139,  15.5195, 2.60562, 0.332152, 0.405161,0.308163,0.0468127,   22.294,-2.94931, 0.249114,0.0627956,0.657873,  0.76767,-0.752594,-7.46986});
    NDArray<double> expCoeffs('c', {1,6}, {1.26198, 1.38824, 1.15567, 1.25667, 1.27682, 0});
    NDArray<double> expPermut('c', {6,6}, {0,0,1,0,0,0, 0,0,0,0,1,0, 0,0,0,1,0,0, 0,1,0,0,0,0, 0,0,0,0,0,1, 1,0,0,0,0,0});

    ops::helpers::HHcolPivQR<double> qr(matrix1);    
        
    ASSERT_TRUE(expQR.equalsTo(&qr._qr));
    ASSERT_TRUE(expCoeffs.equalsTo(&qr._coeffs));
    ASSERT_TRUE(expPermut.equalsTo(&qr._permut));

    ASSERT_TRUE(expQR.isSameShapeStrict(&qr._qr));
    ASSERT_TRUE(expCoeffs.isSameShapeStrict(&qr._coeffs));
    ASSERT_TRUE(expPermut.isSameShapeStrict(&qr._permut));
}




// ///////////////////////////////////////////////////////////////////
// TEST_F(HelpersTests1, SVD_test13) {
            
//     NDArray<double> matrix1('c', {6,5}, {-2 ,-3 ,2 ,1 ,0 ,0 ,-4 ,5 ,-2 ,-3 ,-4 ,0 ,5 ,-1 ,-5 ,-3 ,-5 ,3 ,3 ,3 ,-5 ,5 ,-5 ,0 ,2 ,-2 ,-3 ,-4 ,-5 ,-3});
//     NDArray<double> matrix2('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
//     NDArray<double> matrix3('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
//     NDArray<double> matrix4('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

//     NDArray<double> expM('c', {6,5}, {-2,     -3,      2,      1,      0, 0,12.1676,      0,      0,      0, -4,      0,7.49514,      0,      0, -3,      0,      0,5.00951,      0, -5,      0,      0,      0,1.63594, -2,      0,      0,      0,      0});
//     NDArray<double> expU('c', {6,6}, {0.295543,-0.238695, 0.262095,-0.231772, -0.85631,-10, 0.519708,0.0571492,-0.368706,-0.727615, 0.247527, 20, 0.313717,-0.561567,-0.602941, 0.469567,0.0468295,-19, 0.474589,-0.372165, 0.656962, 0.124776, 0.434845, 14, -0.564717,-0.697061,0.0150082,  -0.4252, 0.119081,-16, 18,       -6,      -18,        1,      -15,-12});
//     NDArray<double> expV('c', {5,5}, {-18,         1,        19,        -7,        1, 2,-0.0366659,  0.977361,-0.0316106, 0.205967, -2, -0.670795, -0.151697, -0.503288, 0.523185, -3,  0.740124,-0.0841435, -0.486714, 0.456339, 16, 0.0300945, -0.121135,   0.71331, 0.689645});

//     ops::helpers::SVD<double> svd(matrix4, true, true, true);    
//     svd._M = matrix1;
//     svd._U = matrix2;
//     svd._V = matrix3;
    
//     svd.DivideAndConquer(0, 3, 1, 1, 1);

//     ASSERT_TRUE(expM.equalsTo(&svd._M));
//     ASSERT_TRUE(expU.equalsTo(&svd._U));
//     ASSERT_TRUE(expV.equalsTo(&svd._V));

//     ASSERT_TRUE(expM.isSameShapeStrict(&svd._M));
//     ASSERT_TRUE(expU.isSameShapeStrict(&svd._U));
//     ASSERT_TRUE(expV.isSameShapeStrict(&svd._V));
// }

