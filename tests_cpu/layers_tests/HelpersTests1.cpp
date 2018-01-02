#include "testlayers.h"
#include <ops/declarable/helpers/householder.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <ops/declarable/helpers/hhSequence.h>


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
    // expTail.printShapeInfo();

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
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();

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
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();

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
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();

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
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    uSeq.mulLeft(matrix);
    
    ASSERT_TRUE(matrix.equalsTo(&exp));
        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test5) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> exp   ('c', {5,4}, {4.52891, 8.09473,-2.73704,-13.0302, -11.0752, 7.41549,-3.75125,0.815252, -7.76818,-15.9102,-9.90869,-11.8677, 1.63942,-17.0312,-9.05102,-4.49088, -9.63311,0.540226,-1.52764, 5.79111});
            
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    uSeq.mulLeft(matrix);
    
    ASSERT_TRUE(matrix.equalsTo(&exp));
        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test6) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,-1,3,9, -4.43019,-15.1713, -3.2854,-7.65743, -9.39162,-7.03599, 8.03827, 9.48453, -2.97785, -16.424, 5.35265,-20.1171, -0.0436177, -13.118,-8.37287,-17.3012, -1.14074, 4.18282,-10.0914,-5.69014});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    uSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test7) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    NDArray<double> exp   ('c', {4,4}, {9,13,3,6,-5.90424,-2.30926,-0.447417, 3.05712, -10.504,-9.31339, -8.85493,-10.8886, -8.29494,-10.6737, -5.94895,-7.55591});
        
    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix);    
    
    ASSERT_TRUE(matrix.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test8) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> exp   ('c', {5,4}, {9,     -13,        3,       6, 13,      11,        7,      -6, -6.90831,-5.01113, 0.381677,0.440128, -0.80107,0.961605,-0.308019,-1.96153, -0.795985, 18.6538,  12.0731, 16.9988});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix);    

    ASSERT_TRUE(matrix.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test9) {
            
    NDArray<double> matrix('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12 ,0,-15,10,2});
    NDArray<double> exp   ('c', {6,4}, {9,     -13,        3,       6, 13,      11,        7,      -6, 3,       7,        4,       7, 3.77597, 18.6226,-0.674868, 4.61365, 5.02738,-14.1486, -2.22877,-8.98245, -0.683766, 1.73722,  14.9859, 12.0843});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix);    

    ASSERT_TRUE(matrix.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test10) {
            
    NDArray<double> matrix('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,      -1,       3,        9, 10,      11,      -7,       -5, 3,       2,       4,        7, 2.58863, 11.0295,-4.17483,-0.641012, -1.21892,-16.3151, 6.12049, -20.0239, -0.901799,-15.0389,-12.4944, -20.2394});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test11) {
            
    NDArray<double> matrix('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,      -1,       3,       9, 10,      11,      -7,      -5, 3,       2,       4,       7, 1.14934, 4.40257, 8.70127,-1.18824, 1.5132,0.220419,-11.6285,-11.7549, 2.32148, 24.3838,0.256531, 25.9116});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test12) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9,      -1,       3,       9, 10,      11,      -7,      -5, 3,       2,       4,       7, -1,       6,       7,      19, -2.62252,-22.2914, 4.76743,-19.6689, -1.05943,-9.00514,-11.8013,-7.94571});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test13) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    NDArray<double> exp   ('c', {6,4}, {9 ,     -1 ,      3 ,      9, -4.65167, 3.44652, 7.83593, 22.6899, -9.48514, -21.902, 5.66559,-13.0533, -0.343184, 15.2895,  7.2888, 14.0489, 0.289638,-1.87752,   3.944,-1.49707, -2.48845, 3.18285,-10.6685,0.406502});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    uSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test14) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    NDArray<double> exp   ('c', {5,5}, {1.78958,  8.06962,-6.13687, 4.36267, 1.06472, -14.9578,  -8.1522, 1.30442,-18.3343,-13.2578, 13.5536,  5.50764, 15.7859, 7.60831, 11.7871, -1.3626,-0.634986, 7.60934, -2.1841, 5.62694, -13.0577,  15.1554, -7.6511, 3.76365,-5.87368});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> uSeq = object.getUsequence();
    uSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test15) {
            
    NDArray<double> matrix('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    NDArray<double> matrix2('c',{5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    NDArray<double> exp   ('c', {5,5}, {9,      -1,       3,       9,      10, 11,      -7,      -5,       3,       2, 4,       7,      -1,       6,       7, -9.26566,-16.4298, 1.64125,-17.3243,-7.70257, -16.7077, 4.80216,-19.1652,-2.42279,-13.0258});

    ops::helpers::BiDiagonalUp<double> object(matrix);    
    ops::helpers::HHsequence<double> vSeq = object.getVsequence();
    vSeq.mulLeft(matrix2);
    
    ASSERT_TRUE(matrix2.equalsTo(&exp));        
}
