//
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests7 : public testing::Test {
public:

    DeclarableOpsTests7() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_LARGE) {
    double inputData[150] = {
            0.00,  0.51,  0.68,  0.69,  0.86,  0.91,  0.96,  0.97,  0.97,  1.03,  1.13,  1.16,  1.16,  1.17,  1.19,  1.25,  1.25,  1.26,  1.27,  1.28,  1.29,  1.29,  1.29,  1.30,  1.31,  1.32,  1.33,  1.33,  1.35,  1.35,  1.36,  1.37,  1.38,  1.40,  1.41,  1.42,  1.43,  1.44,  1.44,  1.45,  1.45,  1.47,  1.47,  1.51,  1.51,  1.51,  1.52,  1.53,  1.56,  1.57,  1.58,  1.59,  1.61,  1.62,  1.63,  1.63,  1.64,  1.64,  1.66,  1.66,  1.67,  1.67,  1.70,  1.70,  1.70,  1.72,  1.72,  1.72,  1.72,  1.73,  1.74,  1.74,  1.76,  1.76,  1.77,  1.77,  1.80,  1.80,  1.81,  1.82,  1.83,  1.83,  1.84,  1.84,  1.84,  1.85,  1.85,  1.85,  1.86,  1.86,  1.87,  1.88,  1.89,  1.89,  1.89,  1.89,  1.89,  1.91,  1.91,  1.91,  1.92,  1.94,  1.95,  1.97,  1.98,  1.98,  1.98,  1.98,  1.98,  1.99,  2.00,  2.00,  2.01,  2.01,  2.02,  2.03,  2.03,  2.03,  2.04,  2.04,  2.05,  2.06,  2.07,  2.08,  2.08,  2.08,  2.08,  2.09,  2.09,  2.10,  2.10,  2.11,  2.11,  2.11,  2.12,  2.12,  2.13,  2.13,  2.14,  2.14,  2.14,  2.14,  2.15,  2.15,  2.16,  2.16,  2.16,  2.16,  2.16,  2.17
    };

    NDArray<double> x(inputData,'c',{1,149});
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {0.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(1);
    auto array = *z;
    ASSERT_EQ(148,array(0));
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_ZERO) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {0.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(1);
    auto array = *z;
    ASSERT_EQ(3,array(0));
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    NDArray<double> scalar('c',{1,1},{0.0});
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x,&scalar}, {1.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_LEFT) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    NDArray<double> scalar('c',{1,1},{0.0});
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&scalar,&x}, {1.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_ONLY_SCALAR) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {1.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_ONLY_SCALAR_GTE) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {1.0},{5});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}





TEST_F(DeclarableOpsTests7, TEST_WHERE) {
    std::vector<double> data;
    std::vector<double> mask;
    std::vector<double> put;
    std::vector<double> resultData;
    std::vector<double> assertion;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
        if(i >  1) {
            assertion.push_back(5.0);
            mask.push_back(1);
        }
        else {
            assertion.push_back(i);
            mask.push_back(0);
        }

        put.push_back(5.0);
        resultData.push_back(0.0);
    }




    NDArray<double> x('c',{1,4},data);
    NDArray<double> maskArr('c',{1,4},mask);
    NDArray<double> putArr('c',{1,4},put);
    NDArray<double> resultArr('c',{1,4},resultData);
    nd4j::ops::where_np<double> op;
    //greater than test
    //            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

    auto result = op.execute({&maskArr,&x,&putArr},{&resultArr}, {},{3},false);
    // ASSERT_EQ(Status::OK(), result->status());
    for(int i = 0; i < 4; i++)
        ASSERT_EQ(assertion[i],resultArr(i));
    // auto z = result->at(0);
    //ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));


}

TEST_F(DeclarableOpsTests7,TEST_WHERE_MASK) {
    double x[300] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    double z[300] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    double mask[300] = {0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00};
    double put[200] = {0.99666107,0.9867112,0.97686064,0.9671082,0.95745337,0.9478948,0.9384318,0.92906314,0.9197881,0.91060543,0.9015147,0.8925147,0.8836044,0.8747831,0.86605,0.85740393,0.8488442,0.84037,0.83198035,0.8236745,0.8154515,0.8073106,0.79925096,0.79127187,0.7833724,0.77555174,0.76780915,0.7601439,0.75255525,0.7450422,0.7376043,0.73024046,0.72295034,0.715733,0.7085876,0.7015135,0.69451016,0.68757665,0.6807124,0.6739167,0.66718876,0.66052806,0.6539338,0.6474054,0.6409421,0.6345435,0.6282087,0.6219371,0.6157281,0.60958105,0.6034956,0.59747064,0.5915059,0.5856007,0.57975453,0.5739667,0.5682366,0.5625637,0.5569475,0.5513874,0.54588276,0.540433,0.53503764,0.5296962,0.52440816,0.51917285,0.5139898,0.5088585,0.50377846,0.4987491,0.4937699,0.48884052,0.48396033,0.47912875,0.47434545,0.4696099,0.46492168,0.46028027,0.45568514,0.4511359,0.44663212,0.4421733,0.43775895,0.43338865,0.42906195,0.42477852,0.4205379,0.41633952,0.41218308,0.40806815,0.40399432,0.3999611,0.3959682,0.39201516,0.38810158,0.384227,0.38039115,0.37659356,0.37283397,0.3691119,0.36542687,0.36177874,0.35816705,0.3545914,0.35105142,0.34754673,0.34407702,0.34064204,0.33724132,0.3338745,0.33054137,0.3272415,0.32397458,0.32074028,0.3175382,0.31436813,0.31122974,0.3081226,0.30504647,0.30200112,0.2989862,0.29600134,0.29304633,0.2901207,0.28722438,0.28435695,0.2815181,0.27870762,0.27592525,0.27317056,0.27044344,0.26774356,0.26507056,0.2624243,0.25980446,0.25721073,0.25464293,0.25210077,0.249584,0.24709237,0.24462552,0.24218333,0.23976555,0.23737194,0.23500215,0.23265606,0.23033342,0.22803394,0.22575743,0.2235036,0.22127232,0.21906327,0.21687631,0.21471114,0.21256764,0.21044552,0.20834461,0.20626466,0.20420544,0.20216681,0.20014854,0.19815037,0.19617215,0.19421372,0.19227484,0.19035533,0.18845497,0.18657354,0.18471093,0.18286693,0.18104129,0.17923392,0.17744459,0.17567308,0.1739193,0.17218304,0.17046405,0.16876228,0.16707748,0.16540948,0.16375816,0.16212334,0.16050482,0.15890247,0.15731607,0.15574552,0.15419069,0.15265137,0.15112738,0.14961864,0.14812498,0.14664622,0.1451822,0.14373279,0.14229788,0.14087726,0.13947085,0.13807845,0.13669999,0.13533528};
    double assertion[300] = {1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,9.966611049434810354e-01,9.867111603284486332e-01,9.768605487739230320e-01,9.671082786103732953e-01,9.574533680683808834e-01,9.478948451798039354e-01,9.384317476799283186e-01,9.290631229105962285e-01,9.197880277243004610e-01,9.106055283892373620e-01,9.015147004953073528e-01,8.925146288610534828e-01,8.836044074415293492e-01,8.747831392370875037e-01,8.660499362030764647e-01,8.574039191604412302e-01,8.488442177072155204e-01,8.403699701308978698e-01,8.319803233217017979e-01,8.236744326866727306e-01,8.154514620646623468e-01,8.073105836421510251e-01,7.992509778699116163e-01,7.912718333805045523e-01,7.833723469065965173e-01,7.755517232000953554e-01,7.678091749520912224e-01,7.601439227135980969e-01,7.525551948170853267e-01,7.450422272987937689e-01,7.376042638218265335e-01,7.302405556000080011e-01,7.229503613225031211e-01,7.157329470791886639e-01,7.085875862867698771e-01,7.015135596156351072e-01,6.945101549174396149e-01,6.875766671534137009e-01,6.807123983233853703e-01,6.739166573955123196e-01,6.671887602367149173e-01,6.605280295438040739e-01,6.539337947752965619e-01,6.474053920839111242e-01,6.409421642497381555e-01,6.345434606140767375e-01,6.282086370139332576e-01,6.219370557171712832e-01,6.157280853583116942e-01,6.095811008749726367e-01,6.034954834449430816e-01,5.974706204238864338e-01,5.915059052836644238e-01,5.856007375512777280e-01,5.797545227484157682e-01,5.739666723316099173e-01,5.682366036329845604e-01,5.625637398015992385e-01,5.569475097453767676e-01,5.513873480736106725e-01,5.458826950400470501e-01,5.404329964865340896e-01,5.350377037872348085e-01,5.296962737933965659e-01,5.244081687786711354e-01,5.191728563849821176e-01,5.139898095689314772e-01,5.088585065487419845e-01,5.037784307517284565e-01,4.987490707622945774e-01,4.937699202704479151e-01,4.888404780208293054e-01,4.839602477622509946e-01,4.791287381977387683e-01,4.743454629350723484e-01,4.696099404378203390e-01,4.649216939768630041e-01,4.602802515824001017e-01,4.556851459964368911e-01,4.511359146257447605e-01,4.466320994952920342e-01,4.421732472021388527e-01,4.377589088697927955e-01,4.333886401030203062e-01,4.290620009431086457e-01,4.247785558235752101e-01,4.205378735263185508e-01,4.163395271382073215e-01,4.121830940081024908e-01,4.080681557043087104e-01,4.039942979724505667e-01,3.999611106937689398e-01,3.959681878438343627e-01,3.920151274516718853e-01,3.881015315592946102e-01,3.842270061816405180e-01,3.803911612669100828e-01,3.765936106572991271e-01,3.728339720501240850e-01,3.691118669593352886e-01,3.654269206774144463e-01,3.617787622376523182e-01,3.581670243768036999e-01,3.545913434981138868e-01,3.510513596347161203e-01,3.475467164133922426e-01,3.440770610186974499e-01,3.406420441574410929e-01,3.372413200235238606e-01,3.338745462631242389e-01,3.305413839402346898e-01,3.272414975025391692e-01,3.239745547476344245e-01,3.207402267895853032e-01,3.175381880258169032e-01,3.143681161043347383e-01,3.112296918912743071e-01,3.081225994387726264e-01,3.050465259531625062e-01,3.020011617634821843e-01,2.989862002903017069e-01,2.960013380148582840e-01,2.930462744485015647e-01,2.901207121024425017e-01,2.872243564578055852e-01,2.843569159359789489e-01,2.815181018692606840e-01,2.787076284717992514e-01,2.759252128108221624e-01,2.731705747781537075e-01,2.704434370620155681e-01,2.677435251191103149e-01,2.650705671469821278e-01,2.624242940566549609e-01,2.598044394455423789e-01,2.572107395706292876e-01,2.546429333219200064e-01,2.521007621961529055e-01,2.495839702707757235e-01,2.470923041781825646e-01,2.446255130802063582e-01,2.421833486428674187e-01,2.397655650113727777e-01,2.373719187853666479e-01,2.350021689944260528e-01,2.326560770738031469e-01,2.303334068404078172e-01,2.280339244690317291e-01,2.257573984688081292e-01,2.235035996599082919e-01,2.212723011504689752e-01,2.190632783137518302e-01,2.168763087655291855e-01,2.147111723416972873e-01,2.125676510761114746e-01,2.104455291786438698e-01,2.083445930134591173e-01,2.062646310775079761e-01,2.042054339792348794e-01,2.021667944174980747e-01,2.001485071607009836e-01,1.981503690261307848e-01,1.961721788595043592e-01,1.942137375147174327e-01,1.922748478337968081e-01,1.903553146270518526e-01,1.884549446534251604e-01,1.865735466010380594e-01,1.847109310679319050e-01,1.828669105430000552e-01,1.810412993871116094e-01,1.792339138144224131e-01,1.774445718738737465e-01,1.756730934308744496e-01,1.739193001491673995e-01,1.721830154728755669e-01,1.704640646087285105e-01,1.687622745084652875e-01,1.670774738514141378e-01,1.654094930272448083e-01,1.637581641188943782e-01,1.621233208856623365e-01,1.605047987464754966e-01,1.589024347633189727e-01,1.573160676248336609e-01,1.557455376300762306e-01,1.541906866724424563e-01,1.526513582237501165e-01,1.511273973184814046e-01,1.496186505381822129e-01,1.481249659960175158e-01,1.466461933214808777e-01,1.451821836452561187e-01,1.437327895842310799e-01,1.422978652266598532e-01,1.408772661174743090e-01,1.394708492437411185e-01,1.380784730202649913e-01,1.366999972753347725e-01,1.353352832366127023e-01};
    int threeHundredShapePointer[8] = {2,1,300,1,1,0,1,99};
    int twoHundredShapePointer[8] = {2,1,200,1,1,0,1,99};
    nd4j::ops::where_np<double> op;

    NDArray<double> xArr(x,threeHundredShapePointer);
    NDArray<double> maskArr(mask,threeHundredShapePointer);
    NDArray<double> putArr(put,twoHundredShapePointer);
    NDArray<double> resultArr(z,threeHundredShapePointer);
    resultArr.assign(0.0);
    NDArray<double> assertArr(assertion,threeHundredShapePointer);
    Nd4jStatus result = op.execute({&maskArr,&xArr,&putArr},{&resultArr},{},{},false);
    ASSERT_EQ(Status::OK(),result);
    ASSERT_TRUE (assertArr.equalsTo(&resultArr));



}

TEST_F(DeclarableOpsTests7, TEST_WHERE_SCALAR) {
    std::vector<double> data;
    std::vector<double> mask;
    std::vector<double> put;
    std::vector<double> resultData;
    std::vector<double> assertion;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
        if(i >  1) {
            assertion.push_back(5.0);
            mask.push_back(1);
        }
        else {
            assertion.push_back(i);
            mask.push_back(0);
        }

        resultData.push_back(0.0);
    }


    put.push_back(5.0);


    NDArray<double> x('c',{1,4},data);
    NDArray<double> maskArr('c',{1,4},mask);
    NDArray<double> putArr('c',{1,1},put);
    NDArray<double> resultArr('c',{1,4},resultData);
    nd4j::ops::where_np<double> op;
    //greater than test
    //            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

    auto result = op.execute({&maskArr,&x,&putArr},{&resultArr}, {},{3},false);
    // ASSERT_EQ(Status::OK(), result->status());
    for(int i = 0; i < 4; i++)
        ASSERT_EQ(assertion[i],resultArr(i));
    // auto z = result->at(0);
    //ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));


}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiagPart_1) {
    NDArray<double> x('c', {2, 4, 4}, {1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.,
                                       5., 0., 0., 0., 0., 6., 0., 0., 0., 0., 7., 0., 0., 0., 0., 8.});
    NDArray<double> z('c', {2, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

    nd4j::ops::matrix_diag_part<double> op;

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(z.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiagPart_2) {
    NDArray<double> x('c', {2, 3, 4}, {1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0.,
                                       5., 0., 0., 0., 0., 6., 0., 0., 0., 0., 7., 0.});
    NDArray<double> z('c', {2, 3}, {1.0, 2.0, 3.0, 5.0, 6.0, 7.0});

    nd4j::ops::matrix_diag_part<double> op;

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(z.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiag_1) {
    NDArray<double> z('c', {2, 4, 4}, {1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.,
                                       5., 0., 0., 0., 0., 6., 0., 0., 0., 0., 7., 0., 0., 0., 0., 8.});
    NDArray<double> x('c', {2, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

    nd4j::ops::matrix_diag<double> op;

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(z.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiag_2) {
    NDArray<double> z('c', {2, 3, 3}, {1., 0., 0., 
                                       0., 2., 0., 
                                       0., 0., 3.,
                                            5., 0., 0., 
                                            0., 6., 0.,
                                            0., 0., 7.});
    NDArray<double> x('c', {2, 3}, {1.0, 2.0, 3.0, 5.0, 6.0, 7.0});

    nd4j::ops::matrix_diag<double> op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(z.equalsTo(result->at(0)));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRandomCrop_1) {
    NDArray<double> x('c', {2, 2, 4}, {
        1.8, 2.5,  4.,  9., 
        2.1, 2.4,  3.,  9.,
        2.1, 2.1, 0.7, 0.1,
         3., 4.2, 2.2, 1. 

    });
    NDArray<double> shape({1.0, 2.0, 3.0});
    nd4j::ops::random_crop<double> op;

    auto result = op.execute({&x, &shape}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    result->at(0)->printIndexedBuffer("Output");
//    ASSERT_TRUE(z.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRandomCrop_2) {
    NDArray<double> x('c', {2, 2, 4}, {
        1.8, 2.5,  4.,  9., 
        2.1, 2.4,  3.,  9.,
        2.1, 2.1, 0.7, 0.1,
         3., 4.2, 2.2, 1. 

    });
    NDArray<double> shape({2.0, 2.0, 2.0});
    nd4j::ops::random_crop<double> op;

    auto result = op.execute({&x, &shape}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    result->at(0)->printIndexedBuffer("Output");
//    ASSERT_TRUE(z.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_1) {
    NDArray<double> x({1.8, 2.5,  
                        4.,  9., 2.1, 2.4,  
                        3.,  
                        9., 2.1, 2.1, 
                       0.7, 0.1, 3., 4.2, 2.2, 1. 
    });
    NDArray<double> idx({0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0});
    NDArray<double> exp({2.5, 9.0, 3.0, 9.0, 4.2});

    nd4j::ops::segment_max<double> op;

    auto result = op.execute({&x, &idx}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_2) {
    NDArray<double> x('c', {4, 4}, {
        1.8, 2.5,  4.,  9., 
        2.1, 2.4,  3.,  9.,
        2.1, 2.1, 0.7, 0.1,
         3., 4.2, 2.2, 1. 
    });
    NDArray<double> idx({0.0, 0.0, 1.0, 2.0});
    NDArray<double> exp('c', {3, 4}, {2.1, 2.5, 4.0, 9.0,
                                      2.1, 2.1, 0.7, 0.1,
                                       3., 4.2, 2.2, 1.}); 

    //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

    nd4j::ops::segment_max<double> op;

    auto result = op.execute({&x, &idx}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    ASSERT_EQ(result->size(), 1);
//    exp.printIndexedBuffer("Expect");
//    exp.printShapeInfo("Exp Shape");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_3) {
    NDArray<double> x('c', {4, 4, 4}, {
     91. ,  82. ,  37. ,  64. ,
     55.1,  46.4,  73. ,  28. ,
    119.1,  12.1, 112.7,  13.1,
     14. , 114.2,  16.2, 117. ,

     51. ,  42. ,  67. ,   24.,
     15.1,  56.4,  93. ,   28.,
    109.1,  82.1,  12.7, 113.1,
    114. ,  14.2, 116.2,  11. ,

     31. ,  22. ,  87.,   44. ,
     55.1,  46.4,  73.,   28. ,
    119.1,  12.1, 112.7,  13.1,
     14. , 114.2,  16.2, 117. ,

     91. ,  82. ,  37.,   64. ,
     55.1,  46.4,  73.,   28. ,
    119.1,  12.1, 112.7,  13.1,
     14. , 114.2,  16.2, 117. });

// ----------------------------------------------------------------

    NDArray<double> idx({0.0, 1.0, 1.0, 2.0});
    NDArray<double> exp('c', {3, 4, 4}, {
                     91. , 82. , 37. , 64.,
                     55.1, 46.4, 73. , 28.,
                    119.1, 12.1,112.7, 13.1,
                     14. ,114.2, 16.2,117.,
    
                     51. , 42. , 87. , 44.,
                     55.1, 56.4, 93. , 28.,
                    119.1, 82.1,112.7,113.1,
                    114. ,114.2,116.2,117.,
    
                     91. , 82. , 37. , 64.,
                     55.1, 46.4, 73. , 28.,
                    119.1, 12.1,112.7, 13.1,
                     14. ,114.2, 16.2,117. }); 

    //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

    nd4j::ops::segment_max<double> op;

    auto result = op.execute({&x, &idx}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    result->at(0)->printIndexedBuffer("Output");
    result->at(0)->printShapeInfo("Out Shape");
    exp.printIndexedBuffer("Expect");
    exp.printShapeInfo("Exp Shape");
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

