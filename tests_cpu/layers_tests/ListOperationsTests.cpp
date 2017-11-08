//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <GraphExecutioner.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;

class ListOperationsTests : public testing::Test {

};

TEST_F(ListOperationsTests, BasicTest_Write_1) {
    NDArrayList<double> list(5);
    NDArray<double> x('c', {1, 128});
    NDArrayFactory<double>::linspace(1, x);

    nd4j::ops::write_list<double> op;

    op.execute(&list, {&x}, {}, {1});

    ASSERT_EQ(1, list.elements());

    op.execute(&list, {&x}, {}, {2});

    ASSERT_EQ(2, list.elements());
}

TEST_F(ListOperationsTests, BasicTest_Stack_1) {
    NDArrayList<double> list(10);
    NDArray<double> exp('c', {10, 100});
    auto tads = NDArrayFactory<double>::allTensorsAlongDimension(&exp, {1});
    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 100});
        row->assign((double) e);
        list.write(e, row);
        tads->at(e)->assign(row);
    }

    nd4j::ops::stack_list<double> op;

    auto result = op.execute(&list, {}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
    delete tads;
}


TEST_F(ListOperationsTests, BasicTest_Read_1) {
    NDArrayList<double> list(10);
    NDArray<double> exp('c', {1, 100});
    exp.assign(4.0f);

    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 100});
        row->assign((double) e);
        list.write(e, row);
    }

    nd4j::ops::read_list<double> op;

    auto result = op.execute(&list, {}, {}, {4});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ListOperationsTests, BasicTest_Pick_1) {
    NDArrayList<double> list(10);
    NDArray<double> exp('c', {4, 100});

    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 100});
        row->assign((double) e);
        list.write(e, row);
    }

    auto tads = NDArrayFactory<double>::allTensorsAlongDimension(&exp, {1});
    tads->at(0)->assign(1.0f);
    tads->at(1)->assign(1.0f);
    tads->at(2)->assign(3.0f);
    tads->at(3)->assign(3.0f);


    nd4j::ops::pick_list<double> op;
    auto result = op.execute(&list, {}, {}, {1, 1, 3, 3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
    delete tads;
}

TEST_F(ListOperationsTests, BasicTest_Size_1) {
    NDArrayList<double> list(10);
    NDArray<double> exp('c', {1, 1});
    exp.putScalar(0, 10);
    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 100});
        row->assign((double) e);
        list.write(e, row);
    }

    nd4j::ops::size_list<double> op;

    auto result = op.execute(&list, {}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ListOperationsTests, BasicTest_Create_1) {
    NDArray<double> matrix('c', {3, 2});
    NDArrayFactory<double>::linspace(1, matrix);

    nd4j::ops::create_list<double> op;

    auto result = op.execute(nullptr, {&matrix}, {}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(0, result->size());

    delete result;
}

TEST_F(ListOperationsTests, BasicTest_Split_1) {
    NDArrayList<double> list(0, true);

    NDArray<double> exp0('c', {2, 5});
    NDArray<double> exp1('c', {3, 5});
    NDArray<double> exp2('c', {5, 5});

    NDArray<double> matrix('c', {10, 5});

    NDArray<double> lengths('c', {1, 3});
    lengths.putScalar(0, 2);
    lengths.putScalar(1, 3);
    lengths.putScalar(2, 5);

    auto tads = NDArrayFactory<double>::allTensorsAlongDimension(&matrix, {1});

    auto tads0 = NDArrayFactory<double>::allTensorsAlongDimension(&exp0, {1});
    auto tads1 = NDArrayFactory<double>::allTensorsAlongDimension(&exp1, {1});
    auto tads2 = NDArrayFactory<double>::allTensorsAlongDimension(&exp2, {1});

    int cnt0 = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 5});
        row->assign((double) e);
        tads->at(e)->assign(row);

        if (e < 2)
            tads0->at(cnt0++)->assign(row);
        else if (e < 5)
            tads1->at(cnt1++)->assign(row);
        else
            tads2->at(cnt2++)->assign(row);
    }

    nd4j::ops::split_list<double> op;
    auto result = op.execute(&list, {&matrix, &lengths}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, list.height());

    ASSERT_TRUE(exp0.isSameShape(list.read(0)));
    ASSERT_TRUE(exp0.equalsTo(list.read(0)));

    ASSERT_TRUE(exp1.isSameShape(list.read(1)));
    ASSERT_TRUE(exp1.equalsTo(list.read(1)));

    ASSERT_TRUE(exp2.isSameShape(list.read(2)));
    ASSERT_TRUE(exp2.equalsTo(list.read(2)));

    delete result;
    delete tads;
    delete tads0;
    delete tads1;
    delete tads2;
}

TEST_F(ListOperationsTests, BasicTest_Scatter_1) {
    NDArrayList<double> list(0, true);

    NDArray<double> matrix('c', {10, 5});
    auto tads = NDArrayFactory<double>::allTensorsAlongDimension(&matrix, {1});
    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 5});
        row->assign((double) e);
        tads->at(e)->assign(row);
    }
    NDArray<double> indices('c', {1, 10});
    for (int e = 0; e < matrix.rows(); e++)
        indices.putScalar(e, 9 - e);

    nd4j::ops::scatter_list<double> op;
    auto result = op.execute(&list, {&matrix, &indices}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    for (int e = 0; e < 10; e++) {
        auto row = tads->at(9 - e);
        auto chunk = list.read(e);

        ASSERT_TRUE(chunk->isSameShape(row));

        ASSERT_TRUE(chunk->equalsTo(row));
    }

    delete tads;
}

TEST_F(ListOperationsTests, BasicTest_Clone_1) {
    auto list = new NDArrayList<double>(0, true);

    VariableSpace<double> variableSpace;
    auto var = new Variable<double>(nullptr, nullptr, -1, 0);
    var->setNDArrayList(list);

    variableSpace.putVariable(-1, var);

    Block<double> block(1, &variableSpace);
    block.pickInput(-1);

    nd4j::ops::clone_list<double> op;

    ASSERT_TRUE(list == block.variable(0)->getNDArrayList());

    auto result = op.execute(&block);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    auto resVar = variableSpace.getVariable(1);

    auto resList = resVar->getNDArrayList();

    ASSERT_TRUE( resList != nullptr);

    ASSERT_TRUE(list->equals(*resList));
}

TEST_F(ListOperationsTests, BasicTest_Gather_1) {
    NDArrayList<double> list(0, true);
    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 10});
        row->assign((double) e);
        list.write(e, row);
    }

    NDArray<double> exp('c', {10, 10});
    auto tads = NDArrayFactory<double>::allTensorsAlongDimension(&exp, {1});
    for (int e = 0; e < 10; e++) {
        auto row = new NDArray<double>('c', {1, 10});
        auto tad = tads->at(9 - e);
        tad->assign(row);
    }

    NDArray<double> indices('c', {1, 10});
    NDArrayFactory<double>::linspace(9, indices, -1);

    nd4j::ops::gather_list<double> op;
    auto result = op.execute(&list, {&indices}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ListOperationsTests, GraphTests_Sequential_1) {
    Graph<float> graph;

    auto matrix = new NDArray<float>('c', {3, 3});
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(matrix, {1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    auto indices = new NDArray<float>('c', {1, 3});
    NDArrayFactory<float>::linspace(0, *indices);


    auto variableSpace = graph.getVariableSpace();
    variableSpace->putVariable(-1, matrix);
    variableSpace->putVariable(-2, indices);


    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1});

    // creating list
    auto nodeB = new Node<float>(OpType_CUSTOM, 0, 2, {1});
    nd4j::ops::create_list<float> opB;
    nodeB->setCustomOp(&opB);


    // filling list with matrix
    auto nodeC = new Node<float>(OpType_CUSTOM, 0, 3, {2, 1, -2});

    nd4j::ops::split_list<float> opC;
    nodeC->setCustomOp(&opC);

    graph.addNode(nodeA);
    graph.addNode(nodeB);
    graph.addNode(nodeC);


    auto result = GraphExecutioner<float>::execute(&graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    delete tads;
}