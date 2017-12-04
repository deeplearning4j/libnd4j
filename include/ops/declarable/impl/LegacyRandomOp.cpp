//
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyRandomOp.h>
#include <helpers/RandomLauncher.h>
#include <NativeOpExcutioner.h>


namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyRandomOp<T>::LegacyRandomOp() : LegacyOp<T>::LegacyOp(1) {
            // just a no-op
        }

        template <typename T>
        LegacyRandomOp<T>::LegacyRandomOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            // just a no-op
        }

        template <typename T>
        Nd4jStatus LegacyRandomOp<T>::validateAndExecute(Context<T> &block) {
            auto input = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            /*
                (0, randomOps::UniformDistribution) ,\
                (1, randomOps::DropOut) ,\
                (2, randomOps::DropOutInverted) ,\
                (3, randomOps::ProbablisticMerge) ,\
                (4, randomOps::Linspace) ,\
                (5, randomOps::Choice) ,\
                (6, randomOps::GaussianDistribution) ,\
                (7, randomOps::BernoulliDistribution) ,\
                (8, randomOps::BinomialDistribution),\
                (9, randomOps::BinomialDistributionEx),\
                (10, randomOps::LogNormalDistribution) ,\
                (11, randomOps::TruncatedNormalDistribution) ,\
                (12, randomOps::AlphaDropOut)
            */
            switch(opNum) {
                case 0: {
                    T from = T_ARG(0);
                    T to = T_ARG(0);

                    RandomLauncher<T>::fillUniform(block.getRNG(), z, from, to);
                }
                break;
                case 1: {
                    T prob = T_ARG(0);

                    if (!block.isInplace())
                        z->assign(input);

                    RandomLauncher<T>::applyDropOut(block.getRNG(), z, prob);
                }
                break;
                case 2: {
                    T prob = T_ARG(0);

                    if (!block.isInplace())
                        z->assign(input);
                        
                    RandomLauncher<T>::applyInvertedDropOut(block.getRNG(), z, prob);
                }
                break;
                case 12: {
                    T prob = T_ARG(0);
                    T a = T_ARG(1);
                    T b = T_ARG(2);
                    T pa = T_ARG(3);

                    if (!block.isInplace())
                        z->assign(input);
                        
                    RandomLauncher<T>::applyAlphaDropOut(block.getRNG(), z, prob, a, b, pa);
                }
                break;
                default: {
                    nd4j_printf("Unknown random op requested: [%i]\n", opNum);
                    return ND4J_STATUS_KERNEL_FAILURE;
                }
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        /**
        * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and col2im. 
        * But these ops already have CustomOp implementations.
        *
        */
        template <typename T>
        ShapeList *LegacyRandomOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &ctx) {
            auto inShape = inputShape->at(0);

            int *newShape;
            ALLOCATE(newShape, ctx.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newShape);
        }


        template class ND4J_EXPORT LegacyRandomOp<float>;
        template class ND4J_EXPORT LegacyRandomOp<double>;
        template class ND4J_EXPORT LegacyRandomOp<float16>;
    }
}