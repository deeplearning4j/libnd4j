//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_roll)

#include <ops/declarable/headers/parity_ops.h>
//#include <ops/declarable/helpers/cross.h>

namespace nd4j {
namespace ops {

    CONFIGURABLE_OP_IMPL(roll, 1, 1, true, 0, 1) {

        NDArray<T>* output = OUTPUT_VARIABLE(0);
        NDArray<T>* input = INPUT_VARIABLE(0);
        bool shiftIsLinear = true;
        //std::vector<int> axes(input->rankOf());
        int shift = INT_ARG(0);
        if (block.numI() > 1)
            shiftIsLinear = false;
        if (shiftIsLinear) {
            int fullLen = output->lengthOf();
            if (shift > 0) {
                shift %= fullLen;
                for (int e = 0; e < shift; ++e) {
                    int sourceIndex = fullLen - shift + e;
                    (*output)(e) = (*input)(sourceIndex);
                }

                for (int e = shift; e < fullLen; ++e) {
                    (*output)(e) = (*input)(e - shift);
                }
             }
             else if (shift < 0) {
                shift %= fullLen;
                for (int e = 0; e < fullLen + shift; ++e) {
                    (*output)(e) = (*input)(e - shift);
                }
                for (int e = fullLen + shift; e < fullLen; ++e) {
                    (*output)(e) = (*input)(e - fullLen - shift);
                }
             }
             else
                output->assign(input);
        }
        else {
            std::vector<int> axes(block.numI() - 1);
            for (unsigned e = 0; e < axes.size(); ++e) {
                int axe = INT_ARG(e + 1);
                REQUIRE_TRUE(axe < input->rankOf() && axe >= -input->rankOf(), 0, "roll: axe value should be between -%i and %i, but %i was given.",
                    input->rankOf(), input->rankOf() - 1, axe);
                axes[e] = axe;
            }
            NDArray<T>* source = input;
            for (int axe: axes) {
                std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(source, {axe}));
                std::unique_ptr<ResultSet<T>> listOfOutTensors(NDArrayFactory<T>::allTensorsAlongDimension(output, {axe}));

                int fullLen = listOfTensors->size();
                for (int k = 0; k < fullLen; k++) {
                    listOfTensors->at(k)->printIndexedBuffer("Tensor at 0");
                }
                if (shift > 0) {
                    shift %= fullLen;
                    for (int e = 0; e < shift; ++e) {
                        int sourceIndex = fullLen - shift + e;
                        listOfOutTensors->at(e)->assign(listOfTensors->at(sourceIndex));
                    }

                for (int e = shift; e < fullLen; ++e) {
                    listOfOutTensors->at(e)->assign(listOfTensors->at(e - shift));
                }
             }
             else if (shift < 0) {
                shift %= fullLen;
                for (int e = 0; e < fullLen + shift; ++e) {
                    listOfOutTensors->at(e)->assign(listOfTensors->at(e - shift));
                }
                for (int e = fullLen + shift; e < fullLen; ++e) {
                    (*output)(e) = (*input)(e - fullLen - shift);
                    listOfOutTensors->at(e)->assign(listOfTensors->at(e - fullLen - shift));
                }
             }
             else
                output->assign(input);
                
                source = output;
            }
        }

        return ND4J_STATUS_OK;
    }
}
}

#endif