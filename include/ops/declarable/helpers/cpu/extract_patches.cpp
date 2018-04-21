//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void extractPatches(NDArray<T>* images, NDArray<T>* output, 
        int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
//        std::vector<int> restDims({3});
//        std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(images, restDims));
        int row = 0;
        int e = 0;
        int ksizeRowsEffective = sizeRow + (sizeRow - 1) * (rateRow - 1);
        int ksizeColsEffective = sizeCol + (sizeCol - 1) * (rateRow - 1);
        int ksize = ksizeRowsEffective * ksizeColsEffective;
        int count = images->lengthOf() / ksize;
        int outputCount = output->lengthOf() / ksize;
        nd4j_printf("Input count %i, output count %i.\n", count, outputCount);
        for (int i = 0; i < images->lengthOf(); ) {
                for (int j = 0; j < output->sizeAt(3); ++j) 
                    (*output)(e++) = (*images)(i + j);
            if (e >= output->lengthOf()) break;
            i += ksize * images->sizeAt(3);
        }
/*
        int e = 0;
        for (int i = 0; i < output->sizeAt(0); i++) {
            for(int j = 0; j < output->sizeAt(1); j++) {
                for(int k = 0; k < output->sizeAt(2); ++k) {
                    for (int l = 0; l < output->sizeAt(3); ++l) {
                        (*output)(i,j,k,l) = (*listOfTensors->at(i))(j, k, l);
                    }
                }
            }
        }
*/
    }

    template void extractPatches(NDArray<float>* input, NDArray<float>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);
    template void extractPatches(NDArray<float16>* input, NDArray<float16>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);
    template void extractPatches(NDArray<double>* input, NDArray<double>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);
}
}
}