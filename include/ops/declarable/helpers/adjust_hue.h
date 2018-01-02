//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    FORCEINLINE void rgb_to_hsv(T r, T g, T b, T* h, T* v_min, T* v_max) {
        float v_mid;
        int h_category;
        // According to the figures in:
        // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
        // For the conditions, we don't care about the case where two components are
        // equal. It is okay to count it in either side in that case.
        if (r < g) {
            if (b < r) {
            // b < r < g
                *v_max = g;
                v_mid = r;
                *v_min = b;
                h_category = 1;
            } else if (b > g) {
            // r < g < b
                *v_max = b;
                v_mid = g;
                *v_min = r;
                h_category = 3;
            } else {
            // r < b < g
                *v_max = g;
                v_mid = b;
                *v_min = r;
                h_category = 2;
            }
        } else {
        // g < r
            if (b < g) {
            // b < g < r
                *v_max = r;
                v_mid = g;
                *v_min = b;
                h_category = 0;
            } else if (b > r) {
            // g < r < b
                *v_max = b;
                v_mid = r;
                *v_min = g;
                h_category = 4;
            } else {
            // g < b < r
                *v_max = r;
                v_mid = b;
                *v_min = g;
                h_category = 5;
            }
        }
        if (*v_max == *v_min) {
            *h = 0;
            return;
        }
        auto ratio = (v_mid - *v_min) / (*v_max - *v_min);
        bool increase = ((h_category & 0x1) == 0);
        *h = h_category + (increase ? ratio : (1 - ratio));
    }
}
}
}