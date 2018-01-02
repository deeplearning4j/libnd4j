//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <templatemath.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    FORCEINLINE void rgb_to_hsv(T r, T g, T b, T* h, T* s, T* v) {
        T vv = nd4j::math::nd4j_max<T>(r, nd4j::math::nd4j_max<T>(g, b));
        T range = vv - nd4j::math::nd4j_min<T>(r, nd4j::math::nd4j_min<T>(g, b));
        if (vv > 0) {
            *s = range / vv;
        } else {
            *s = 0;
        }
        T norm = 1.0f / (6.0f * range);
        T hh;
        if (r == vv) {
            hh = norm * (g - b);
        } else if (g == vv) {
            hh = norm * (b - r) + 2.0 / 6.0;
        } else {
            hh = norm * (r - g) + 4.0 / 6.0;
        }
        if (range <= 0.0) {
            hh = 0;
        }
        if (hh < 0.0) {
            hh = hh + 1;
        }
        *v = vv;
        *h = hh;
    }

    template <typename T>
    FORCEINLINE void hsv_to_rgb(T h, T s, T v, T* r, T* g, T* b) {
        T c = s * v;
        T m = v - c;
        T dh = h * 6;
        T rr, gg, bb;
        int h_category = static_cast<int>(dh);
        T fmodu = dh;
        while (fmodu <= 0)
            fmodu += 2.0f;
        
        while (fmodu >= 2.0f)
            fmodu -= 2.0f;
        
        T x = c * (1 - nd4j::math::nd4j_abs<T>(fmodu - 1));
        switch (h_category) {
            case 0:
                rr = c;
                gg = x;
                bb = 0;
                break;
            case 1:
                rr = x;
                gg = c;
                bb = 0;
                break;
            case 2:
                rr = 0;
                gg = c;
                bb = x;
                break;
            case 3:
                rr = 0;
                gg = x;
                bb = c;
                break;
            case 4:
                rr = x;
                gg = 0;
                bb = c;
                break;
            case 5:
                rr = c;
                gg = 0;
                bb = x;
                break;
            default:
                rr = 0;
                gg = 0;
                bb = 0;
        }
        
        *r = rr + m;
        *g = gg + m;
        *b = bb + m;
    }

    template <typename T>
    void _adjust_saturation(NDArray<T> *input, NDArray<T> *output, T delta, bool isNHWC);
}
}
}