//
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_im2col(const T* data_im, const int channels,
                                const int height, const int width, const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w,
                                const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                T* data_col) {
            const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
            const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
            const int channels_col = channels * kernel_h * kernel_w;

            for (int c_col = 0; c_col < channels_col; ++c_col) {
                int w_offset = c_col % kernel_w;
                int h_offset = (c_col / kernel_w) % kernel_h;
                int c_im = c_col / kernel_h / kernel_w;

                for (int h_col = 0; h_col < height_col; ++h_col) {
                    for (int w_col = 0; w_col < width_col; ++w_col) {
                        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                        data_col[(c_col * height_col + h_col) * width_col + w_col] = (h_im >= (int) 0 && w_im >= (int) 0 && h_im < height &&
                                                                                      w_im < width) ?
                                                                                     data_im[(c_im * height + h_im) * width +
                                                                                             w_im] : (T) 0.f;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_calcPadding2D(int& pH, int& pW, int oH, int oW, int inH, int inW, int kH, int kW, int sH, int sW, int dH, int dW) {
            int eKH, eKW;

            if (dH == 1 && dW == 1) {
                eKH = kH;
                eKW = kW;
            } else {
                eKH = kH + (kH - 1) * (dH - 1);
                eKW = kW + (kW - 1) * (dW - 1);
            }

            pH = ((oH - 1) * sH + eKH - inH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
            pW = ((oW - 1) * sW + eKW - inW) / 2;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW) {

            int eKD, eKH, eKW;
            
            if (dD == 1 && dH == 1 && dW == 1) {
                eKD = kD;
                eKH = kH;
                eKW = kW;
            } else {
                eKD = kD + (kD - 1) * (dD - 1);
                eKH = kH + (kH - 1) * (dH - 1);
                eKW = kW + (kW - 1) * (dW - 1);
            }

            pD = ((oD - 1) * sD + eKD - iD) / 2;       // Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
            pH = ((oH - 1) * sH + eKH - iH) / 2; 
            pW = ((oW - 1) * sW + eKW - iW) / 2;

        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::vol2col(const T *inBuff, T* outBuff, const int iC, 
                                            const int iD, const int iH, const int iW, 
                                            const int oD, const int oH, const int oW, 
                                            const int kD, const int kH, const int kW, 
                                            const int sD, const int sH, const int sW,
                                            const int pD, const int pH, const int pW,  
                                            const int dD, const int dH, const int dW ) {
            int c, d, h, w;    
            int outDim = iC * kD * kH * kW;
            
            for (c = 0; c < outDim; ++c) {
                
                int w_offset = c % kW;
                int h_offset = (c / kW) % kH;
                int d_offset = (c / kW / kH) % kD;
                int c_vol = c / kD / kH / kW;
                
                for (d = 0; d < oD; ++d) {
                    for (h = 0; h < oH; ++h) {
                        for (w = 0; w < oW; ++w) {
                            
                            int d_pad = d * sD - pD + d_offset * dD;
                            int h_pad = h * sH - pH + h_offset * dH;
                            int w_pad = w * sW - pW + w_offset * dW;
                            
                            if (d_pad >= 0 && d_pad < iD && h_pad >= 0 && h_pad < iH && w_pad >= 0 && w_pad < iW)
                                outBuff[((c * oD + d) * oH + h) * oW + w] = inBuff[((c_vol * iD + d_pad) * iH + h_pad) * iW + w_pad];
                            else
                                outBuff[((c * oD + d) * oH + h) * oW + w] = 0;
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::col2vol(const T* inBuff, T* outBuff, 
                                          const int iD, const int iH, const int iW, 
                                          const int oC, const int oD, const int oH, const int oW, 
                                          const int kD, const int kH, const int kW, 
                                          const int sD, const int sH, const int sW, 
                                          const int pD, const int pH, const int pW, 
                                          const int dD, const int dH, const int dW) {

            int c, t, h, w;
            memset(outBuff, 0, sizeof(T) * oD * oH * oW * oC);

            int kEffectiveD = kD + (kD - 1) * (dD - 1);
            int kEffectiveH = kH + (kH - 1) * (dH - 1);
            int kEffectiveW = kW + (kW - 1) * (dW - 1);            

            int inDim = oC * kEffectiveD * kEffectiveH * kW;
            for (c = 0; c < inDim; ++c) {
                
                int w_offset = c % kEffectiveW;
                int h_offset = (c / kEffectiveW) % kEffectiveH;
                int t_offset = (c / kEffectiveW / kEffectiveH) % kEffectiveD;
                int c_vol = c / kEffectiveD / kEffectiveH / kEffectiveW;

                for (t = 0; t < iD; ++t) {
                    for (h = 0; h < iH; ++h) {
                        for (w = 0; w < iW; ++w) {
                
                            int t_pad = t * sD - pD + t_offset * dD;
                            int h_pad = h * sH - pH + h_offset * dH;
                            int w_pad = w * sW - pW + w_offset * dW;
                
                            if (t_pad >= 0 && t_pad < oD && h_pad >= 0 && h_pad < oH && w_pad >= 0 && w_pad < oW)
                                outBuff[((c_vol * oD + t_pad) * oH + h_pad) * oW + w_pad] += inBuff[((c * iD + t) * iH + h) * iW + w];
                        }
                    }
                }
            }
        }        

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_vol2col(const T *data_vol, const int channels, const int depth, const int height, const int width, const int kT, const int kH, const int kW, const int pT, const int pH, const int pW, const int dT, const int dH, const int dW, const int dilationT, const int dilationH, const int dilationW, T *data_col) {
            int c, t, h, w;
            int depth_col  = (depth  + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1;
            int height_col = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1;
            int width_col  = (width  + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1;
            int channels_col = channels * kT * kH * kW;
            for (c = 0; c < channels_col; ++c)
            {
                int w_offset = c % kW;
                int h_offset = (c / kW) % kH;
                int t_offset = (c / kW / kH) % kT;
                int c_vol = c / kT / kH / kW;
                for (t = 0; t < depth_col; ++t)
                {
                    for (h = 0; h < height_col; ++h)
                    {
                        for (w = 0; w < width_col; ++w)
                        {
                            int t_pad = t * dT - pT + t_offset * dilationT;
                            int h_pad = h * dH - pH + h_offset * dilationH;
                            int w_pad = w * dW - pW + w_offset * dilationW;
                            if (t_pad >= 0 && t_pad < depth &&
                                h_pad >= 0 && h_pad < height &&
                                w_pad >= 0 && w_pad < width)
                                data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                                        data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad];
                            else
                                data_col[((c * depth_col + t) * height_col + h) * width_col + w] = 0;
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_col2vol(const T* data_col, const int channels, const int depth, const int height, const int width, const int out_depth, const int out_height, const int out_width, const int kT, const int kH, const int kW, const int pT, const int pH, const int pW, const int dT, const int dH, const int dW, const int dilationT, const int dilationH, const int dilationW, T* data_vol) {
            int c, t, h, w;
            memset(data_vol, 0, sizeof(T) * depth * height * width * channels);
            int depth_col  = out_depth;
            int height_col = out_height;
            int width_col  = out_width;
            int channels_col = channels * kT * kH * kW;
            for (c = 0; c < channels_col; ++c)
            {
                int w_offset = c % kW;
                int h_offset = (c / kW) % kH;
                int t_offset = (c / kW / kH) % kT;
                int c_vol = c / kT / kH / kW;
                for (t = 0; t < depth_col; ++t)
                {
                    for (h = 0; h < height_col; ++h)
                    {
                        for (w = 0; w < width_col; ++w)
                        {
                            int t_pad = t * dT - pT + t_offset * dilationT;
                            int h_pad = h * dH - pH + h_offset * dilationH;
                            int w_pad = w * dW - pW + w_offset * dilationW;
                            if (t_pad >= 0 && t_pad < depth &&
                                h_pad >= 0 && h_pad < height &&
                                w_pad >= 0 && w_pad < width)
                                data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
                                        data_col[((c * depth_col + t) * height_col + h) * width_col + w];
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_avgPool3D_bp(T *gradI_p, T *gradO_p, Nd4jIndex iC, Nd4jIndex iD, Nd4jIndex iH, Nd4jIndex iW, Nd4jIndex oD, Nd4jIndex oH, Nd4jIndex oW, int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, bool count_include_pad) {
            for (int k = 0; k < iC; k++)
            {
                Nd4jIndex i, j, ti;

                /* local pointers */
                T *ip = gradI_p + k * iD * iW * iH;
                T *op = gradO_p + k * oD * oW * oH;
                for (i = 0; i < iD*iW*iH; i++)
                    *(ip + i) = 0;

                /* loop over output */
                for (ti = 0; ti < oD; ti++)
                {
                    for (i = 0; i < oH; i++)
                    {
                        for (j = 0; j < oW; j++)
                        {
                            Nd4jIndex cstart = ti * sD - pD;
                            Nd4jIndex hstart = i  * sH - pH;
                            Nd4jIndex wstart = j  * sW - pW;
                            Nd4jIndex cend = nd4j::math::nd4j_min<Nd4jIndex>(cstart + kD, iD + pD);
                            Nd4jIndex hend = nd4j::math::nd4j_min<Nd4jIndex>(hstart + kH, iH + pH);
                            Nd4jIndex wend = nd4j::math::nd4j_min<Nd4jIndex>(wstart + kW, iW + pW);
                            Nd4jIndex pool_size = (cend -cstart) * (hend - hstart) * (wend - wstart);
                            cstart = nd4j::math::nd4j_max<Nd4jIndex>(cstart, 0);
                            hstart = nd4j::math::nd4j_max<Nd4jIndex>(hstart, 0);
                            wstart = nd4j::math::nd4j_max<Nd4jIndex>(wstart, 0);
                            cend = nd4j::math::nd4j_min<Nd4jIndex>(cend, iD);
                            hend = nd4j::math::nd4j_min<Nd4jIndex>(hend, iH);
                            wend = nd4j::math::nd4j_min<Nd4jIndex>(wend, iW);

                            Nd4jIndex divide_factor;
                            if (count_include_pad)
                                divide_factor = pool_size;
                            else
                                divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                            /* scatter gradients out to footprint: */
                            T val  = *op++;

                            long x,y,z;
                            for (z = cstart; z < cend; z++)
                            {
                                for (y = hstart; y < hend; y++)
                                {
                                    for (x = wstart; x < wend; x++)
                                    {
                                        *(ip + z * iH * iW + y * iW + x) += val / divide_factor;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_avgPool3D(T *input_p, T *output_p, Nd4jIndex iC, Nd4jIndex iD, Nd4jIndex iH, Nd4jIndex iW, Nd4jIndex oD, Nd4jIndex oH, Nd4jIndex oW, int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, bool count_include_pad) {
            for (Nd4jIndex k = 0; k < iC; k++)
            {
                long i, j, ti;

                /* local pointers. */
                T *ip = input_p + k * iD * iW * iH;
                T *op = output_p + k * oD * oW * oH;
                for (i = 0; i < oD * oH * oW; ++i)
                    *(op + i) = 0;

                /* loop over output */
                for (ti = 0; ti < oD; ti++)
                {
                    for (i = 0; i < oH; i++)
                    {
                        for (j = 0; j < oW; j++)
                        {
                            /* compute pool range. */
                            Nd4jIndex cstart = ti * sD - pD;
                            Nd4jIndex hstart = i  * sH - pH;
                            Nd4jIndex wstart = j  * sW - pW;
                            Nd4jIndex cend = nd4j::math::nd4j_min<Nd4jIndex>(cstart + kD, iD + pD);
                            Nd4jIndex hend = nd4j::math::nd4j_min<Nd4jIndex>(hstart + kH, iH + pH);
                            Nd4jIndex wend = nd4j::math::nd4j_min<Nd4jIndex>(wstart + kW, iW + pW);
                            Nd4jIndex pool_size = (cend - cstart) * (hend - hstart) * (wend - wstart);
                            cstart = nd4j::math::nd4j_max<Nd4jIndex>(cstart, 0);
                            hstart = nd4j::math::nd4j_max<Nd4jIndex>(hstart, 0);
                            wstart = nd4j::math::nd4j_max<Nd4jIndex>(wstart, 0);
                            cend = nd4j::math::nd4j_min<Nd4jIndex>(cend, iD);
                            hend = nd4j::math::nd4j_min<Nd4jIndex>(hend, iH);
                            wend = nd4j::math::nd4j_min<Nd4jIndex>(wend, iW);

                            Nd4jIndex divide_factor;
                            if (count_include_pad)
                                divide_factor = pool_size;
                            else
                                divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                            /* compute local sum: */
                            T sum = (T) 0.0f;
                            long x, y, z;

                            for (z = cstart; z < cend; z++)
                            {
                                for (y = hstart; y < hend; y++)
                                {
                                    for (x = wstart; x < wend; x++)
                                    {
                                        sum +=  *(ip + z * iW * iH + y * iW + x);
                                    }
                                }
                            }

                            /* set output to local max */
                            *op++ += sum / divide_factor;
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_dilatedMaxPool3D_bp(T *gradInput_p, T *gradOutput_p, T *indz_p, Nd4jIndex nslices, Nd4jIndex  itime, Nd4jIndex  iwidth, Nd4jIndex  iheight, Nd4jIndex otime, Nd4jIndex owidth, Nd4jIndex oheight, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH) {
            for (int k = 0; k < nslices; k++)
            {
                T *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
                T *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
                T *indz_p_k = indz_p + k * otime * owidth * oheight;

                /* calculate max points */
                long ti, i, j;
                for (ti = 0; ti < otime; ti++)
                {
                    for (i = 0; i < oheight; i++)
                    {
                        for (j = 0; j < owidth; j++)
                        {
                            /* retrieve position of max */
                            T * indzp = &indz_p_k[ti * oheight * owidth + i * owidth + j];
                            long maxti = ((unsigned char*)(indzp))[0] * dilationT + ti * dT - pT;
                            long maxi  = ((unsigned char*)(indzp))[1] * dilationH + i * dH - pH;
                            long maxj  = ((unsigned char*)(indzp))[2] * dilationW + j * dW - pW;

                            if (maxti != -1) {
                                /* update gradient */
                                gradInput_p_k[maxti * iheight * iwidth + maxi * iwidth + maxj] += gradOutput_p_k[ti * oheight * owidth + i * owidth + j];
                            }
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_dilatedMaxPool3D(T *input_p, T *output_p, T *indz_p, Nd4jIndex nslices, Nd4jIndex itime, Nd4jIndex iwidth, Nd4jIndex iheight, Nd4jIndex otime, Nd4jIndex owidth, Nd4jIndex oheight, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH) {
            Nd4jIndex k;
//#pragma omp parallel for private(k)
            for (k = 0; k < nslices; k++)
            {
                /* loop over output */
                Nd4jIndex i, j, ti;
                for (ti = 0; ti < otime; ti++)
                {
                    for (i = 0; i < oheight; i++)
                    {
                        for (j = 0; j < owidth; j++)
                        {
                            /* local pointers */

                            Nd4jIndex start_t = ti * dT - pT;
                            Nd4jIndex start_h = i * dH - pH;
                            Nd4jIndex start_w = j * dW - pW;

                            Nd4jIndex kernel_t = nd4j::math::nd4j_min<Nd4jIndex>(kT, kT + start_t);
                            Nd4jIndex kernel_h = nd4j::math::nd4j_min<Nd4jIndex>(kH, kH + start_h);
                            Nd4jIndex kernel_w = nd4j::math::nd4j_min<Nd4jIndex>(kW, kW + start_w);

                            while(start_t < 0)
                                start_t += dilationT;
                            while(start_h < 0)
                                start_h += dilationH;
                            while(start_w < 0)
                                start_w += dilationW;

                            T *ip = input_p + k * itime * iwidth * iheight + start_t * iwidth * iheight + start_h * iwidth + start_w;
                            T *op = output_p + k * otime * owidth * oheight + ti * owidth * oheight + i * owidth + j;
                            T *indzp = indz_p + k * otime * owidth * oheight + ti * owidth * oheight + i * owidth + j;

                            /* compute local max: */
                            T maxval = -MAX_FLOAT;
                            int x,y,z;
                            int mx, my, mz;
                            mx = my = mz = -1;

                            for (z = 0; z < kernel_t; z++)
                            {
                                for (y = 0; y < kernel_h; y++)
                                {
                                    for (x = 0; x < kernel_w; x++)
                                    {
                                        if ((start_t + z * dilationT < itime) && (start_h + y * dilationH < iheight) && (start_w + x * dilationW < iwidth))
                                        {
                                            T val = *(ip + z * dilationT * iwidth * iheight + y * dilationH * iwidth + x * dilationW);
                                            if (val > maxval)
                                            {
                                                maxval = val;
                                                // Store indices w.r.t the kernel dimension
                                                mz = z + (kT - kernel_t);
                                                my = y + (kH - kernel_h);
                                                mx = x + (kW - kernel_w);
                                            }
                                        }
                                    }
                                }
                            }

                            // set max values
                            ((unsigned char*)(indzp))[0] = mz;
                            ((unsigned char*)(indzp))[1] = my;
                            ((unsigned char*)(indzp))[2] = mx;
                            ((unsigned char*)(indzp))[3] = 0;

                            /* set output to local max */
                            *op = maxval;
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::validXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc) {
            Nd4jIndex tot = (it - kt) / st + 1;
            Nd4jIndex tor = (ir - kr) / sr + 1;
            Nd4jIndex toc = (ic - kc) / sc + 1;

            Nd4jIndex zz, xx, yy;

            for (zz = 0; zz < tot; zz++) {
                for(yy = 0; yy < tor; yy++) {
                    for(xx = 0; xx < toc; xx++) {
                        /* Dot product in two dimensions... (between input image and the mask) */
                        T *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
                        T *pw_ = k_;
                        T sum = 0;
                        Nd4jIndex kz, kx, ky;
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                for(kx = 0; kx < kc; kx++) {
                                    sum += pi_[kx]*pw_[kx];
                                }
                                pi_ += ic; /* next input line */
                                pw_ += kc; /* next mask line */
                            }
                            pi_ += (ir-kr)*ic; /* next input slice */
                        }
                        /* Update output */
                        *r_++ += sum*alpha;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::validConv3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc) {
            Nd4jIndex tot = (it - kt) / st + 1;
            Nd4jIndex tor = (ir - kr) / sr + 1;
            Nd4jIndex toc = (ic - kc) / sc + 1;

            Nd4jIndex zz, xx, yy;

            for(zz = 0; zz < tot; zz++) {
                for(yy = 0; yy < tor; yy++) {
                    for(xx = 0; xx < toc; xx++) {
                        /* Dot product in two dimensions... (between input image and the mask) */
                        T *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
                        T *pw_ = k_ + kt*kr*kc - 1;
                        T sum = 0;
                        Nd4jIndex kz, kx, ky;
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                for(kx = 0; kx < kc; kx++) {
                                    sum += pi_[kx]*pw_[-kx];
                                }
                                pi_ += ic; /* next input line */
                                pw_ -= kc; /* next mask line */
                            }
                            pi_ += (ir-kr)*ic; /* next input slice */
                        }
                        /* Update output */
                        *r_++ += alpha*sum;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::fullConv3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc) {
            Nd4jIndex tor = (ir - 1) * sr + kr;
            Nd4jIndex toc = (ic - 1) * sc + kc;

            Nd4jIndex zz, xx, yy;

            for(zz = 0; zz < it; zz++) {
                for(yy = 0; yy < ir; yy++) {
                    for(xx = 0; xx < ic; xx++) {
                        /* Outer product in two dimensions... (between input image and the mask) */
                        T *po_ = r_ + zz*st*tor*toc + yy*sr*toc + xx*sc;
                        T *pw_ = k_;
                        Nd4jIndex kz, kx, ky;
                        /* printf("Output Plane : %ld,%ld,%ld, input val=%g\n",zz,yy,xx,*t_); */
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                T z = *t_ * alpha;
                                for(kx = 0; kx < kc; kx++) {
                                    /* printf("o=%g,k=%g," , po_[kx],pw_[kx]); */
                                    po_[kx] += z * pw_[kx];
                                    /* printf("o=%g " , po_[kx]); */
                                }
                                /* printf("\n"); */
                                po_ += toc; /* next input line */
                                pw_ += kc; /* next mask line */
                            }
                            po_ += (tor-kr)*toc; /* next output slice */
                            /* printf("\n"); */
                        }
                        t_++;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::fullXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc) {
            Nd4jIndex tor = (ir - 1) * sr + kr;
            Nd4jIndex toc = (ic - 1) * sc + kc;

            Nd4jIndex zz, xx, yy;

            for(zz = 0; zz < it; zz++) {
                for(yy = 0; yy < ir; yy++) {
                    for(xx = 0; xx < ic; xx++) {
                        /* Outer product in two dimensions... (between input image and the mask) */
                        T *po_ = r_ + zz * st * tor * toc + yy*sr*toc + xx*sc;
                        T *pw_ = k_ + kt*kr*kc -1;
                        Nd4jIndex kz, kx, ky;
                        for(kz = 0; kz < kt; kz++) {
                            for(ky = 0; ky < kr; ky++) {
                                T z = *t_ * alpha;
                                for(kx = 0; kx < kc; kx++) {
                                    po_[kx] += z * pw_[-kx];
                                }
                                po_ += toc; /* next input line */
                                pw_ -= kc; /* next mask line */
                            }
                            po_ += (tor-kr)*toc; /* next output slice */
                        }
                        t_++;
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jIndex ConvolutionUtils<T>::convsize(Nd4jIndex x, Nd4jIndex k, Nd4jIndex s, const char* vf) {
            if (*vf == 'V')
                return (x-k)/s + 1;
            else
                return (x-1)*s + k;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jStatus ConvolutionUtils<T>::conv3Dmv(NDArray<T>* r_, T beta, T alpha, NDArray<T>* t_, NDArray<T>* k_,
                                       Nd4jIndex sdepth, Nd4jIndex srow, Nd4jIndex scol, const char *vf, const char *xc) {

            Nd4jIndex nInputPlane, nInputDepth, nInputRows, nInputCols;
            Nd4jIndex nKernelDepth, nKernelRows, nKernelCols;
            Nd4jIndex nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
            Nd4jIndex istride0, kstride0, kstride1;
            NDArray<T> *input;
            NDArray<T> *kernel;
            T* input_data;
            T* weight_data;
            T* output_data;
            Nd4jIndex nelem;
            Nd4jIndex k, i;

            if (t_->rankOf() != 4)
                throw "Boom";
            //return ND4J_STATUS_BAD_DIMENSIONS;

            if (k_->rankOf() != 5)
                throw "Boom";
            //return ND4J_STATUS_BAD_DIMENSIONS;

            if (sdepth < 1 || srow < 1 || scol < 1)
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            if (!(*vf == 'V' || *vf == 'F'))
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            if (!(*xc == 'X' || *xc == 'C'))
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            bool kD = false;
            input = t_->isContiguous() ? t_ : t_->dup(t_->ordering());
            if (!(k_->stridesOf()[4] == 1 || k_->stridesOf()[3] == k_->sizeAt(4))) {
                kernel = k_->isContiguous() ? k_ : k_->dup(k_->ordering());
                kD = true;
            } else {
                kernel = k_;
            }


            nInputPlane = input->sizeAt(0);
            istride0    = input->stridesOf()[0];
            nInputDepth = input->sizeAt(1);
            nInputRows  = input->sizeAt(2);
            nInputCols  = input->sizeAt(3);

            kstride0    = kernel->stridesOf()[0];
            kstride1    = kernel->stridesOf()[1];
            nKernelDepth = kernel->sizeAt(2);
            nKernelRows = kernel->sizeAt(3);
            nKernelCols = kernel->sizeAt(4);
            nOutputPlane = kernel->sizeAt(0);

            if (kernel->sizeAt(1) != nInputPlane)
                throw "Boom";
            //return ND4J_STATUS_BAD_DIMENSIONS;


            if (!((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F'))
                throw "Boom";
            //return ND4J_STATUS_BAD_PARAMS;

            nOutputDepth = convsize(nInputDepth, nKernelDepth, sdepth, vf);
            nOutputRows = convsize(nInputRows, nKernelRows, srow, vf);
            nOutputCols = convsize(nInputCols, nKernelCols, scol, vf);

            nelem = r_->lengthOf();

            if (r_->sizeAt(0) != nOutputPlane || r_->sizeAt(1) != nOutputDepth || r_->sizeAt(2) != nOutputRows || r_->sizeAt(3)!= nOutputCols) {
                nd4j_printf("Failed at r_ size: {%i, %i, %i, %i} vs {}", r_->sizeAt(0), r_->sizeAt(1), r_->sizeAt(2), r_->sizeAt(3), nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);
                throw "Boom";
                //return ND4J_STATUS_BAD_DIMENSIONS;
            }

            if (nelem == 0 || beta == (T) 0.0f || nelem != r_->lengthOf()) {
                r_->assign((T) 0.0f);
            }
            else if (beta != (T) 1.0f) // stupid comparison
                r_->template applyScalar<simdOps::Multiply<T>>(beta);


            input_data = input->getBuffer();
            weight_data = kernel->getBuffer();
            output_data = r_->getBuffer();

            for(k = 0; k < nOutputPlane; k++) {
                for(i = 0; i < nInputPlane; i++) {
                    /* get kernel */
                    T* ptr_weight = weight_data + k*kstride0 + i*kstride1;
                    /* get input */
                    T* ptr_input = input_data + i*istride0;

                    /* do image, kernel convolution */
                    ConvolutionUtils<T>::conv3D(output_data,
                           alpha,
                           ptr_input,  nInputDepth, nInputRows,  nInputCols,
                           ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                           sdepth, srow, scol, vf, xc);
                }
                /* Next output plane */
                output_data += nOutputDepth*nOutputCols*nOutputRows;
            }

            if (kD)
                delete kernel;

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jStatus ConvolutionUtils<T>::conv3D(T* output_data,
                                     T alpha,
                                     T* ptr_input, Nd4jIndex nInputDepth, Nd4jIndex nInputRows, Nd4jIndex nInputCols,
                                     T* ptr_weight, Nd4jIndex nKernelDepth, Nd4jIndex nKernelRows, Nd4jIndex nKernelCols,
                                     Nd4jIndex sdepth, Nd4jIndex srow, Nd4jIndex scol,
                                     const char *vf, const char *xc) {

            if (!(*vf == 'V' || *vf == 'F'))
                return ND4J_STATUS_BAD_PARAMS;

            if (!(*xc == 'X' || *xc == 'C'))
                return ND4J_STATUS_BAD_PARAMS;


            if (*vf == 'F')
                if (*xc == 'X') {
                    ConvolutionUtils<T>::fullXCorr3Dptr(output_data,
                                                 alpha,
                                                 ptr_input, nInputDepth, nInputRows,  nInputCols,
                                                 ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                 sdepth, srow, scol);
                } else {
                    ConvolutionUtils<T>::fullConv3Dptr(output_data,
                                                alpha,
                                                ptr_input, nInputDepth, nInputRows,  nInputCols,
                                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                sdepth, srow, scol);
                }
            else
            if (*xc == 'X') {
                ConvolutionUtils<T>::validXCorr3Dptr(output_data,
                                              alpha,
                                              ptr_input, nInputDepth, nInputRows,  nInputCols,
                                              ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                              sdepth, srow, scol);
            } else {
                ConvolutionUtils<T>::validConv3Dptr(output_data,
                                             alpha,
                                             ptr_input, nInputDepth, nInputRows,  nInputCols,
                                             ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                             sdepth, srow, scol);
            }

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
// calculation of output height and width in 2D pooling procedure
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode) {
            if(isSameMode > 0) {
                oH = (int) nd4j::math::nd4j_ceil(iH * 1.f / sH);
                oW = (int) nd4j::math::nd4j_ceil(iW * 1.f / sW);
            }
            else {
                oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
                oW = (iW - (kW + (kW-1)*(dW-1)) + 2*pW)/sW + 1;
            }
        }

//////////////////////////////////////////////////////////////////////////
// calculation of output depth, height and width in conv3d procedure        
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int paddingMode) {

            if(paddingMode) {                                           // valid
                
                oD = (iD - (kD + (kD - 1) * (dD - 1)) + 2 * pD) / sD + 1;
                oH = (iH - (kH + (kH - 1) * (dH - 1)) + 2 * pH) / sH + 1;
                oW = (iW - (kW + (kW - 1) * (dW - 1)) + 2 * pW) / sW + 1;
            }
            else {                                                      // same
                
                oD = (int) nd4j::math::nd4j_ceil(iD * 1.f / sD);
                oH = (int) nd4j::math::nd4j_ceil(iH * 1.f / sH);
                oW = (int) nd4j::math::nd4j_ceil(iW * 1.f / sW);
            }
        }





        template class ND4J_EXPORT ConvolutionUtils<float>;
        template class ND4J_EXPORT ConvolutionUtils<float16>;
        template class ND4J_EXPORT ConvolutionUtils<double>;
    
}
}