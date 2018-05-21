//
// @author raver119@gmail.com, created on 07.10.2017.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/generic/helpers/convolutions.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcPadding2D(int& pH, int& pW, int oH, int oW, int iH, int iW, int kH, int kW, int sH, int sW, int dH, int dW) {
            int eKH, eKW;

            if (dH == 1 && dW == 1) {
                eKH = kH;
                eKW = kW;
            } else {
                eKH = kH + (kH - 1) * (dH - 1);
                eKW = kW + (kW - 1) * (dW - 1);
            }

            pH = ((oH - 1) * sH + eKH - iH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
            pW = ((oW - 1) * sW + eKW - iW) / 2;
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
        void ConvolutionUtils<T>::col2vol2(NDArray<T>& col, NDArray<T>& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
            
            const T* colBuff = col.getBuffer();            
            T* volBuff       = vol.getBuffer();            

            Nd4jLong *colShapeOnly = shape::shapeOf(col.getShapeInfo());
            Nd4jLong *colStrides   = shape::stride(col.getShapeInfo());
            Nd4jLong *volShapeOnly = shape::shapeOf(vol.getShapeInfo());
            Nd4jLong volOrder     = shape::order(vol.getShapeInfo());
            Nd4jLong *volStrides   = shape::stride(vol.getShapeInfo());

            Nd4jLong strideBS   = colStrides[0];
            Nd4jLong strideColC = colStrides[1];
            Nd4jLong strideKD   = colStrides[2];
            Nd4jLong strideKH   = colStrides[3];
            Nd4jLong strideKW   = colStrides[4];
            Nd4jLong strideColD = colStrides[5];
            Nd4jLong strideColH = colStrides[6];
            Nd4jLong strideColW = colStrides[7];

            int bS   = volShapeOnly[0];
            int volC = volShapeOnly[1];
            int volD = volShapeOnly[2];
            int volH = volShapeOnly[3];
            int volW = volShapeOnly[4];

            int kD   = colShapeOnly[2];
            int kH   = colShapeOnly[3];
            int kW   = colShapeOnly[4];            
            int colD = colShapeOnly[5];
            int colH = colShapeOnly[6];
            int colW = colShapeOnly[7];            

            //Effective kernel size, accounting for dilation
            int effKD = kD + (kD - 1) * (dD - 1);
            int effKH = kH + (kH - 1) * (dH - 1);
            int effKW = kW + (kW - 1) * (dW - 1);

            Nd4jLong n = bS * volC * volD * volH * volW;                        

#pragma omp parallel for schedule(guided) proc_bind(close)
            for (Nd4jLong i = 0; i < n; i++) {
                
                T val = 0;
                Nd4jLong w_vol = i % volW + pW;
                Nd4jLong h_vol = (i / volW) % volH + pH;
                Nd4jLong d_vol = (i / volW / volH) % volD + pD;
                Nd4jLong c_vol = i / volW / volH / volD;

                Nd4jLong num_vol   = c_vol / volC;
                Nd4jLong depth_vol = c_vol % volC;

                // compute the start and end of the output
                Nd4jLong w_col_start = (w_vol < effKW) ? 0 : (w_vol - effKW) / sW + 1;
                Nd4jLong w_col_end = nd4j::math::nd4j_min<int>(w_vol / sW + 1, colW);

                Nd4jLong h_col_start = (h_vol < effKH) ? 0 : (h_vol - effKH) / sH + 1;
                Nd4jLong h_col_end = nd4j::math::nd4j_min<int>(h_vol / sH + 1, colH);

                Nd4jLong d_col_start = (d_vol < effKD) ? 0 : (d_vol - effKD) / sD + 1;
                Nd4jLong d_col_end = nd4j::math::nd4j_min<int>(d_vol / sD + 1, colD);

                //Iterate over col entries in the 6d array... these are added up
                for (int d_col = d_col_start; d_col < d_col_end; ++d_col) {
                    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
                        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {                  

                            Nd4jLong d_k = (d_vol - d_col * sD);
                            Nd4jLong h_k = (h_vol - h_col * sH);
                            Nd4jLong w_k = (w_vol - w_col * sW);
                            
                            if(d_k % dD == 0 && h_k % dH == 0 && w_k % dW == 0) {
                                   
                                   d_k /= dD;
                                   h_k /= dH;
                                   w_k /= dW;
                                   val += colBuff[num_vol * strideBS + depth_vol * strideColC + d_k * strideKD + h_k * strideKH + w_k * strideKW + d_col * strideColD + h_col * strideColH + w_col * strideColW];
                             }
                        }
                    }
                }
                Nd4jLong i_f = 0;
                Nd4jLong i_c = i;
                for (int dim = 4; dim >= 0; --dim)
                {
                    i_f += (i_c % volShapeOnly[dim])  * volStrides[dim];
                    i_c = i_c / volShapeOnly[dim];
                }
                volBuff[i_f] += val;
            }

        }


//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::avgPool3DBP(NDArray<T>& gradO, NDArray<T>& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad) {
    
    T* pO = gradO.getBuffer();
    T* pI = gradI.getBuffer();

    const int bS = gradI.sizeAt(0);
    const int iC = gradI.sizeAt(1);
    const int iD = gradI.sizeAt(2);
    const int iH = gradI.sizeAt(3);
    const int iW = gradI.sizeAt(4);

    const int oD = gradO.sizeAt(2);
    const int oH = gradO.sizeAt(3);
    const int oW = gradO.sizeAt(4);        

    const int iStride1 = iD * iH * iW;
    const int oStride1 = oD * oH * oW;
    const int iStride0 = iC * iStride1;
    const int oStride0 = iC * oStride1;
    const int size0 = bS * iC;
        
#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)        
    for (int s = 0; s < bS; ++s) {
        for (int k = 0; k < iC; ++k) {

            /* local pointers */
            T *ip = pI + s*iStride0 + k*iStride1;
            T *op = pO + s*oStride0 + k*oStride1;
            
#pragma omp parallel for simd                
            for (int i = 0; i < iStride1; i++)
                *(ip + i) = 0;

#pragma omp parallel for if(oStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)
            /* loop over output */
            for (int ti = 0; ti < oD; ti++) {
                for (int i = 0; i < oH; i++) {
                    for (int j = 0; j < oW; j++) {
                            
                        int cstart = ti * sD - pD;
                        int hstart = i  * sH - pH;
                        int wstart = j  * sW - pW;
                        int cend = nd4j::math::nd4j_min<int>(cstart + kD, iD + pD);
                        int hend = nd4j::math::nd4j_min<int>(hstart + kH, iH + pH);
                        int wend = nd4j::math::nd4j_min<int>(wstart + kW, iW + pW);
                        int pool_size = (cend -cstart) * (hend - hstart) * (wend - wstart);
                        cstart = nd4j::math::nd4j_max<int>(cstart, 0);
                        hstart = nd4j::math::nd4j_max<int>(hstart, 0);
                        wstart = nd4j::math::nd4j_max<int>(wstart, 0);
                        cend = nd4j::math::nd4j_min<int>(cend, iD);
                        hend = nd4j::math::nd4j_min<int>(hend, iH);
                        wend = nd4j::math::nd4j_min<int>(wend, iW);

                        int divide_factor;
                        if (count_include_pad)
                            divide_factor = pool_size;
                        else
                            divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                        /* scatter gradients out to footprint: */
                        T val  = *op++;
                        
                        for (int z = cstart; z < cend; z++)
                            for (int y = hstart; y < hend; y++)
                                for (int x = wstart; x < wend; x++)
                                    *(ip + z * iH * iW + y * iW + x) += val / divide_factor;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::avgPool3D(NDArray<T>& input, NDArray<T>& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad) {

    T* in  = input.getBuffer();
    T* out = output.getBuffer();
    
    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);

    const int oD = output.sizeAt(2);
    const int oH = output.sizeAt(3);
    const int oW = output.sizeAt(4);    

    const int inStride1  = iD * iH * iW;
    const int outStride1 = oD * oH * oW;
    const int inStride0  = iC * inStride1;
    const int outStride0 = iC * outStride1;
    const int size0 = bS * iC;
        
#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
    for(int s = 0; s < bS; ++s)  {            
        for (int k = 0; k < iC; k++) {
                
            /* local pointers. */
            T *ip = in  + s*inStride0  + k*inStride1;
            T *op = out + s*outStride0 + k*outStride1;
#pragma omp parallel for simd
            for (int i = 0; i < outStride1; ++i)
                *(op + i) = 0.;

            /* loop over output */
#pragma omp parallel for if(outStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)
            for (int ti = 0; ti < oD; ti++) {
                for (int i = 0; i < oH; i++) {
                    for (int j = 0; j < oW; j++) {

                        /* compute pool range. */
                        int cstart = ti * sD - pD;
                        int hstart = i  * sH - pH;
                        int wstart = j  * sW - pW;
                        int cend = nd4j::math::nd4j_min<int>(cstart + kD, iD + pD);
                        int hend = nd4j::math::nd4j_min<int>(hstart + kH, iH + pH);
                        int wend = nd4j::math::nd4j_min<int>(wstart + kW, iW + pW);
                        int pool_size = (cend - cstart) * (hend - hstart) * (wend - wstart);
                        cstart = nd4j::math::nd4j_max<int>(cstart, 0);
                        hstart = nd4j::math::nd4j_max<int>(hstart, 0);
                        wstart = nd4j::math::nd4j_max<int>(wstart, 0);
                        cend = nd4j::math::nd4j_min<int>(cend, iD);
                        hend = nd4j::math::nd4j_min<int>(hend, iH);
                        wend = nd4j::math::nd4j_min<int>(wend, iW);

                        int divide_factor;
                        if (count_include_pad)
                            divide_factor = pool_size;
                        else
                            divide_factor = (cend - cstart) * (hend - hstart) * (wend - wstart);

                        /* compute local sum: */
                        T sum = 0.;

                        for (int z = cstart; z < cend; z++) 
                            for (int y = hstart; y < hend; y++) 
                                for (int x = wstart; x < wend; x++) 
                                    sum +=  *(ip + z * iW * iH + y * iW + x);

                        /* set output to local max */
                        *op++ += sum / divide_factor;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::_dilatedMaxPool3D_bp(T *gradInput_p, T *gradOutput_p, T *indBuff, Nd4jLong nslices, Nd4jLong  itime, Nd4jLong  iwidth, Nd4jLong  iheight, Nd4jLong otime, Nd4jLong owidth, Nd4jLong oheight, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH) {
            for (int k = 0; k < nslices; k++)
            {
                T *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
                T *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
                T *indz_p_k = indBuff + k * otime * owidth * oheight;

                /* calculate max points */
                long ti, i, j;
                for (ti = 0; ti < otime; ti++)
                {
                    for (i = 0; i < oheight; i++)
                    {
                        for (j = 0; j < owidth; j++)
                        {
                            /* retrieve position of max */
                            T * indP = &indz_p_k[ti * oheight * owidth + i * owidth + j];
                            long maxti = ((unsigned char*)(indP))[0] * dilationT + ti * dT - pT;
                            long maxi  = ((unsigned char*)(indP))[1] * dilationH + i * dH - pH;
                            long maxj  = ((unsigned char*)(indP))[2] * dilationW + j * dW - pW;

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
        void ConvolutionUtils<T>::_dilatedMaxPool3D(T *input_p, T *output_p, T *indBuff, Nd4jLong nslices, Nd4jLong itime, Nd4jLong iwidth, Nd4jLong iheight, Nd4jLong otime, Nd4jLong owidth, Nd4jLong oheight, int kD, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH) {
            Nd4jLong k;
#pragma omp parallel for private(k)
            for (k = 0; k < nslices; k++)
            {
                /* loop over output */
                Nd4jLong i, j, ti;
                for (ti = 0; ti < otime; ti++)
                {
                    for (i = 0; i < oheight; i++)
                    {
                        for (j = 0; j < owidth; j++)
                        {
                            /* local pointers */

                            Nd4jLong start_t = ti * dT - pT;
                            Nd4jLong start_h = i * dH - pH;
                            Nd4jLong start_w = j * dW - pW;

                            Nd4jLong kernel_d = nd4j::math::nd4j_min<Nd4jLong>(kD, kD + start_t);
                            Nd4jLong kernel_h = nd4j::math::nd4j_min<Nd4jLong>(kH, kH + start_h);
                            Nd4jLong kernel_w = nd4j::math::nd4j_min<Nd4jLong>(kW, kW + start_w);

                            while(start_t < 0)
                                start_t += dilationT;
                            while(start_h < 0)
                                start_h += dilationH;
                            while(start_w < 0)
                                start_w += dilationW;

                            T *ip = input_p + k * itime * iwidth * iheight + start_t * iwidth * iheight + start_h * iwidth + start_w;
                            T *op = output_p + k * otime * owidth * oheight + ti * owidth * oheight + i * owidth + j;
                            T *indP = indBuff + k * otime * owidth * oheight + ti * owidth * oheight + i * owidth + j;

                            /* compute local max: */
                            T maxval = -MAX_FLOAT;
                            int x,y,z;
                            int mx, my, mz;
                            mx = my = mz = -1;

                            for (z = 0; z < kernel_d; z++)
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
                                                mz = z + (kD - kernel_d);
                                                my = y + (kH - kernel_h);
                                                mx = x + (kW - kernel_w);
                                            }
                                        }
                                    }
                                }
                            }

                            // set max values
                            ((unsigned char*)(indP))[0] = mz;
                            ((unsigned char*)(indP))[1] = my;
                            ((unsigned char*)(indP))[2] = mx;
                            ((unsigned char*)(indP))[3] = 0;

                            /* set output to local max */
                            *op = maxval;
                        }
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::validXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tot = (it - kt) / st + 1;
            Nd4jLong tor = (ir - kr) / sr + 1;
            Nd4jLong toc = (ic - kc) / sc + 1;

            Nd4jLong zz, xx, yy;

            for (zz = 0; zz < tot; zz++) {
                for(yy = 0; yy < tor; yy++) {
                    for(xx = 0; xx < toc; xx++) {
                        /* Dot product in two dimensions... (between input image and the mask) */
                        T *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
                        T *pw_ = k_;
                        T sum = 0;
                        Nd4jLong kz, kx, ky;
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
        void ConvolutionUtils<T>::validConv3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tot = (it - kt) / st + 1;
            Nd4jLong tor = (ir - kr) / sr + 1;
            Nd4jLong toc = (ic - kc) / sc + 1;

            Nd4jLong zz, xx, yy;

            for(zz = 0; zz < tot; zz++) {
                for(yy = 0; yy < tor; yy++) {
                    for(xx = 0; xx < toc; xx++) {
                        /* Dot product in two dimensions... (between input image and the mask) */
                        T *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
                        T *pw_ = k_ + kt*kr*kc - 1;
                        T sum = 0;
                        Nd4jLong kz, kx, ky;
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
        void ConvolutionUtils<T>::fullConv3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tor = (ir - 1) * sr + kr;
            Nd4jLong toc = (ic - 1) * sc + kc;

            Nd4jLong zz, xx, yy;

            for(zz = 0; zz < it; zz++) {
                for(yy = 0; yy < ir; yy++) {
                    for(xx = 0; xx < ic; xx++) {
                        /* Outer product in two dimensions... (between input image and the mask) */
                        T *po_ = r_ + zz*st*tor*toc + yy*sr*toc + xx*sc;
                        T *pw_ = k_;
                        Nd4jLong kz, kx, ky;
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
        void ConvolutionUtils<T>::fullXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jLong it, Nd4jLong ir, Nd4jLong ic, T *k_, Nd4jLong kt, Nd4jLong kr, Nd4jLong kc, Nd4jLong st, Nd4jLong sr, Nd4jLong sc) {
            Nd4jLong tor = (ir - 1) * sr + kr;
            Nd4jLong toc = (ic - 1) * sc + kc;

            Nd4jLong zz, xx, yy;

            for(zz = 0; zz < it; zz++) {
                for(yy = 0; yy < ir; yy++) {
                    for(xx = 0; xx < ic; xx++) {
                        /* Outer product in two dimensions... (between input image and the mask) */
                        T *po_ = r_ + zz * st * tor * toc + yy*sr*toc + xx*sc;
                        T *pw_ = k_ + kt*kr*kc -1;
                        Nd4jLong kz, kx, ky;
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
        Nd4jLong ConvolutionUtils<T>::convsize(Nd4jLong x, Nd4jLong k, Nd4jLong s, const char* vf) {
            if (*vf == 'V')
                return (x-k)/s + 1;
            else
                return (x-1)*s + k;
        }

//////////////////////////////////////////////////////////////////////////
        template<typename T>
        Nd4jStatus ConvolutionUtils<T>::conv3Dmv(NDArray<T>* r_, T beta, T alpha, NDArray<T>* t_, NDArray<T>* k_,
                                       Nd4jLong sdepth, Nd4jLong srow, Nd4jLong scol, const char *vf, const char *xc) {

            Nd4jLong nInputPlane, nInputDepth, nInputRows, nInputCols;
            Nd4jLong nKernelDepth, nKernelRows, nKernelCols;
            Nd4jLong nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
            Nd4jLong istride0, kstride0, kstride1;
            NDArray<T> *input;
            NDArray<T> *kernel;
            T* input_data;
            T* weight_data;
            T* output_data;
            Nd4jLong nelem;
            Nd4jLong k, i;

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
                    T* pIn = input_data + i*istride0;

                    /* do image, kernel convolution */
                    ConvolutionUtils<T>::conv3D(output_data,
                           alpha,
                           pIn,  nInputDepth, nInputRows,  nInputCols,
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
                                     T* pIn, Nd4jLong nInputDepth, Nd4jLong nInputRows, Nd4jLong nInputCols,
                                     T* ptr_weight, Nd4jLong nKernelDepth, Nd4jLong nKernelRows, Nd4jLong nKernelCols,
                                     Nd4jLong sdepth, Nd4jLong srow, Nd4jLong scol,
                                     const char *vf, const char *xc) {

            if (!(*vf == 'V' || *vf == 'F'))
                return ND4J_STATUS_BAD_PARAMS;

            if (!(*xc == 'X' || *xc == 'C'))
                return ND4J_STATUS_BAD_PARAMS;


            if (*vf == 'F')
                if (*xc == 'X') {
                    ConvolutionUtils<T>::fullXCorr3Dptr(output_data,
                                                 alpha,
                                                 pIn, nInputDepth, nInputRows,  nInputCols,
                                                 ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                 sdepth, srow, scol);
                } else {
                    ConvolutionUtils<T>::fullConv3Dptr(output_data,
                                                alpha,
                                                pIn, nInputDepth, nInputRows,  nInputCols,
                                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                                sdepth, srow, scol);
                }
            else
            if (*xc == 'X') {
                ConvolutionUtils<T>::validXCorr3Dptr(output_data,
                                              alpha,
                                              pIn, nInputDepth, nInputRows,  nInputCols,
                                              ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                              sdepth, srow, scol);
            } else {
                ConvolutionUtils<T>::validConv3Dptr(output_data,
                                             alpha,
                                             pIn, nInputDepth, nInputRows,  nInputCols,
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
        void ConvolutionUtils<T>::calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode) {

            if(!isSameMode) {                                           // valid
                
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

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::maxPool3d(NDArray<T>& input, NDArray<T>& output,  const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
        
    T* inBuff = input.getBuffer();
    T* outBuff = output.getBuffer();

    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);

    const int oD = output.sizeAt(2);
    const int oH = output.sizeAt(3);
    const int oW = output.sizeAt(4);    
    
    const int iStride2 = iH * iW;
    const int oStride2 = oH * oW;
    const int iStride1 = iD * iStride2;
    const int oStride1 = oD * oStride2;
    const int iStride0 = iC * iStride1;
    const int oStride0 = iC * oStride1;
    const int size0 = bS * iC;
        
#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
    for (int s = 0; s < bS; ++s) {
        for (int k = 0; k < iC; ++k) {

            /* local pointers. */
            T *in  = inBuff  + s*iStride0 + k*iStride1;
            T *out = outBuff + s*oStride0 + k*oStride1;

#pragma omp parallel for if(oStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)
            /* loop over output */
            for (int ti = 0; ti < oD; ti++) {
                for (int i = 0; i < oH; i++) {
                    for (int j = 0; j < oW; j++){
          
                        /* local pointers */
                        int start_d = ti * sD - pD;
                        int start_w = j  * sW - pW;
                        int start_h = i  * sH - pH;
                    
                        int kernel_d = math::nd4j_min<int>(kD, kD + start_d);
                        int kernel_h = math::nd4j_min<int>(kH, kH + start_h);
                        int kernel_w = math::nd4j_min<int>(kW, kW + start_w);

                        while(start_d < 0)
                            start_d += dD;
                        while(start_h < 0)
                            start_h += dH;
                        while(start_w < 0)
                            start_w += dW;

                        T* ip = in + start_d * iStride2 + start_h * iW + start_w;
                        T* op = out + ti * oStride2 + i * oW + j;          

                        // compute local max
                        T maxval = - DataTypeUtils::max<T>();                        

                        for (int z = 0; z < kernel_d; z++) 
                            for (int y = 0; y < kernel_h; y++) 
                                for (int x = 0; x < kernel_w; x++) 
                                    if ((start_d + z * dD < iD) && (start_h + y * dH < iH) && (start_w + x * dW < iW)) {                                    
                                        T val = *(ip + z * dD * iW * iH + y * dH * iW + x * dW);
                                        if (val > maxval)                  
                                            maxval = val;                                     
                                    }
                        // set output to local max
                        *op = maxval;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::maxPool3dIndices(NDArray<T>& input, int* indices, const int oD, const int oH, const int oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    T* inBuff  = input.getBuffer();  

    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);

    const int iStride2 = iH * iW;
    const int oStride2 = oH * oW;
    const int iStride1 = iD * iStride2;
    const int oStride1 = oD * oStride2;
    const int iStride0 = iC * iStride1;
    const int oStride0 = iC * oStride1;
    const int size0 = bS * iC;

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
    for (int s = 0; s < bS; ++s) {
        for (int k = 0; k < iC; ++k) {
            
            T* in    = inBuff  + s*iStride0 + k*iStride1;
            int* ind = indices + s*oStride0 + k*oStride1;

#pragma omp parallel for if(oStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)
            for (int ti = 0; ti < oD; ti++) {
                for (int i = 0; i < oH; i++) {
                    for (int j = 0; j < oW; j++){
          
                        /* local pointers */
                        int start_d = ti * sD - pD;
                        int start_h = i  * sH - pH;
                        int start_w = j  * sW - pW;                    
                    
                        int kernel_d = math::nd4j_min<int>(kD, kD + start_d);
                        int kernel_h = math::nd4j_min<int>(kH, kH + start_h);
                        int kernel_w = math::nd4j_min<int>(kW, kW + start_w);

                        while(start_d < 0)
                            start_d += dD;
                        while(start_h < 0)
                            start_h += dH;
                        while(start_w < 0)
                            start_w += dW;                    

                        T* ip     = in  + start_d * iStride2 + start_h * iW + start_w;                    
                        int* indP = ind + ti * oStride2 + i * oW + j;
                    
                        T maxval = - DataTypeUtils::max<T>();
                        int mx, my, mz;
                        mx = my = mz = -1;

                        for (int z = 0; z < kernel_d; z++)
                            for (int y = 0; y < kernel_h; y++)
                                for (int x = 0; x < kernel_w; x++)
                                    if ((start_d + z * dD < iD) && (start_h + y * dH < iH) && (start_w + x * dW < iW)) {   
                                        T val = *(ip + z * dD * iH * iW + y * dH * iW + x * dW);
                                        if (val > maxval) {                 
                                        
                                            maxval = val;                                  
                                            mz = z + (kD - kernel_d);
                                            my = y + (kH - kernel_h);
                                            mx = x + (kW - kernel_w);
                                        }
                                    }
                        // set max values
                        ((unsigned char*)(indP))[0] = mz;
                        ((unsigned char*)(indP))[1] = my;
                        ((unsigned char*)(indP))[2] = mx;
                        ((unsigned char*)(indP))[3] = 0;                    
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::maxPool3dBP(NDArray<T>& input, const int* indices, NDArray<T>& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    T* inBuff  = input.getBuffer();
    T* outBuff = output.getBuffer();    

    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);
    const int oD = output.sizeAt(2);
    const int oH = output.sizeAt(3);
    const int oW = output.sizeAt(4);  

    const int iStride2 = iH * iW;
    const int oStride2 = oH * oW;
    const int iStride1 = iD * iStride2;
    const int oStride1 = oD * oStride2;
    const int iStride0 = iC * iStride1;
    const int oStride0 = iC * oStride1;
    const int size0 = bS * iC;  
    
#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
    for (int s = 0; s < bS; ++s) {
        for (int k = 0; k < iC; k++) {
        
            T* oP           = outBuff + s*oStride0 + k*oStride1;
            T* iP           = inBuff  + s*iStride0 + k*iStride1;    
            const int* indP = indices + s*iStride0 + k*iStride1;

#pragma omp parallel for if(iStride1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3)            
            for (int ti = 0; ti < iD; ++ti) {
                for (int i = 0; i < iH; ++i)  {
                    for (int j = 0; j < iW; ++j) {
                                        
                        const int* indzP = indP + ti * iStride2 + i * iW + j;

                        int maxti = ((unsigned char*)(indzP))[0] * dD + ti * sD - pD;
                        int maxi  = ((unsigned char*)(indzP))[1] * dH + i  * sH - pH;
                        int maxj  = ((unsigned char*)(indzP))[2] * dW + j  * sW - pW;
                    
                        if (maxti != -1) 
                            oP[maxti * oStride2 + maxi * oW + maxj] += iP[ti * iStride2 + i * iW + j];      
                    }
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////

template<typename T>
void ConvolutionUtils<T>::vol2col2(NDArray<T>& vol, NDArray<T>& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    T* colBuff = col.getBuffer();    

    Nd4jLong *colShape  = shape::shapeOf(col.getShapeInfo());
    char colOrder  = shape::order(col.getShapeInfo());
    Nd4jLong *colStride = shape::stride(col.getShapeInfo());

    Nd4jLong *volShape  = shape::shapeOf(vol.getShapeInfo());
    Nd4jLong *volStride = shape::stride(vol.getShapeInfo());

    int bS   = volShape[0];
    int volC = volShape[1];
    int volD = volShape[2];
    int volH = volShape[3];
    int volW = volShape[4];

    Nd4jLong strideBS   = volStride[0];
    Nd4jLong strideVolC = volStride[1];
    Nd4jLong strideVolD = volStride[2];
    Nd4jLong strideVolH = volStride[3];
    Nd4jLong strideVolW = volStride[4];

    int kD   = colShape[2];
    int kH   = colShape[3];
    int kW   = colShape[4];            
    int colD = colShape[5];
    int colH = colShape[6];
    int colW = colShape[7];

    int kSize = kD * kW * kH;

    Nd4jLong n = bS * volC * colD * colH * colW;

#pragma omp parallel for schedule(guided) proc_bind(close)
    for (Nd4jLong index = 0; index < n; index++) {
                
        Nd4jLong w_col = index % colW;
        Nd4jLong h_col = (index / colW) % colH;
        Nd4jLong d_col = (index / colW / colH) % colD;
    
        Nd4jLong c_vol = index / colW / colH / colD;
        Nd4jLong c_col = c_vol * kSize;
    
        Nd4jLong depth_vol = c_vol % volC;
        Nd4jLong num_vol   = c_vol / volC;
        Nd4jLong d_offset = d_col * sD - pD;
        Nd4jLong h_offset = h_col * sH - pH;
        Nd4jLong w_offset = w_col * sW - pW;

        T* data_col_ptr = col.getBuffer();
        T* data_vol_ptr = vol.getBuffer();

        Nd4jLong i_c = ((c_col * colD + d_col) * colH + h_col) * colW + w_col;
        data_col_ptr += ((c_col * colD + d_col) * colH + h_col) * colW + w_col;
        data_vol_ptr += num_vol * strideBS + depth_vol * strideVolC + d_offset * strideVolD + h_offset * strideVolH + w_offset * strideVolW;

        for (int z = 0; z < kD; ++z) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                            
                    Nd4jLong d_vol = d_offset + z * dD;
                    Nd4jLong h_vol = h_offset + i * dH;
                    Nd4jLong w_vol = w_offset + j * dW;
                            
                    Nd4jLong i_f = 0;
                    Nd4jLong i_c_temp = i_c;
                            
                    for (int dim = 7; dim >= 0; dim--) {
                        i_f += (i_c_temp % colShape[dim])  * colStride[dim];
                        i_c_temp = i_c_temp / colShape[dim];
                    }
                                
                    if (d_vol >= 0 && h_vol >= 0 && w_vol >= 0 && d_vol < volD && h_vol < volH && w_vol < volW)
                        colBuff[i_f] = data_vol_ptr[z * dD * strideVolD + i * dH * strideVolH + j * dW * strideVolW];
                    else 
                        colBuff[i_f] = 0;

                     data_col_ptr += colD * colH * colW;
                     i_c          += colD * colH * colW;
                }
            }
        }
    }
}
 
//////////////////////////////////////////////////////////////////////////
        template<typename T>
        void ConvolutionUtils<T>::calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode) {
            
            if (isSameMode) {
                oH = sH * iH;
                oW = sW * iW;
            } 
            else {
                int ekH, ekW;
                if (dH == 1 && dW == 1) {
                    ekH = kH;
                    ekW = kW;
                } else {
                    ekH = kH + (kH - 1) * (dH - 1);
                    ekW = kW + (kW - 1) * (dW - 1);
                }

                oH = sH * (iH - 1) + ekH - 2 * pH;
                oW = sW * (iW - 1) + ekW - 2 * pW;
            }
        }
       

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indOoH = 1; indWoC = 3; indWiC = 2;
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indOoH = 2; indWoC = 0; indWiC = 1;              
    }    

    bS = inShapeInfo[1];                          // batch size
    iC = inShapeInfo[indIOioC+1];                   // input channels        
    iH = inShapeInfo[indIiH+1];                     // input height
    iW = inShapeInfo[indIiH+2];                   // input width
    oC = outShapeInfo[indIOioC+1];                  // output channels
    oH = outShapeInfo[indOoH+1];                    // output height
    oW = outShapeInfo[indOoH+2];                  // output width    
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::getSizesAndIndexesConv2d(const bool isNCHW, const NDArray<T>& input, const NDArray<T>& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    getSizesAndIndexesConv2d(isNCHW, input.getShapeInfo(), output.getShapeInfo(), bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void ConvolutionUtils<T>::getSizesAndIndexesConv3d(const bool isNCDHW, const NDArray<T>& input, const NDArray<T>& output, int& bS, int& iC, int& iD, int& iH, int& iW, int& oC, int& oD, int& oH, int& oW, int& indIOioC, int& indIOioD, int& indWiC, int& indWoC, int& indWkD) {
    
    // input   [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    // weights [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)    
    // output  [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    if(!isNCDHW) {
        indIOioC = 4; indIOioD = 1; indWkD = 0; indWoC = 4; indWiC = 3; 
    }
    else {        
        indIOioC = 1; indIOioD = 2; indWkD = 2; indWoC = 0; indWiC = 1;
    }    

    bS = input.sizeAt(0);                          // batch size
    iC = input.sizeAt(indIOioC);                   // input channels        
    iD = input.sizeAt(indIOioD);                   // input depth
    iH = input.sizeAt(indIOioD+1);                 // input height
    iW = input.sizeAt(indIOioD+2);                 // input width
    oC = output.sizeAt(indIOioC);                  // output channels    
    oD = output.sizeAt(indIOioD);                  // output depth
    oH = output.sizeAt(indIOioD+1);                // output height
    oW = output.sizeAt(indIOioD+2);                // output width    

}
 
//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::conv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = inArrs[2];                                    // [oC]
    
    // output [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
                                         
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
    
    std::vector<int> weightsAxesForDot = {indWiC, indWkH, indWkH+1};                                                        // iC, kH, kW
    
    std::vector<int> permutForOutput;
    if(!isNCHW)
        input = input->permute({0, 3, 1, 2});                                       // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC
    else
        permutForOutput = {0, indOoH, indOoH+1, indIOioC};                          // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
     
    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        

    //----- calculation of output -----//
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                    // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    NDArrayFactory<T>::tensorDot(&columns, weights, output, {1,2,3}, weightsAxesForDot, permutForOutput); // [bS, iC, kH, kW, oH, oW] x [kH, kW, iC, oC]/[oC, iC, kH, kW] = [bS, oH, oW, oC]

    //----- add biases if required -----//
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCHW)
        delete input;                
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::conv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                        // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                        // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = inArrs[2];                        // [oC]
    NDArray<T> *gradO   = inArrs[3];                        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    
    NDArray<T> *gradI = outArrs[0];                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    NDArray<T> *gradW = outArrs[1];                         // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    NDArray<T> *gradB = outArrs[2];                         // [oC]
                                     
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::vector<int> gradOaxesForDot, permutForGradW, permutForColumns;    

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW        
        permutForGradW   = {2, 0, 1, 3};                                        // [kH, kW, iC, oC] -> [iC, kH, kW, oC]        
        permutForColumns = {2, 3, 1, 0, 4, 5};                                  // [bS, iC, kH, kW, oH, oW] -> [kH, kW, iC, bS, oH, oW]
    }
    else {
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
        permutForGradW   = {1, 2, 3, 0};                                        // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
        permutForColumns = {1, 2, 3, 0, 4, 5};                                  // [bS, iC, kH, kW, oH, oW] -> [iC, kH, kW, bS, oH, oW]
    }
    
    if(isSameMode)                       // SAME        
        calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    // ----- calculation of gradW and gradB ----- // 
    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]        
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, permutForGradW);       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]

    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, gradOaxesForDot);                          // sum over bS, oH, oW
        if(gradB != outArrs[2]) 
            delete gradB;
    }

    //----- calculation of gradI -----//
    nd4j::NDArrayFactory<T>::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, permutForColumns);       // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]/[iC, kH, kW, bS, oH, oW]
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    columns.template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());                            // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
  
    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::depthwiseConv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = inArrs[1];                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = inArrs[2];                                    // [oC] = iC*mC
    
    // output is [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)        
                                     
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 0-NCHW,  1-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weights->sizeAt(indWmC);                           // channels multiplier
    
    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
    std::vector<std::vector<Nd4jLong>> modifWeights, modifOutput;
    std::vector<Nd4jLong> outReShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifWeights = {{1,2,3,0},{iC,kH*kW,mC}};                                       // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());                
    NDArray<T>* outputReshaped = output->reshape(output->ordering(), outReShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});

    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                                 // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::NDArrayFactory<T>::tensorDot(&columns, weights, outputReshaped, modifColumns, modifWeights, modifOutput);    // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
    
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCHW)
        delete input;                  
    
    delete outputReshaped;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::depthwiseConv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs) {

    NDArray<T> *input   = inArrs[0];                            // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    NDArray<T> *weights = inArrs[1];                            // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *bias    = inArrs[2];                            // [oC] = [iC*mC]
    NDArray<T> *gradO   = inArrs[3];                            // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = outArrs[0];                             // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = outArrs[1];                             // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *gradB = outArrs[2];                             // [oC]        
                                     
    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weights->sizeAt(indWmC);                           // channels multiplier    

    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,2,3,0,4,5}, {iC, kH*kW, bS*oH*oW}};      // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
    std::vector<std::vector<Nd4jLong>> modifGradW, modifGradO1, modifGradO2;
    std::vector<Nd4jLong> gradOreShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradI = gradI->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifGradW = {{2,0,1,3},{iC,kH*kW,mC}};                                         // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifGradW = {{1,2,3,0},{iC,kH*kW,mC}};                                         // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        
    NDArray<T>* gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    
    // ----- calculation of gradW and gradB ----- //            
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradOreshaped, gradW, modifColumns, modifGradO1, modifGradW);  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    // ----- calculation of gradB ----- //
    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW
        if(gradB != outArrs[2]) 
            delete gradB;
    }

    //----- calculation of gradI -----//                
    nd4j::NDArrayFactory<T>::tensorDot(weights, gradO, &columns, modifGradW, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]    
    columns.template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());                     // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {        
        delete input;        
        delete gradI;
    }

    delete gradOreshaped;      
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::sconv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs) {

    NDArray<T> *input        = inArrs[0];                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    NDArray<T> *weightsDepth = inArrs[1];                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
    NDArray<T> *weightsPoint = inArrs[2];                                           // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *bias         = inArrs[3];                                           // [oC], oC = iC*mC if weightsPoint=nullptr
    
    // output is [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)

    int kH = intArgs[0];                                                        // filter(kernel) height
    int kW = intArgs[1];                                                        // filter(kernel) width
    int sH = intArgs[2];                                                        // strides height
    int sW = intArgs[3];                                                        // strides width
    int pH = intArgs[4];                                                        // paddings height
    int pW = intArgs[5];                                                        // paddings width
    int dH = intArgs[6];                                                        // dilations height
    int dW = intArgs[7];                                                        // dilations width
    int isSameMode = intArgs[8];                                                // 0-VALID, 1-SAME
    int isNCHW     = intArgs[9];                                                // 1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    NDArray<T>* outputDepth = output;
    if(weightsPoint)                        // if pointwise convolution is expected
        outputDepth = new NDArray<T>(output->ordering(), !isNCHW ? std::vector<Nd4jLong>({bS, oH, oW, iC*mC}) : std::vector<Nd4jLong>({bS, iC*mC, oH, oW}));    

    // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //    
    ConvolutionUtils<T>::depthwiseConv2d({input, weightsDepth, weightsPoint ? nullptr : bias}, outputDepth, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, isNCHW});                                   
    
    // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
    if (weightsPoint) {
        ConvolutionUtils<T>::conv2d({outputDepth, weightsPoint, bias}, output, {1,1, 1,1, 0,0, 1,1, isSameMode, isNCHW});             // in this case oH=iH, oW=iW                
        delete outputDepth;
    }
}


//////////////////////////////////////////////////////////////////////////
// [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
template <typename T>
void ConvolutionUtils<T>::vol2col(NDArray<T>& volume, NDArray<T>& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const int bS = volume.sizeAt(0);
    const int iC = volume.sizeAt(1);
    const int iD = volume.sizeAt(2);
    const int iH = volume.sizeAt(3);
    const int iW = volume.sizeAt(4);
    const int kD = columns.sizeAt(2);
    const int kH = columns.sizeAt(3);
    const int kW = columns.sizeAt(4);
    const int oD = columns.sizeAt(5);
    const int oH = columns.sizeAt(6);
    const int oW = columns.sizeAt(7);
    const Nd4jLong colStride0 = columns.stridesOf()[0];
    const Nd4jLong colStride1 = columns.stridesOf()[1];
    const Nd4jLong colStride2 = columns.stridesOf()[2];
    const Nd4jLong colStride3 = columns.stridesOf()[3];
    const Nd4jLong colStride4 = columns.stridesOf()[4];
    const Nd4jLong colStride5 = columns.stridesOf()[5];
    const Nd4jLong colStride6 = columns.stridesOf()[6];
    const Nd4jLong colStride7 = columns.stridesOf()[7];  
    const Nd4jLong volStride0 = volume.stridesOf()[0];
    const Nd4jLong volStride1 = volume.stridesOf()[1];
    const Nd4jLong volStride2 = volume.stridesOf()[2];
    const Nd4jLong volStride3 = volume.stridesOf()[3];
    const Nd4jLong volStride4 = volume.stridesOf()[4];    
    
    T* vol = volume.getBuffer();
    T* col = columns.getBuffer();

    T* col0, *vol0;
    int volDep, volRow, volCol;

if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) 

#pragma omp parallel for schedule(static) proc_bind(close) private(vol0, col0, volDep, volRow, volCol)
    for (int b = 0; b < bS; b++) {
        for (int c = 0; c < iC; ++c) {        
            for (int kDep = 0; kDep < kD; ++kDep) { 
                for (int kRow = 0; kRow < kH; ++kRow) {                        
                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        for (int colD = 0; colD < oD; ++colD) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {                    
                                
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col0 = col + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;;
                                    vol0 = vol + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                        *col0 = 0.;                                                     
                                    else 
                                        *col0 = *vol0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  

else 

#pragma omp parallel for schedule(static) proc_bind(close) private(vol0, col0, volDep, volRow, volCol)    
    for (int b = 0; b < bS; b++) {
        for (int colD = 0; colD < oD; ++colD) {
            for (int colH = 0; colH < oH; ++colH) {
                for (int colW = 0; colW < oW; ++colW) {
                    for (int c = 0; c < iC; ++c) {
                        for (int kDep = 0; kDep < kD; ++kDep) { 
                            for (int kRow = 0; kRow < kH; ++kRow) {                        
                                for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col0 = col + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;;
                                    vol0 = vol + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                        *col0 = 0.;                                                     
                                    else 
                                        *col0 = *vol0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  
}

//////////////////////////////////////////////////////////////////////////
// [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to [bS, iC, iD, iH, iW]
template <typename T>
void ConvolutionUtils<T>::col2vol(NDArray<T>& columns, NDArray<T>& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const int bS = volume.sizeAt(0);
    const int iC = volume.sizeAt(1);
    const int iD = volume.sizeAt(2);
    const int iH = volume.sizeAt(3);
    const int iW = volume.sizeAt(4);
    const int kD = columns.sizeAt(2);
    const int kH = columns.sizeAt(3);
    const int kW = columns.sizeAt(4);
    const int oD = columns.sizeAt(5);
    const int oH = columns.sizeAt(6);
    const int oW = columns.sizeAt(7);
    const Nd4jLong colStride0 = columns.stridesOf()[0];
    const Nd4jLong colStride1 = columns.stridesOf()[1];
    const Nd4jLong colStride2 = columns.stridesOf()[2];
    const Nd4jLong colStride3 = columns.stridesOf()[3];
    const Nd4jLong colStride4 = columns.stridesOf()[4];
    const Nd4jLong colStride5 = columns.stridesOf()[5];
    const Nd4jLong colStride6 = columns.stridesOf()[6];
    const Nd4jLong colStride7 = columns.stridesOf()[7];  
    const Nd4jLong volStride0 = volume.stridesOf()[0];
    const Nd4jLong volStride1 = volume.stridesOf()[1];
    const Nd4jLong volStride2 = volume.stridesOf()[2];
    const Nd4jLong volStride3 = volume.stridesOf()[3];
    const Nd4jLong volStride4 = volume.stridesOf()[4];    
    
    T* vol = volume.getBuffer();
    T* col = columns.getBuffer();

    T* col0, *vol0;
    int volDep, volRow, volCol;

if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) 

#pragma omp parallel for schedule(static) proc_bind(close) private(vol0, col0, volDep, volRow, volCol)    
    for (int b = 0; b < bS; b++) {
        for (int c = 0; c < iC; ++c) {        
            for (int kDep = 0; kDep < kD; ++kDep) { 
                for (int kRow = 0; kRow < kH; ++kRow) {                        
                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        for (int colD = 0; colD < oD; ++colD) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {                    

                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;

                                    col0 = col + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;;
                                    vol0 = vol + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;

                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW))
                                        if(volDep == -pD &&  volRow == -pH && volCol == -pW)
                                            *vol0 = *col0;
                                        else
                                            *vol0 += *col0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  

else 

#pragma omp parallel for schedule(static) proc_bind(close) private(vol0, col0, volDep, volRow, volCol)    
    for (int b = 0; b < bS; b++) {
        for (int colD = 0; colD < oD; ++colD) {
            for (int colH = 0; colH < oH; ++colH) {
                for (int colW = 0; colW < oW; ++colW) {
                    for (int c = 0; c < iC; ++c) {
                        for (int kDep = 0; kDep < kD; ++kDep) { 
                            for (int kRow = 0; kRow < kH; ++kRow) {                        
                                for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col0 = col + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;;
                                    vol0 = vol + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW))
                                        if(volDep == -pD &&  volRow == -pH && volCol == -pW)
                                            *vol0 = *col0;
                                        else
                                            *vol0 += *col0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling2d(const NDArray<T>& input, NDArray<T>& output, const int factorH, const int factorW, const bool isNCHW) {
    // input  has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC) 
    // output has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    
    int indIn[8]  = {0,0,  0,0,  0,0,  0,0};
    int indOut[8] = {0,0,  0,0,  0,0,  0,0};
    const int dimIH = isNCHW ? 2 : 1;    
    const int j0 = 2*dimIH;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3;
    const int size0 = input.sizeAt(dimIH) * input.sizeAt(dimIH+1);
    // const int size1 = factorH * factorW;

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indIn, indOut) 
    for(int ih = 0; ih < input.sizeAt(dimIH); ++ih) {
        for(int iw = 0; iw < input.sizeAt(dimIH+1); ++iw) {
            indIn[j0] = ih; indIn[j1] = ih+1; 
            indIn[j2] = iw; indIn[j3] = iw+1; 

// #pragma omp parallel for if(size1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indOut) 
            for(int fh = 0; fh < factorH; ++fh) {
                for(int fw = 0; fw < factorW; ++fw) {
                    
                    indOut[j0] = ih * factorH + fh; indOut[j1] = indOut[j0] + 1; 
                    indOut[j2] = iw * factorW + fw; indOut[j3] = indOut[j2] + 1;                     
                    output(indOut).assign(input(indIn));
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling3d(const NDArray<T>& input, NDArray<T>& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
    // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    int indIn[10]  = {0,0,  0,0,  0,0,  0,0,  0,0};
    int indOut[10] = {0,0,  0,0,  0,0,  0,0,  0,0};
    const int dimID = isNCDHW ? 2 : 1;    
    const int j0 = 2*dimID;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3, j4 = j0+4, j5 = j0+5;;
    const int size0 = input.sizeAt(dimID) * input.sizeAt(dimID+1) * input.sizeAt(dimID+2);
    // const int size1 = factorD * factorH * factorW;

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indIn, indOut) 
    for(int id = 0; id < input.sizeAt(dimID); ++id) {
        for(int ih = 0; ih < input.sizeAt(dimID+1); ++ih) {
            for(int iw = 0; iw < input.sizeAt(dimID+2); ++iw) {
                indIn[j0] = id; indIn[j1] = id+1;
                indIn[j2] = ih; indIn[j3] = ih+1;
                indIn[j4] = iw; indIn[j5] = iw+1;

// #pragma omp parallel for if(size1 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indOut) 
            for(int fd = 0; fd < factorD; ++fd) {
                for(int fh = 0; fh < factorH; ++fh) {
                    for(int fw = 0; fw < factorW; ++fw) {
                            indOut[j0] = id * factorD + fd; indOut[j1] = indOut[j0] + 1; 
                            indOut[j2] = ih * factorH + fh; indOut[j3] = indOut[j2] + 1; 
                            indOut[j4] = iw * factorW + fw; indOut[j5] = indOut[j4] + 1;                     
                            output(indOut).assign(input(indIn));
                        }
                    }
                }
            }
        }
    }    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling2dBP(const NDArray<T>& gradO, NDArray<T>& gradI, const bool isNCHW) {
    // gradO has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    // gradI has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)     
    int indIn[8]  = {0,0,  0,0,  0,0,  0,0};
    int indOut[8] = {0,0,  0,0,  0,0,  0,0};
    const int dimIH = isNCHW ? 2 : 1;    
    const int factorH = gradO.sizeAt(dimIH)   / gradI.sizeAt(dimIH);
    const int factorW = gradO.sizeAt(dimIH+1) / gradI.sizeAt(dimIH+1);
    const int j0 = 2*dimIH;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3;
    const int size0 = gradI.sizeAt(dimIH) * gradI.sizeAt(dimIH+1);

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2) firstprivate(indIn, indOut) 
    for(int ih = 0; ih < gradI.sizeAt(dimIH); ++ih) {
        for(int iw = 0; iw < gradI.sizeAt(dimIH+1); ++iw) {
            indIn[j0] = ih; indIn[j1] = ih+1; 
            indIn[j2] = iw; indIn[j3] = iw+1; 
            NDArray<T> subGradI = gradI(indIn);

            for(int fh = 0; fh < factorH; ++fh) {
                for(int fw = 0; fw < factorW; ++fw) {                    
                    indOut[j0] = ih * factorH + fh; indOut[j1] = indOut[j0] + 1; 
                    indOut[j2] = iw * factorW + fw; indOut[j3] = indOut[j2] + 1;                     
                    if(!fh && !fw)
                        subGradI.assign(gradO(indOut));
                    else
                        subGradI += gradO(indOut);
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::upsampling3dBP(const NDArray<T>& gradO, NDArray<T>& gradI, const bool isNCDHW) {
    // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    int indIn[10]  = {0,0,  0,0,  0,0,  0,0,  0,0};
    int indOut[10] = {0,0,  0,0,  0,0,  0,0,  0,0};
    const int dimID = isNCDHW ? 2 : 1;
    const int factorD = gradO.sizeAt(dimID)   / gradI.sizeAt(dimID);
    const int factorH = gradO.sizeAt(dimID+1) / gradI.sizeAt(dimID+1);
    const int factorW = gradO.sizeAt(dimID+2) / gradI.sizeAt(dimID+2);
    const int j0 = 2*dimID;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3, j4 = j0+4, j5 = j0+5;;
    const int size0 = gradI.sizeAt(dimID) * gradI.sizeAt(dimID+1) * gradI.sizeAt(dimID+2);

#pragma omp parallel for if(size0 > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(3) firstprivate(indOut, indIn) 
    for(int id = 0; id < gradI.sizeAt(dimID); ++id) {
        for(int ih = 0; ih < gradI.sizeAt(dimID+1); ++ih) {
            for(int iw = 0; iw < gradI.sizeAt(dimID+2); ++iw) {
                indIn[j0] = id; indIn[j1] = id+1;
                indIn[j2] = ih; indIn[j3] = ih+1;
                indIn[j4] = iw; indIn[j5] = iw+1;
                NDArray<T> subGradI = gradI(indIn);

            for(int fd = 0; fd < factorD; ++fd) {
                for(int fh = 0; fh < factorH; ++fh) {
                    for(int fw = 0; fw < factorW; ++fw) {
                            indOut[j0] = id * factorD + fd; indOut[j1] = indOut[j0] + 1; 
                            indOut[j2] = ih * factorH + fh; indOut[j3] = indOut[j2] + 1; 
                            indOut[j4] = iw * factorW + fw; indOut[j5] = indOut[j4] + 1;                     
                            if(!fd && !fh && !fw)
                                subGradI.assign(gradO(indOut));
                            else
                                subGradI += gradO(indOut);
                        }
                    }
                }
            }
        }
    }    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::maxPool2d(NDArray<T>* input, NDArray<T>* output, const std::vector<int>& params, NDArray<T>* indices) {

    int kH = params[0];
    int kW = params[1];
    int sH = params[2];
    int sW = params[3];
    int pH = params[4];
    int pW = params[5];
    int dH = params[6];
    int dW = params[7];

    const int bS  = input->sizeAt(0);
    const int inD = input->sizeAt(1);
    const int iH = input->sizeAt(2);
    const int iW = input->sizeAt(3);
    const int oH  = output->sizeAt(2);
    const int oW  = output->sizeAt(3);

    const bool isSameMode = params[8];

    if (isSameMode)
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, pH, pW);                    

    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; poolingMode; 9 - divisor;
    std::vector<T> argT = {(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T)dW, 0., 1.};

    input->template applyTransform<simdOps::Pooling2D<T>>(output, argT.data());
    
    if (indices != nullptr) {
        // for max_pool_with_argmax 
        int part = input->lengthOf() / bS;
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)
        for (int b = 0; b < input->lengthOf(); b += part) 
            for (int i = 0; i < part; i++)
                (*indices)(b+i) = i;                
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling3d(NDArray<T>& input, NDArray<T>& output, const T* extraParams) {
    // input is  [bS, iC, iD, iH, iW]
    // output is [bS, iC, oD, oH, oW]
    T* out = output.getBuffer();
    T* in  = input.getBuffer();

    const int kD = (int)extraParams[0];
    const int kH = (int)extraParams[1];
    const int kW = (int)extraParams[2];
    const int sD = (int)extraParams[3];
    const int sH = (int)extraParams[4];
    const int sW = (int)extraParams[5];
    const int pD = (int)extraParams[6];
    const int pH = (int)extraParams[7];
    const int pW = (int)extraParams[8];
    const int dD = (int)extraParams[9]; 
    const int dH = (int)extraParams[10];
    const int dW = (int)extraParams[11];

    int poolingMode = (int)extraParams[12];
    T extraParam0 = extraParams[13];

    const int kDEff = kD + (kD-1)*(dD-1);
    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);
    const int oD = output.sizeAt(2);
    const int oH = output.sizeAt(3);
    const int oW = output.sizeAt(4);
    const Nd4jLong iStride0 = input.stridesOf()[0];
    const Nd4jLong iStride1 = input.stridesOf()[1];
    const Nd4jLong iStride2 = input.stridesOf()[2];
    const Nd4jLong iStride3 = input.stridesOf()[3];
    const Nd4jLong iStride4 = input.stridesOf()[4];
    const Nd4jLong oStride0 = output.stridesOf()[0];
    const Nd4jLong oStride1 = output.stridesOf()[1];
    const Nd4jLong oStride2 = output.stridesOf()[2];
    const Nd4jLong oStride3 = output.stridesOf()[3];
    const Nd4jLong oStride4 = output.stridesOf()[4];
    const Nd4jLong iStep2   = dD*iStride2;
    const Nd4jLong iStep3   = dH*iStride3;
    const Nd4jLong iStep4   = dW*iStride4;
    const Nd4jLong size01   = bS*iC;
    const Nd4jLong size234  = oD*oH*oW;
    const int kProd    = kD*kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;
    const T iStep4Inv = 1./iStep4;

    const bool weirdStride = output.ordering() == 'f' || output.ews() != 1;

#pragma omp parallel for if(size01 > nd4j::Environment::getInstance()->elementwiseThreshold()) collapse(2) schedule(guided)
    for(int b = 0; b < bS; ++b) {
        for(int c = 0; c < iC; ++c) {
                                
            T *pOut = out + b * oStride0 + c * oStride1;
            T *pIn  = in  + b * iStride0 + c * iStride1;

#pragma omp parallel for if(size234 > nd4j::Environment::getInstance()->elementwiseThreshold()) collapse(3) schedule(guided)
            for(int od = 0; od < oD; ++od) {
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                            
                        Nd4jLong dstart = od * sD - pD;
                        Nd4jLong hstart = oh * sH - pH;
                        Nd4jLong wstart = ow * sW - pW;
                        Nd4jLong dend = dstart + kDEff;
                        Nd4jLong hend = hstart + kHEff;
                        Nd4jLong wend = wstart + kWEff;

                        if(dstart < 0)
                            dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-dstart / dD);
                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-hstart / dH);
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-wstart / dW);
                        if(dend > iD)
                            dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(dend-iD) / dD);                            
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(hend-iH) / dH);
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(wend-iW) / dW);

                        T sum = poolingMode == 0 ? (T) -MAX_FLOAT : (T) 0;

                        dstart *= iStride2;
                        dend   *= iStride2;
                        hstart *= iStride3;
                        hend   *= iStride3;
                        wstart *= iStride4;
                        wend   *= iStride4;
                        
                        switch(poolingMode) {
/*************************************************************************/
                            case 0: {   // max
#pragma omp simd reduction(maxT:sum) collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                            T val = pIn[kd + kh + kw];
                                            if (val > sum)
                                                sum = val;
                                        }
                                break;
                            }
/*************************************************************************/                            
                            case 1: {   // avg
#pragma omp simd reduction(sumT:sum) collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                            sum += pIn[kd + kh + kw];
                                
                                if ((int) extraParam0 == 0)         //Exclude padding
                                    sum /= (Nd4jLong)nd4j::math::nd4j_ceil<T>((dend-dstart) * iStep2Inv) * (Nd4jLong)nd4j::math::nd4j_ceil<T>((hend-hstart) * iStep3Inv) * (Nd4jLong)nd4j::math::nd4j_ceil<T>((wend-wstart) * iStep4Inv);   //Accounts for dilation
                                else if ((int) extraParam0 == 1)    //Include padding
                                    sum /= kProd;
                                break;
                            }
/*************************************************************************/                            
                            case 2: {   // pnorm
#pragma omp simd reduction(sumT:sum) collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                            sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);
                                
                                sum = nd4j::math::nd4j_pow<T>(sum, (T) 1. / extraParam0);
                                break;
                            }
                            
                            default: {
                                nd4j_printf("ConvolutionUtils::pooling3d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                                throw "";
                            }
                        }

                        if (weirdStride)
                            pOut[od * oStride2 + oh * oStride3 + ow * oStride4] = sum;                            
                        else                            
                            *pOut++ = sum;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling2dBP(NDArray<T>& input, NDArray<T>& gradO, NDArray<T>& gradI, const T* extraParams) {
    // input [bS, iC, iH, iW]
    // gradI [bS, iC, iH, iW] -> gradI is output in this function
    // gradO [bS, iC, oH, oW]    
    
    T* in = input.getBuffer();
    T* gI = gradI.getBuffer();
    T* gO = gradO.getBuffer();

    const int kH = (int)extraParams[0];
    const int kW = (int)extraParams[1];
    const int sH = (int)extraParams[2];
    const int sW = (int)extraParams[3];
    const int pH = (int)extraParams[4];
    const int pW = (int)extraParams[5];
    const int dH = (int)extraParams[6];           //Dilation, height dimension
    const int dW = (int)extraParams[7];           //Dilation, width dimension
    const int poolingMode = (int)extraParams[8];
    const T extraParam0 = extraParams[9];

    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = gradI.sizeAt(0);
    const int iC = gradI.sizeAt(1);
    const int iH = gradI.sizeAt(2);
    const int iW = gradI.sizeAt(3);            
    const int oH = gradO.sizeAt(2);
    const int oW = gradO.sizeAt(3);
    const Nd4jLong iStride0 = gradI.stridesOf()[0];
    const Nd4jLong iStride1 = gradI.stridesOf()[1];
    const Nd4jLong iStride2 = gradI.stridesOf()[2];
    const Nd4jLong iStride3 = gradI.stridesOf()[3];
    const Nd4jLong oStride0 = gradO.stridesOf()[0];
    const Nd4jLong oStride1 = gradO.stridesOf()[1];
    const Nd4jLong oStride2 = gradO.stridesOf()[2];
    const Nd4jLong oStride3 = gradO.stridesOf()[3];         
    const Nd4jLong iStep2 = dH*iStride2;
    const Nd4jLong iStep3 = dW*iStride3;
    const Nd4jLong ihEnd  = iH*iStride2;
    const Nd4jLong iwEnd  = iW*iStride3;
    const Nd4jLong size01 = bS*iC;
    const Nd4jLong size23 = oH*oW;
    const int kProd  = kH*kW;
    const T iStep2Inv = 1./iStep2; 
    const T iStep3Inv = 1./iStep3;

    const bool weirdStrideGradO = gradO.ordering() == 'f' || gradO.ews() != 1;
    const bool weirdStrideGradI = gradI.ordering() == 'f' || gradI.ews() != 1;
    
#pragma omp parallel for if(size01 > nd4j::Environment::getInstance()->elementwiseThreshold()) collapse(2) schedule(guided)
    for(int b = 0; b < bS; ++b) {
        for(int c = 0; c < iC; ++c) {
                    
            T *pgO = gO + b * oStride0 + c * oStride1;
            T *pgI = gI + b * iStride0 + c * iStride1;
            T *pIn = in + (pgI - gI);                               // == in + b * iStride0 + c * iStride1

            // below we do initial zeroing of all elements of gradI
            if(weirdStrideGradI)
#pragma omp parallel for simd collapse(2)                        
                for(Nd4jLong ih = 0; ih < ihEnd; ih+=iStride2) 
                    for(Nd4jLong iw = 0; iw < iwEnd; iw+=iStride3)
                        pgI[ih + iw] = 0.;
            else
                memset(pgI, 0, iStride1*sizeof(T));
                

#pragma omp parallel for if(size23 > nd4j::Environment::getInstance()->elementwiseThreshold()) collapse(2) schedule(guided)
            for(int oh = 0; oh < oH; ++oh) {
                for(int ow = 0; ow < oW; ++ow) {
                            
                    Nd4jLong hstart = oh * sH - pH;
                    Nd4jLong wstart = ow * sW - pW;
                    Nd4jLong hend = hstart + kHEff;
                    Nd4jLong wend = wstart + kWEff;

                    if(hstart < 0)
                        hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-hstart / dH);
                    if(wstart < 0)
                        wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-wstart / dW);
                    if(hend > iH)
                        hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(hend-iH) / dH);                            
                    if(wend > iW)
                        wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(wend-iW) / dW);                        

                    hstart *= iStride2;
                    hend   *= iStride2;
                    wstart *= iStride3;
                    wend   *= iStride3;
                    
                    T valOut = weirdStrideGradO ? pgO[oh * oStride2 + ow * oStride3] : *pgO++;

                    switch(poolingMode) {
/*************************************************************************/
                        case 0: {   // max
                            T sum = -MAX_FLOAT;
                            Nd4jLong maxKH, maxKW;
#pragma omp simd reduction(maxT:sum) collapse(2)
                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) {
                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                    T valIn = pIn[kh + kw];
                                    if (valIn > sum) {
                                        sum = valIn;
                                        maxKH = kh;
                                        maxKW = kw;
                                    }
                                }
                            }                               
                            pgI[maxKH + maxKW] += valOut;
                            break;
                        }
/*************************************************************************/                           
                        case 1: {   // avg
                            if ((int) extraParam0 == 0)         //Exclude padding
                                valOut /= (Nd4jLong)(nd4j::math::nd4j_ceil<T>((hend-hstart) * iStep2Inv)) * (Nd4jLong)nd4j::math::nd4j_ceil<T>((wend-wstart) * iStep3Inv);   //Accounts for dilation
                            else if ((int) extraParam0 == 1)    //Include padding
                                valOut /= kProd;
#pragma omp simd collapse(2)
                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                    pgI[kh + kw] += valOut;
                            break;
                        }
/*************************************************************************/                            
                        case 2: {   // pnorm
                            T sum = 0.;
#pragma omp simd reduction(sumT:sum) collapse(2)
                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                    sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);
                                
                            valOut *= nd4j::math::nd4j_pow<T>(sum, ((T)1. - extraParam0) / extraParam0);
#pragma omp simd collapse(2)
                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                    pgI[kh + kw] += valOut * nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0 - 1.);
                            break;
                        }
                        default: {
                            nd4j_printf("special_ops::pooling2dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                            throw "";
                        }
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void ConvolutionUtils<T>::pooling3dBP(NDArray<T>& input, NDArray<T>& gradO, NDArray<T>& gradI, const T* extraParams) {
    // input [bS, iC, iD, iH, iW]
    // gradI [bS, iC, iD, iH, iW] -> gradI is output in this function
    // gradO [bS, iC, oD, oH, oW]    
    
    T* in = input.getBuffer();
    T* gI = gradI.getBuffer();
    T* gO = gradO.getBuffer();

    const int kD = (int)extraParams[0];
    const int kH = (int)extraParams[1];
    const int kW = (int)extraParams[2];
    const int sD = (int)extraParams[3];
    const int sH = (int)extraParams[4];
    const int sW = (int)extraParams[5];
    const int pD = (int)extraParams[6];
    const int pH = (int)extraParams[7];
    const int pW = (int)extraParams[8];
    const int dD = (int)extraParams[9];
    const int dH = (int)extraParams[10];
    const int dW = (int)extraParams[11];
    const int poolingMode = (int)extraParams[12];
    const T extraParam0 = extraParams[13];

    const int kDEff = kD + (kD-1)*(dD-1);
    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = gradI.sizeAt(0);
    const int iC = gradI.sizeAt(1);
    const int iD = gradI.sizeAt(2);
    const int iH = gradI.sizeAt(3);
    const int iW = gradI.sizeAt(4);            
    const int oD = gradO.sizeAt(2);
    const int oH = gradO.sizeAt(3);
    const int oW = gradO.sizeAt(4);
    const Nd4jLong iStride0 = gradI.stridesOf()[0];
    const Nd4jLong iStride1 = gradI.stridesOf()[1];
    const Nd4jLong iStride2 = gradI.stridesOf()[2];
    const Nd4jLong iStride3 = gradI.stridesOf()[3];
    const Nd4jLong iStride4 = gradI.stridesOf()[4];
    const Nd4jLong oStride0 = gradO.stridesOf()[0];
    const Nd4jLong oStride1 = gradO.stridesOf()[1];
    const Nd4jLong oStride2 = gradO.stridesOf()[2];
    const Nd4jLong oStride3 = gradO.stridesOf()[3];
    const Nd4jLong oStride4 = gradO.stridesOf()[4];

    const Nd4jLong iStep2  = dD*iStride2;
    const Nd4jLong iStep3  = dH*iStride3;
    const Nd4jLong iStep4  = dW*iStride4;
    const Nd4jLong idEnd   = iD*iStride2;
    const Nd4jLong ihEnd   = iH*iStride3;
    const Nd4jLong iwEnd   = iW*iStride4;
    const Nd4jLong size01  = bS*iC;
    const Nd4jLong size234 = oD*oH*oW;
    const int kProd   = kD*kH*kW;
    const T iStep2Inv = 1./iStep2; 
    const T iStep3Inv = 1./iStep3;
    const T iStep4Inv = 1./iStep4;

    const bool weirdStrideGradO = gradO.ordering() == 'f' || gradO.ews() != 1;
    const bool weirdStrideGradI = gradI.ordering() == 'f' || gradI.ews() != 1;
    
// #pragma omp parallel for if(size01 > nd4j::Environment::getInstance()->elementwiseThreshold()) collapse(2) schedule(guided)
#pragma omp parallel for collapse(2) schedule(guided)    
    for(int b = 0; b < bS; ++b) {
        for(int c = 0; c < iC; ++c) {
                    
            T *pgO = gO + b * oStride0 + c * oStride1;
            T *pgI = gI + b * iStride0 + c * iStride1;
            T *pIn = in + (pgI - gI);                               // == in + b * iStride0 + c * iStride1

            // below we do initial zeroing of all elements of gradI
            if(weirdStrideGradI)
#pragma omp parallel for simd collapse(3)
                for(Nd4jLong id = 0; id < idEnd; id+=iStride2) 
                    for(Nd4jLong ih = 0; ih < ihEnd; ih+=iStride3) 
                        for(Nd4jLong iw = 0; iw < iwEnd; iw+=iStride4)
                            pgI[id + ih + iw] = 0.;
            else
                memset(pgI, 0, iStride1*sizeof(T));
                

// #pragma omp parallel for if(size234 > nd4j::Environment::getInstance()->elementwiseThreshold()) collapse(3) schedule(guided)
#pragma omp parallel for collapse(3) schedule(guided)            
            for(int od = 0; od < oD; ++od) {
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                            
                        Nd4jLong dstart = od * sD - pD;
                        Nd4jLong hstart = oh * sH - pH;
                        Nd4jLong wstart = ow * sW - pW;
                        Nd4jLong dend = dstart + kDEff;
                        Nd4jLong hend = hstart + kHEff;
                        Nd4jLong wend = wstart + kWEff;

                        if(dstart < 0)
                            dstart += dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-dstart / dD);
                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-hstart / dH);
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)-wstart / dW);
                        if(dend > iD)
                            dend -= dD * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(dend-iD) / dD);
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(hend-iH) / dH);
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>((T)(wend-iW) / dW);                        

                        dstart *= iStride2;
                        dend   *= iStride2;
                        hstart *= iStride3;
                        hend   *= iStride3;
                        wstart *= iStride4;
                        wend   *= iStride4;
                    
                        T valOut = weirdStrideGradO ? pgO[od*oStride2 + oh*oStride3 + ow*oStride4] : *pgO++;

                        switch(poolingMode) {
/*************************************************************************/
                            case 0: {   // max
                                T sum = -MAX_FLOAT;
                                Nd4jLong maxKD, maxKH, maxKW;
#pragma omp simd reduction(maxT:sum) collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) {
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) {
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                            T valIn = pIn[kd + kh + kw];
                                            if (valIn > sum) {
                                                sum = valIn;
                                                maxKD = kd;
                                                maxKH = kh;
                                                maxKW = kw;
                                            }
                                        }
                                    }                               
                                }
                                pgI[maxKD + maxKH + maxKW] += valOut;
                                break;
                            }
/*************************************************************************/                           
                            case 1: {   // avg
                                if ((int) extraParam0 == 0)         //Exclude padding
                                    valOut /= (Nd4jLong)(nd4j::math::nd4j_ceil<T>((dend-dstart) * iStep2Inv)) * (Nd4jLong)(nd4j::math::nd4j_ceil<T>((hend-hstart) * iStep3Inv)) * (Nd4jLong)nd4j::math::nd4j_ceil<T>((wend-wstart) * iStep4Inv);   //Accounts for dilation
                                else if ((int) extraParam0 == 1)    //Include padding
                                    valOut /= kProd;
#pragma omp simd collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                            pgI[kd + kh + kw] += valOut;
                                break;
                            }
/*************************************************************************/                            
                            case 2: {   // pnorm
                                T sum = 0.;
#pragma omp simd reduction(sumT:sum) collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                            sum += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);
                                
                                valOut *= nd4j::math::nd4j_pow<T>(sum, ((T)1. - extraParam0) / extraParam0);
#pragma omp simd collapse(3)
                                for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                            pgI[kd + kh + kw] += valOut * nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0 - 1.);
                                break;
                            }
                            default: {
                                nd4j_printf("special_ops::pooling2dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                                throw "";
                            }
                        }
                    }
                }
            }
        }
    }
}

template class ND4J_EXPORT ConvolutionUtils<float>;
template class ND4J_EXPORT ConvolutionUtils<float16>;
template class ND4J_EXPORT ConvolutionUtils<double>;
    
}
}
