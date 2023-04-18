#pragma once
// Copyright (C) 2017-2021 Basile Fraboni
// Copyright (C) 2014 Ivan Kutskir (for the original fast blur implmentation)
// All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the MIT license. For further details please refer
// to : https://mit-license.org/
//


#include <iostream>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <vector>

//!
//! \file fast_gaussian_blur_template.cpp
//! \author Basile Fraboni
//! \date 2021
//!
//! \brief This contains a C++ implementation of a fast Gaussian blur algorithm in linear time.
//! The image buffer is supposed to be of size w * h * c, with w its width, h its height, and c its number of channels.
//! The default implementation only supports up to 4 channels images, but one can easily add support for any number of channels
//! using either specific template cases or a generic function that takes the number of channels as an explicit parameter.
//! This implementation is focused on learning and readability more than on performance.
//! The fast blur algorithm is performed with several box blur passes over an image.
//! The filter converges towards a true Gaussian blur after several passes. In practice,
//! three passes are sufficient for good quality results.
//! For further details please refer to:
//!
//! http://blog.ivank.net/fastest-gaussian-blur.html
//! https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
//! https://github.com/bfraboni/FastGaussianBlur
//!

//!
//! \brief This function performs a single separable horizontal pass for box blur.
//! To complete a box blur pass we need to do this operation two times, one horizontally
//! and one vertically. Templated by buffer data type T, buffer number of channels C, and border policy P.
//! For a detailed description of border policies please refer to:
//! https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_Handling
//!
//! \param[in] in           source buffer
//! \param[in,out] out      target buffer
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] r            box dimension
//!
enum Policy {EXTEND, KERNEL_CROP};

template<typename T, int C, Policy P = KERNEL_CROP>
void horizontal_blur(const T * in, T * out, const int w, const int h, const int r)
{
    float iarr = 1.f / (r+r+1);
    #pragma omp parallel for
    for(int i=0; i<h; i++)
    {
        int ti = i*w, li = ti, ri = ti+r;
        float fv[C], lv[C], val[C];

        for(int ch = 0; ch < C; ++ch)
        {
            fv[ch] =  P == EXTEND ? in[ti*C+ch]        : 0; // unused with kcrop policy
            lv[ch] =  P == EXTEND ? in[(ti+w-1)*C+ch]  : 0; // unused with kcrop policy
            val[ch] = P == EXTEND ? (r+1)*fv[ch]       : 0;
        }

        // initial acucmulation
        for(int j=0; j<r; j++)
        for(int ch = 0; ch < C; ++ch)
        {
            val[ch] += in[(ti+j)*C+ch];
        }

        // left border - filter kernel is incomplete
        for(int j=0; j<=r; j++, ri++, ti++)
        for(int ch = 0; ch < C; ++ch)
        {
            val[ch] +=     P == EXTEND ? in[ri*C+ch] - fv[ch] : in[ri*C+ch];
            out[ti*C+ch] = P == EXTEND ? val[ch]*iarr         : val[ch]/(r+j+1);
        }

        // center of the image - filter kernel is complete
        for(int j=r+1; j<w-r; j++, ri++, ti++, li++)
        for(int ch = 0; ch < C; ++ch)
        {
            val[ch] += in[ri*C+ch] - in[li*C+ch];
            out[ti*C+ch] = val[ch]*iarr;
        }

        // right border - filter kernel is incomplete
        for(int j=w-r; j<w; j++, ti++, li++)
        for(int ch = 0; ch < C; ++ch)
        {
            val[ch] +=     P == EXTEND ? lv[ch] - in[li*C+ch] : -in[li*C+ch];
            out[ti*C+ch] = P == EXTEND ? val[ch]*iarr         : val[ch]/(r+w-j);
        }
    }
}


//!
//! \brief This function performs a 2D tranposition of an image. The transposition is done per
//! block to reduce the number of cache misses and improve cache coherency for large image buffers.
//! Templated by buffer data type T and buffer number of channels C.
//!
//! \param[in] in           source buffer
//! \param[in,out] out      target buffer
//! \param[in] w            image width
//! \param[in] h            image height
//!
template<typename T, int C>
void flip_block(const T * in, T * out, const int w, const int h)
{
    constexpr int block = 256/C;
    #pragma omp parallel for collapse(2)
    for(int x= 0; x < w; x+= block)
    for(int y= 0; y < h; y+= block)
    {
        const T * p = in + y*w*C + x*C;
        T * q = out + y*C + x*h*C;

        const int blockx= std::min(w, x+block) - x;
        const int blocky= std::min(h, y+block) - y;
        for(int xx= 0; xx < blockx; xx++)
        {
            for(int yy= 0; yy < blocky; yy++)
            {
                for(int k= 0; k < C; k++)
                    q[k]= p[k];
                p+= w*C;
                q+= C;
            }
            p+= -blocky*w*C + C;
            q+= -blocky*C + h*C;
        }
    }
}

//!
//! \brief this function converts the standard deviation of
//! Gaussian blur into a box radius for each box blur pass. For
//! further details please refer to :
//! https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
//!
//! \param[out] boxes   box radiis for kernel sizes of 2*boxes[i]+1
//! \param[in] sigma    Gaussian standard deviation
//! \param[in] n        number of box blur pass
//!
static void sigma_to_box_radius(int boxes[], const float sigma, const int n)
{
    // ideal filter width
    float wi = std::sqrt((12*sigma*sigma/n)+1);
    int wl = wi; // no need std::floor
    if(wl%2==0) wl--;
    int wu = wl+2;

    float mi = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4);
    int m = mi+0.5f; // avoid std::round by adding 0.5f and cast to integer type

    for(int i=0; i<n; i++)
        boxes[i] = ((i < m ? wl : wu) - 1) / 2;
}


template<typename T, int C, int N>
void fast_gaussian_blur(T * image, const int w, const int h, const float sigma)
{
   auto image_size = w*h;
   std::vector<T> src_vec(image, image + image_size);
   std::vector<T> dst_vec(image_size);
   T* in = src_vec.data();
   T* out = dst_vec.data();
    // compute box kernel sizes
    std::vector<int> boxes(N);
    sigma_to_box_radius(boxes.data(), sigma, N);

    // perform N horizontal blur passes
    for(int i = 0; i < N; ++i)
    {
        horizontal_blur<T,C>(in, out, w, h, boxes[i]);
        std::swap(in, out);
    }

    // flip buffer
    flip_block<T,C>(in, out, w, h);
    std::swap(in, out);

    // perform N horizontal blur passes
    for(int i = 0; i < N; ++i)
    {
        horizontal_blur<T,C>(in, out, h, w, boxes[i]);
        std::swap(in, out);
    }

    // flip buffer
    flip_block<T,C>(in, out, h, w);
    std::copy(out, out+image_size, image);
}

