/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023 Vladimir Poslavskiy
 * vovach777@yandex.ru
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cmath>
#include <vector>
#include <cstdlib>
#include <iterator>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <functional>
#include <type_traits>
#include "blur.hpp"

#define MAX_INTENSITY 255


/*! Computes image histogram
    \param input Input image
    \param hist Image histogram that is returned
*/


template<typename T>
void computeHistogram(size_t N, T *input, std::vector<size_t> &hist){
  // Compute number of pixels

  // Initialize array
  if (hist.size() == 0)
      hist.resize(MAX_INTENSITY+1);
   else
      std::fill(hist.begin(), hist.end(), 0);

  // Iterate image
  auto bound = hist.size()-1;
  for (int i=0; i<N; i++) {
    const auto v = input[i];
    hist[ v > bound ? bound : v ] += 1;
  }
  //hist[ MAX_INTENSITY ] = 0;

}

/*! Segments the image using the computed threshold
    \param input Input image
    \param output Output segmented image
    \param threshold Threshold used for segmentation
*/

/*! Computes Otsus segmentation
    \param input Input image
    \param hist Image histogram
    \param output Output segmented image
    \param overrided_threshold Input param that overrides threshold
*/

template<typename T>
int computeOtsusSegmentation(size_t N, T *input, std::vector<size_t> &hist){
  // Compute number of pixels
  //long int N = input.get_width() * input.get_height();
  int threshold = 0;

  // Compute threshold
  // Init variables
  float sum = 0;
  float sumB = 0;
  int q1 = 0;
  int q2 = 0;
  float varMax = 0;

  // Auxiliary value for computing m2

  for (int i=0; i<hist.size(); i++){
    sum += i * static_cast<float>(hist[i]);
  }

  for (int i=0; i<hist.size(); i++){
    // Update q1
    q1 += hist[i];
    if (q1 == 0)
      continue;
    // Update q2
    q2 = N - q1;

    if (q2 == 0)
      break;
    // Update m1 and m2
    sumB += i * static_cast<float>(hist[i]);
    float m1 = sumB / q1;
    float m2 = (sum - sumB) / q2;

    // Update the between class variance
    float varBetween = static_cast<float>(q1) * static_cast<float>(q2) * (m1 - m2) * (m1 - m2);

    // Update the threshold if necessary
    if (varBetween > varMax) {
      varMax = varBetween;
      threshold = i;
    }
  }
  // // Perform the segmentation
  // //segmentImage(input, output, threshold);
  // // Iterate image
  // for (int i = 0; i < N; i++) {
  //   const int v = input[i];
  //   if (v == 255)
  //     continue;
  //   // Build the segmented image
  //   output[i] =  (v > threshold) ? 255 : 0;
  //   // if (v <= threshold && output[i] == 255)
  //   //     output[i] =  value;
  // }
  return threshold;
}

template<typename T>
int otsu_of(size_t N, T*begin)
{
  std::vector<size_t> hist;
  computeHistogram(N, begin, hist);
  return computeOtsusSegmentation(N, begin, hist);
}



namespace  max {
template<typename T>
struct Surface {
  static constexpr double max_color = static_cast<double>(std::numeric_limits<T>::max());
  T* surface;
  T color;
  int width;
  int height;
  Surface(T *surface, int width, int height) : surface(surface), width(width), height(height) {};
  Surface& operator=(const Surface& other) {
    if (width == other.width && height == other.height) {
      std::copy(other.surface, other.surface+width*height, surface);
    } else {
      for (int y=0; y<height; y++)
      for (int x=0; x<height; y++) {
        color = other.get_point_clipped(x,y);
        draw_point(x,y);
      }
    }
    return *this;
  }
  ~Surface() {};
inline T get_point(int x, int y) {
   return (x<width && y<height && x>=0 && y>=0) ? surface[y*width+x] : 0;
}
inline T get_point_clipped(int x, int y) const {
  auto clamp_x = std::clamp(x,0,width-1);
  auto clamp_y = std::clamp(y,0,height-1);
   return surface[  clamp_y*width+clamp_x ];
}

inline void draw_point(int x, int y)
 {
   if(x<width && y<height && x>=0 && y>=0)
      surface[y*width+x] = color;
 }



template<typename F>
inline auto process_point(int x, int y, F f) -> decltype(f(std::declval<T&>()) , void())
{
  if(x<width && y<height && x>=0 && y>=0)
    f( surface[y*width+x] );
}

template<typename F>
inline auto process_point(int x, int y, F f) -> decltype(f(std::declval<int>(), std::declval<int>()) , void())
{
  if(x<width && y<height && x>=0 && y>=0)
    f( x, y );
}

template<typename F>
inline auto process_point(int x, int y, F f) -> decltype(f(std::declval<int>(), std::declval<int>(), std::declval<T&>()) , void())
{
  if(x<width && y<height && x>=0 && y>=0)
    f( x, y, surface[y*width+x] );
}

void fill() {
  std::fill(surface, surface+width*height,color);
}

static inline T gamma_correction( T v, double gamma)
{
  return static_cast<T>(pow(v / max_color, 1 / gamma) * max_color);
}

void gamma_correction_circle( int centerX, int centerY, int radius, double gamma)
{
  process_filled_circle(centerX,centerY,radius,[gamma] (T&v) {
    v = gamma_correction(v,gamma);
  });

}

void draw_circle( int xc, int yc, int r)
{
    int x = 0, y = r;
    int d = 3 - 2 * r;
    //drawCircle(xc, yc, x, y);
    while (y >= x)
    {
        // for each pixel we will
        // draw all eight pixels

        x++;

        // check for decision parameter
        // and correspondingly
        // update d, x, y
        if (d > 0)
        {
            y--;
            d = d + 4 * (x - y) + 10;
        }
        else
            d = d + 4 * x + 6;
        //drawCircle(xc, yc, x, y);
        draw_point(xc+x, yc + y );

    }
}

#if 0
void draw_filled_circle(int center_x, int center_y, int radius) {
  // Начальные значения
  int x = 0;
  int y = radius;
  int error = -radius;

  // Рисуем первую линию
  while (x <= y) {

    draw_horizontal_line(center_x - y, center_x + y, center_y + x);
    draw_horizontal_line(center_x - x, center_x + x, center_y + y);
    draw_horizontal_line(center_x - y, center_x + y, center_y - x);
    draw_horizontal_line(center_x - x, center_x + x, center_y - y);


    if (error <= 0) {
      error += 2 * x + 1;
      ++x;
    }
    else {
      error -= 2 * y - 1;
      --y;
    }
  }
}
#endif

template<typename F>
void process_rect(int x1, int y1, int x2, int y2, F f) {
  for (int y=y1; y<=y2;y++)
  for (int x=x1; x<=x2;x++)
    process_point(x,y,f);
}

template<typename F>
auto process_all(F f) -> decltype(f(std::declval<T&>()) , void())
{
  for (auto it=surface,end=surface+width*height; it<end; it++)
  {
    f(*it);
  }
}

template<typename F>
auto process_all(F f) -> decltype(f(std::declval<int>(),std::declval<int>()) , void())
{
  process_rect(0,0,width-1,height-1,f);
}


void draw_filled_circle(int cx, int cy, int radius) {
  process_filled_circle(cx,cy,radius, [*this](T&v) { v=this->color; } );
}


template<typename F>
void process_filled_circle(int cx, int cy, int radius, F f)
{
    int error = -radius;
    int x = radius;
    int y = 0;

    while (x >= y)
    {
        int lastY = y;

        error += y;
        ++y;
        error += y;

        plot4points(cx, cy, x, lastY, f);

        if (error >= 0)
        {
            if (x != lastY)
                plot4points(cx, cy, lastY, x, f);

            error -= x;
            --x;
            error -= x;
        }
    }
}



template<typename F>
inline void plot4points(int cx, int cy, int x, int y,F f)
{
    draw_horizontal_line(cx - x, cx + x, cy + y, f);
    if (y != 0)
        draw_horizontal_line(cx - x, cx + x, cy - y, f);
}


template<typename F>
auto draw_horizontal_line(int x1, int x2, int y, F f) {
  for (int x = x1; x <= x2; ++x) {
    process_point(x,y,f);
  }
}

std::vector<T> get_circle_vector( int centerX, int centerY, int radius)
{
  std::vector<T> result;
  result.reserve(radius*radius*4);
  process_filled_circle( centerX, centerY, radius,
   [&result](uint8_t& v) {
      result.push_back(v);
   });
  return result;
}

std::vector<T> get_row( int y ) {
  y = std::clamp(y,0,height-1);
  return std::vector<T>(surface+(y*width),surface+((y+1)*width));
}

std::vector<T> get_col( int x ) {
  x = std::clamp(x,0,width-1);
  std::vector<T> col;
  col.reserve(height);

  for (int ofs=x, end=width*height; ofs<end; ofs+=width) {
    col.push_back(surface[ofs]);
  }

  return col;
}

std::vector<T> get_rect_vector(int x1, int y1, int x2, int y2) {
  std::vector<T> res;
  process_rect(x1,y1,x2,y2, [&res](T&v) { res.push_back(v); } );
  return res;
}

template<typename F>
void process_row( int y, F f ) {
  for (int x=0; x<width; x++) {
      process_point(x,y,f);
  }
}

template<typename F>
void process_col( int x, F f ) {
  for (int y=0; y<height; y++) {
      process_point(x,y,f);
  }
}


int otsu_of_circle( int centerX, int centerY, int radius)
{
    auto vec = get_circle_vector(centerX,centerY, radius);
    std::vector<size_t> hist;
    computeHistogram(vec.size(), vec.data(), hist);
    return computeOtsusSegmentation(vec.size(), vec.data(), hist);
}

int bw_percent_circle(Surface &src, int radius, int centerX, int centerY, int threshold)
{
  int b_count =0;
  int count = 0;
   process_filled_circle( centerX, centerY, radius,
        [threshold, &src,*this,&b_count,&count](int x, int y) {
          if (src.get_point(x,y) <= threshold && this->get_point(x,y)==0)
            b_count++;
          else
            b_count--;
          count++;

   });
  return b_count*100/count;
}


};
}


template <typename T, int levels>
void splite(std::vector<T> & codebook,  T* begin, T*end,int level=0, T rangeLo=std::numeric_limits<T>::min(), T rangeHi=std::numeric_limits<T>::max())
{
  auto sum = 0;
  auto count = 0;
  auto min = rangeHi;
  auto max = rangeLo;
  for (T* it=begin; it<end; it++) {
    const auto color = *it;
    if (color >= rangeLo && color <= rangeHi) {
      sum += color;
      min = std::min(min,color);
      max = std::max(max,color);
      count++;
    }
  }
  auto threshold = static_cast<T>( sum / count );

  if (level == levels || max-min < 8) {
    codebook.push_back(threshold);
    // for (T *it=begin; it<end; it++)
    // {
    //     if (*it >= rangeLo && *it <= rangeHi) {
    //         //*it = threshold;

    //     }
    // }
  }
  else
  {
    splite<T,levels>(codebook,begin,end,level+1, min,         threshold);
    splite<T,levels>(codebook,begin,end,level+1, threshold+1, max);
  }
}


template <typename T>
auto getcodebook(T* begin, T*end) {
  std::unordered_map<uint8_t,size_t> codebook;
  for (T*it=begin; it<end; it++) {
    codebook[ *it ] += 1;
  }
  std::vector<T> keys;
  keys.reserve(codebook.size());
  for (auto v : codebook)
    keys.push_back(v.first);
  std::sort(keys.begin(),keys.end());
  return keys;
}


template <typename iterator, typename T = std::decay_t<iterator>>
T find_nerest(iterator begin, iterator end, T val)
{
    if (begin >= end)
        return 0;
    auto it = std::lower_bound(begin, end, val);

    if (it == end ) {
        return *std::prev(end);
    };
    if (it == begin)
        return *it;


    const auto prev_value = *std::prev(it);
    const auto lower_distance = std::abs(prev_value - val);
    const auto upper_distance = std::abs(*it - val);
    return lower_distance < upper_distance ? prev_value : *it;

}


double distance2d(int x1, int y1, int x2, int y2) {

        auto x = (x1 - x2);
        auto y = (y1 - y2);
				return sqrt(  x*x + y*y );
}



int main(int argc, char **argv) {


  // Load input image
  int Nx,Ny, num_channels;

  //TODO: check argument
  unsigned char* input = stbi_load(argv[1], &Nx, &Ny, &num_channels, 0);
  const size_t N = Nx * Ny;
  
  if (num_channels > 1){

    //make gray
    uint8_t *img_color_p = input;
    uint8_t *img_gray_p  = input;
    size_t pixels_nb = N;
    
    while(pixels_nb--) {
      //my gray formula
      const auto v = (img_color_p[0] + img_color_p[1] + img_color_p[1] + img_color_p[2])/4;
      *img_gray_p = v;
      img_color_p += num_channels;
      img_gray_p += 1;
    }
  }
  unsigned char* output = (uint8_t*) malloc(N);

  max::Surface surface_input(input,Nx,Ny);
  max::Surface surface_output(output,Nx,Ny);
  
  
   std::string in_filename = argv[1];
   auto out_filename = in_filename.substr(0,in_filename.size()-4) + "-output.png";


    //негатив
    for (int i=0; i<N; i++) {
      input[i] = 255 - input[i];
    }

    std::copy(input,input + N, output);

    fast_gaussian_blur<uint8_t, 1, 3>(output,Nx,Ny,8);

    //blure
   //  for (int y=0; y<Ny; y+=1)
   //  for (int x=0; x<Nx; x+=1)
   //  {
   //    auto sum_x = 0;
   //    auto sum_y = 0;
   //    auto count = 0;
   //    auto sum_m = 0;
   //    //calc center of mass without this point
   //    surface_input.process_filled_circle(x,y,radius,
   //        [&](int x, int y, uint8_t &color) {
   //            sum_m += color;
   //            count++;
   //        });
   //    auto avg_color = std::max(16, sum_m / count);
   //    surface_output.color =  avg_color;
   //    surface_output.draw_point(x,y);

   //  }

    for (int i=0; i<N; i++) {
      auto thresthold = std::max(30, output[i]+5); //+5 - magic noise cutoff
      input[i] = input[i] >=  thresthold ? 0 : 255;
    }

   // squize image dimentions
   int new_x = (Nx-4)/4;
   int new_y = (Ny-4)/4;
   surface_output.width = new_x;
   surface_output.height = new_y;
   for (int y=0; y<new_y; y++)
   for (int x=0; x<new_x; x++)
   {
      int black_count = 0;
      surface_input.process_filled_circle((x+2)*4,(y+2)*4,2,[&black_count](uint8_t &v) {  if (v == 0) black_count++; else black_count--; });
      surface_output.color = black_count > 0 ? 0 : 255;
      surface_output.draw_point(x,y);
   }


  // Save output

   stbi_write_png(out_filename.data() ,new_x,new_y,1,output,new_x);


  free(output);
  free(input);
}
