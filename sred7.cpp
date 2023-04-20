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

#define SRED7
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
#include <filesystem>
#ifndef SRED7
#include "blur.hpp"
#endif
#include "myargs.hpp"

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

template <typename T>
int sred7(int size, T *input )
{
  auto d = 0, b = 0, c = 0;

   for (int i=0; i<size; i++)
      d+=input[i];
   d/=size;

   auto count = 0;
   for (int i=0; i<size; i++)
   {
      if (input[i] <= d)
      {
         b+=input[i];
         count++;
      }
   }
   b/=count;

   count = 0;
   for (int i=0; i<size; i++)
   {
      if (input[i] > b)
      {
         c+=input[i];
         count++;
      }
   }
   return count ? c / count : 0;
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



#if 0
const option::Descriptor usage[] =
{
 {UNKNOWN, 0,"","", Arg::Illegal, ""},
 {HELP,     0,"?" , "help",   Arg::Illegal, "  --help  \tPrint usage and exit." },
 {CONTRAST, 0,"c", "contrast", Arg::Numeric, "  --contrast=<value>, -c<value>  \tApply contrast: (v-128)*value+128." },
 {NEGATIVE, 0,"n", "negative",  Arg::Optional, "  --negitive, -n  \tprocess negotive of image: (255-v)." },
 {THRESHOLD_SHIFT,0,"t","threshold-shift", Arg::Numeric, "  --threshold-shift=<shift>, -t<shift>  \tcorrect black-to-white threshold. default:-1" },
 {BLACK_SHIFT, 0, "b", "black-shift", Arg::Numeric,   "  --black-shift=<shift>, -b<shift>  \tshift black-width balance. greatter -> whitely. default:2" },
 {0,0,0,0,0,0}
};
#endif


int main(int argc, char **argv) {
  // Optional parameters

  myargs::Args args(argc,argv);



  std::string in_filename = args[1];

  if ( in_filename.empty() || not std::filesystem::exists( in_filename ) ) {
    std::cerr << "input file \""<< in_filename << "\" is not exists!\n";
    return 1;
  }


  std::string out_filename = std::filesystem::path(in_filename).replace_filename( std::filesystem::path(in_filename).stem().string() + args.get("suffix", "-output")).replace_extension("png").string();

  // Load input image
  int Nx,Ny, num_channels;

  unsigned char* input = stbi_load(in_filename.data(), &Nx, &Ny, &num_channels, 0);
  const size_t N = Nx * Ny;
  if (input == NULL || N==0) {
    std::cerr << "can not read image!\n";
    return 2;
  }



  if (num_channels > 1){

    //make gray
    uint8_t *img_color_p = input;
    uint8_t *img_gray_p  = input;
    size_t pixels_nb = N;


    while(pixels_nb--) {
      const auto v = (img_color_p[0] + img_color_p[1] + img_color_p[1] + img_color_p[2])/4;
      *img_gray_p = v;
      img_color_p += num_channels;
      img_gray_p += 1;
    }

  }
  unsigned char* output = (uint8_t*) malloc(N);

  max::Surface surface_input(input,Nx,Ny);
  max::Surface surface_output(output,Nx,Ny);
  if (args.has("n") ||  args.has("negative")) {
    //негатив
    surface_input.process_all([](uint8_t &v) { v= (255-v); });
  }

   //contrast
  if (args.has("c") || args.has("contrast")) {
    auto contrast_multipler = args.has("c") ? args.get("c") : args.get("contrast");
    surface_input.process_all([contrast_multipler](uint8_t &v) { v= std::clamp<int>( (v-128) * contrast_multipler + 128,0,255); });
  }

  if (args.has("cn-file"))
    stbi_write_png(args["cn-file"].data() ,Nx,Ny,1,input,Nx);



#ifdef OTSU_1
   int new_x = (Nx-4)/4;
   int new_y = (Ny-4)/4;
   surface_output.width = new_x;
   surface_output.height = new_y;
   for (int y=0; y<new_y; y++)
   for (int x=0; x<new_x; x++)
   {
      int black_count = 0;
      auto threshold = surface_input.otsu_of_circle((x+2)*4,(y+2)*4,8);
      surface_input.process_filled_circle((x+2)*4,(y+2)*4,2,[&](uint8_t &v) {  if (v <= threshold-1) black_count++; else black_count--; });
      surface_output.color = black_count > 0 ? 0 : 255;
      surface_output.draw_point(x,y);
   }
#endif

#ifdef OTSU_2
   constexpr auto squze_size = 2;
   int new_x = (Nx-squze_size)/squze_size;
   int new_y = (Ny-squze_size)/squze_size;
   surface_output.width = new_x;
   surface_output.height = new_y;
   for (int y=0; y<new_y; y++)
   for (int x=0; x<new_x; x++)
   {
      int black_count = 0;
      auto xx = (x + squze_size/2)*squze_size;
      auto yy = (y + squze_size/2)*squze_size;
      auto threshold = surface_input.otsu_of_circle(xx,yy,squze_size*3);
      surface_input.process_filled_circle(xx,yy,squze_size/2,[&](uint8_t &v) {  if (v <= threshold-1) black_count++; else black_count--; });
      surface_output.color = black_count > 0 ? 0 : 255;
      surface_output.draw_point(x,y);
   }

#endif

#ifdef SRED7

   constexpr auto squze_size = 2;
   //constexpr auto S = (squze_size/2+1)*(squze_size/2+1)*3;
   int new_x = (Nx-squze_size)/squze_size;
   int new_y = (Ny-squze_size)/squze_size;
   surface_output.width = new_x;
   surface_output.height = new_y;
   auto threshold_shift = args.get<-1>("t");
   auto black_shift = args.get<5>("b");
   auto min_threshold = args.get<0, 0,255>("min");
   auto max_threshold = args.get<160,0,255>("max");
   auto show_diagonal = args.has("print-diagonal");
   for (int y=0; y<new_y; y++)
   for (int x=0; x<new_x; x++)
   {
      int black_count = 0;
      auto xx = (x + squze_size/2)*squze_size;
      auto yy = (y + squze_size/2)*squze_size;
      auto vecir = surface_input.get_circle_vector(xx,yy,squze_size*3);

      auto threshold = std::clamp<int>( sred7(vecir.size(), vecir.data() ) + threshold_shift, min_threshold, max_threshold );
      if (show_diagonal && x==y)
         std::cout << threshold << " ";
      surface_input.process_filled_circle(xx,yy,squze_size,[&](uint8_t &v) {  if (v <= threshold ) black_count++; else black_count--; });
      surface_output.color = black_count >= black_shift ? 0 : 255;
      surface_output.draw_point(x,y);
   }

#endif

  // Save output

   stbi_write_png(out_filename.data() ,new_x,new_y,1,output,new_x);



  free(output);
  free(input);
}
