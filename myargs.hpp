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

#pragma once
#include <unordered_map>
#include <type_traits>
#include <string_view>
#include <algorithm>

namespace myargs {
class Args
{
   using ArgsMap = std::unordered_map<std::string,std::string>;
   public:
   using iterator = ArgsMap::iterator;
   ArgsMap m;
   size_t cmd_nb;
   std::string empty_s{""};
   iterator begin() { return m.begin(); }
   iterator end() { return m.end(); }
   Args() = delete;
   Args(int argc, char**argv) : m(), cmd_nb(0)
   {
      m.reserve(argc*2);
      for (char **it = argv, **end = argv+argc; it < end; it++)
      {
         auto arg = std::string_view( *it );
         if (arg.empty() )
            continue;
         auto prefix_len=0;
         while (arg[0] == '-') {
            arg.remove_prefix(1);
            prefix_len++;
         }

         if (prefix_len==0) {
            m[ std::string("%") + std::to_string(cmd_nb++) ] = arg;
         } else
         {
            if (prefix_len==1) {
               m[ std::string{arg.substr(0,1)} ] = arg.substr(1);
            }
            else {
               auto pos = arg.find('=');
               if (pos == std::string::npos)
                  m[ std::string{arg} ] = {"true"};
               else
                  m[ std::string{ arg.substr(0,pos) } ] = arg.substr(pos+1);
            }
         }
      }
   }
   template<typename T>
   auto operator[](T && opt) -> decltype( m[opt] )
   {
      return has(opt) ? m[opt] : empty_s;
   }


   std::string& operator[](int cmd)
   {
      return has(cmd) ? m[ std::string("%") + std::to_string(cmd) ] : empty_s;
   }

   template<typename T>
   auto has(T && opt) -> decltype( m.contains(opt) )
   {
      return m.contains(opt);
   }

   bool has(int cmd)
   {
      return cmd >= 0 && cmd < cmd_nb;
   }

   template<int v=0,typename T>
   auto get(T&& opt) -> decltype( m[opt], v )
   {
      if (has(opt))
      {

         try {
            return std::stol( m[opt] );
         } catch(...) {

         }
      }
      return v;
   }

   template<int v, int lo, int hi, typename T>
   auto get(T&& opt) -> decltype( m[opt], v )
   {
      if (has(opt)) {
         try {
            return std::clamp<int>(std::stol( m[opt] ),lo,hi);
         } catch(...) {

         }
      }
      return std::clamp(v,lo,hi);
   }



   template<typename T, typename Default >
   auto get(T && opt, Default v) -> decltype( m[opt], m[{v}], std::string() )
   {
      return { has(opt) ? m[opt] : v};
   }
};
}