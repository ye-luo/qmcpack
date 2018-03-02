//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////





/** @file accumulators.h
 * @brief Define and declare accumulator_set
 *
 * A temporary implementation to handle scalar samples and will be replaced by
 * boost.Accumulator
 */
#ifndef QMCPLUSPLUS_ACCUMULATORS_H
#define QMCPLUSPLUS_ACCUMULATORS_H

#include <config/stdlib/limits.h>
#include <iostream>
#include <type_traits>

/** generic accumulator of a scalar type
 *
 * To simplify i/o, the values are storged in contens
 */
template<typename T,
  typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
struct accumulator_set
{
  typedef T value_type;
  typedef T return_type;

  //enum {VALUE=0,VALUESQ=1,WEIGHT=2,CAPACITY=4};
  enum {VALUE=0,WEIGHT=1,CAPACITY=2};
  T properties[CAPACITY];

  inline accumulator_set()
  {
    for(int i=0; i<CAPACITY; ++i)
      properties[i]=value_type();
  }

  /** add a sample */
  inline void operator()(value_type x)
  {
    properties[VALUE]  +=x;
    //properties[VALUESQ]+=x*x;
    properties[WEIGHT] +=1.0;
  }

  /** add a sample and  */
  inline void operator()(value_type x, value_type w)
  {
    properties[VALUE]  +=w*x;
    //properties[VALUESQ]+=w*x*x;
    properties[WEIGHT] +=w;
  }

  /** reset properties
   * @param v cummulative value
   * @param vv cummulative valuesq
   * @param w cummulative weight
  inline void reset(value_type v, value_type vv, value_type w)
  {
    properties[VALUE]  =v;
    //properties[VALUESQ]=vv;
    properties[WEIGHT] =w;
  }
  */

  /** reset properties
   * @param v cummulative value
   * @param w cummulative weight
   */
  inline void reset(value_type v, value_type w)
  {
    properties[VALUE]  =v;
    //properties[VALUESQ]=v*v;
    properties[WEIGHT] =w;
  }

  /** return true if Weight!= 0 */
  inline bool good() const
  {
    return properties[WEIGHT]>0;
  }

  /** return the sum */
  inline return_type result() const
  {
    return properties[VALUE];
  }

  /** return the count
   *
   * Will return the sum of weights of each sample
   */
  inline return_type count() const
  {
    return properties[WEIGHT];
  }
  
  /** return the capacity */
  inline int capacity() const
  {
    return CAPACITY;
  } 

  ///return the mean
  inline return_type mean() const
  {
    return good()?properties[VALUE]/properties[WEIGHT]:0.0;
  }

  inline void clear()
  {
    for(int i=0; i<CAPACITY; ++i)
      properties[i]=value_type();
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, accumulator_set<T>& rhs)
{
  os << "accumulator_set: "
     << " value = " << rhs.properties[0]
     << " weight = " << rhs.properties[1];
  return os;
}


#endif
