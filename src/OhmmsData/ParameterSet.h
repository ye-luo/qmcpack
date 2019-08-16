//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: D. Das, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef OHMMS_OHMMSPARAMETERSET_H
#define OHMMS_OHMMSPARAMETERSET_H

#include <map>
#include <string>
#include "OhmmsData/OhmmsParameter.h"

/** class to handle a set of parameters
 */
class ParameterSet
{
public:
  typedef std::map<std::string, OhmmsElementBase*> Container_t;
  typedef Container_t::iterator iterator;
  typedef Container_t::const_iterator const_iterator;

  ParameterSet(const std::string aname = "parameter")
    : myName(aname)
  {}

  ~ParameterSet()
  {
    for (const auto& param : m_param)
      delete param.second;
  }

  inline bool get(std::ostream& os) const
  {
    for (const auto& param : m_param)
      param.second->get(os);
    return true;
  }

  inline bool put(std::istream& is) { return true; }

  /** assign parameters to the set
   * @param cur the xml node to work on
   * @return true, if any valid parameter is processed.
   *
   * Accept both
   * - <aname> value </aname>
   * - <parameter name="aname"> value </parameter>
   * aname is converted into lower cases.
   */
  inline bool put(xmlNodePtr cur)
  {
    if (cur == NULL)
      return true; //handle empty node
    cur            = cur->xmlChildrenNode;
    bool something = false;
    while (cur != NULL)
    {
      std::string cname((const char*)(cur->name));
      tolower(cname);
      iterator it_tag = m_param.find(cname);
      if (it_tag == m_param.end())
      {
        if (cname == myName)
        {
          const xmlChar* aptr = xmlGetProp(cur, (const xmlChar*)"name");
          if (aptr)
          {
            //string aname = (const char*)(xmlGetProp(cur, (const xmlChar *) "name"));
            std::string aname((const char*)aptr);
            tolower(aname);
            iterator it = m_param.find(aname);
            if (it != m_param.end())
            {
              something = true;
              (*it).second->put(cur);
            }
          }
        }
      }
      else
      {
        something = true;
        (*it_tag).second->put(cur);
      }
      cur = cur->next;
    }
    return something;
  }

  inline void reset() {}

  /** add a new parameter corresponding to an xmlNode <parameter/>
   *@param aparam reference the object which this parameter is assigned to.
   *@param aname the value of the name attribute
   *@param uname the value of the condition attribute
   *
   *The attributes of a parameter are
   * - name, the name of the parameter
   * - condition, the unit of the parameter
   *The condition will be used to convert the external unit to the internal unit.
   */
  template<class PDT>
  inline void add(PDT& aparam, const char* aname_in, const char* uname)
  {
    std::string aname(aname_in);
    tolower(aname);
    iterator it = m_param.find(aname);
    if (it == m_param.end())
    {
      m_param[aname] = new OhmmsParameter<PDT>(aparam, aname.c_str(), uname);
    }
  }

  template<class PDT>
  inline void setValue(const std::string& aname_in, PDT aval)
  {
    std::string aname(aname_in);
    tolower(aname);
    iterator it = m_param.find(aname);
    if (it != m_param.end())
    {
      (dynamic_cast<OhmmsParameter<PDT>*>((*it).second))->setValue(aval);
    }
  }

private:
  // all the parameters hold by this set
  Container_t m_param;
  // name
  const std::string myName;
};
#endif /*OHMMS_OHMMSPARAMETERSET_H*/
