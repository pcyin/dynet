#ifndef DYNET_LSTMCELL_H_
#define DYNET_LSTMCELL_H_

#include "dynet/dynet.h"
#include "dynet/expr.h"

using namespace std;

namespace dynet {

class ParameterCollection;

class LSTMCell {
  LSTMCell() = default;
    
public:
  LSTMCell(unsigned input_dim,
           unsigned hidden_dim,
           ParameterCollection& model);

  void on_new_graph(ComputationGraph& cg);

  std::vector<Expression> step(const Expression& x, const std::vector<Expression>& s_tm1);

  unsigned input_dim;

  unsigned hidden_dim;
  
  ParameterCollection local_model;

  std::vector<Parameter> params;
  
  std::vector<Expression> param_vars;

private:
};

} // namespace dynet

#endif