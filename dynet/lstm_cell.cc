#include "dynet/lstm_cell.h"

#include "dynet/param-init.h"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

namespace dynet {

enum { X2I, H2I, BI };

LSTMCell::LSTMCell(unsigned input_dim, 
            unsigned hidden_dim, 
            ParameterCollection& model): input_dim(input_dim), hidden_dim(hidden_dim) {
    
  local_model = model.add_subcollection("lstm-cell");

  Parameter W_x2i = local_model.add_parameters({hidden_dim * 4, input_dim});
  Parameter W_h2i = local_model.add_parameters({hidden_dim * 4, hidden_dim});
  Parameter bi = local_model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));

  params = {W_x2i, W_h2i, bi};
}

void LSTMCell::on_new_graph(ComputationGraph& cg) {
  param_vars.clear();

  Expression p_x2i = parameter(cg, params[X2I]);
  Expression p_h2i = parameter(cg, params[H2I]);
  Expression p_bi = parameter(cg, params[BI]);

  param_vars = {p_x2i, p_h2i, p_bi};
}

std::vector<Expression> LSTMCell::step(const Expression& x, const std::vector<Expression>& s_tm1) {
  Expression c_tm1 = s_tm1[0];
  Expression h_tm1 = s_tm1[1];

  Expression p_x2i = param_vars[X2I];
  Expression p_h2i = param_vars[H2I];
  Expression bi = param_vars[BI];
  
  Expression tmp = affine_transform({bi, p_x2i, x, p_h2i, h_tm1});

  Expression i_ait = pick_range(tmp, 0, hidden_dim);
  Expression i_aft = pick_range(tmp, hidden_dim, hidden_dim * 2);
  Expression i_aot = pick_range(tmp, hidden_dim * 2, hidden_dim * 3);
  Expression i_agt = pick_range(tmp, hidden_dim * 3, hidden_dim * 4);
  Expression i_it = logistic(i_ait);
  Expression i_ft = logistic(i_aft + 1.f);  // is this standard?
  Expression i_ot = logistic(i_aot);
  Expression i_gt = tanh(i_agt);

  Expression c_t = cmult(i_ft, c_tm1) + cmult(i_it, i_gt);
  Expression h_t = cmult(i_ot, tanh(c_t));

  std::vector<Expression> s_t = {c_t, h_t};

  return s_t;
}


}