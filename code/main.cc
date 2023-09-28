#include <cstdio>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/types.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_project.h>
#include <deal.II/numerics/vector_tools_rhs.h>
#include <exception>
#include <iostream>
#include <fstream>

#include "include/reactMath.hpp"

// Deal.II Libraries

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/utilities.h>

namespace reactionDiffusion
{
    using namespace dealii;

    template<int dim>
    class ReactionDiffusionEquation
    {
    public:
        ReactionDiffusionEquation();
        void run();
    private:
        void setup_system();
        void solve_q();
        void solve_r();
        void output_results() const;

        Triangulation<dim> triangulation;
        FE_Q<dim>          fe;
        DoFHandler<dim>    dof_handler;
    };

    
}
