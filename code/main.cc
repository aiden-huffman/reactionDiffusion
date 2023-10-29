#include "reactMath.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deal.II/grid/grid_tools.h>
#include <exception>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <reactMath.hpp>

// Deal.II Libraries
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_project.h>
#include <deal.II/numerics/vector_tools_rhs.h>

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
    
    // Class definition
    template<int dim> class ReactionDiffusionEquation
    {
    public:
        ReactionDiffusionEquation();
        void run(const std::unordered_map<std::string, double>   params,
                 const double                                    totalSimulationTime);

    private:
        void setup_system(const std::unordered_map<std::string, double> params,
                          const double                                  totalSimulationTime);
        void solveQ();
        void solveR();
        void outputResults() const;

        Triangulation<dim> triangulation;
        FE_Q<dim>          fe;
        DoFHandler<dim>    dofHandler;

        AffineConstraints<double>   constraints;

        SparsityPattern         sparsityPattern;
        SparseMatrix<double>    massMatrix;
        SparseMatrix<double>    laplaceMatrix;
        SparseMatrix<double>    matrixQ;
        SparseMatrix<double>    matrixR;

        Vector<double>  solutionQ, solutionR;
        Vector<double>  oldSolutionQ, oldSolutionR;
        Vector<double>  systemRightHandSideQ;
        Vector<double>  systemRightHandSideR;

        double          timeStep;
        double          time;
        double          totalSimulationTime;
        unsigned int    timestepNumber;

        // Reaction Parameters
        double          a;
        double          b;
        double          gamma;
        double          D;

    };

    // Constructor
    template<int dim> ReactionDiffusionEquation<dim>::ReactionDiffusionEquation()
        : fe(1)
        , dofHandler(triangulation)
        , timeStep( 1. / 256 )
        , time(timeStep)
        , timestepNumber(1)
    {}

    template<int dim> class InitialConditionsQ : public Function<dim>
    {
        public:
            InitialConditionsQ(double a, double b);
            virtual double value(const Point<dim> & p,
                                 const unsigned int component = 0) const override;
        private:
            double a;
            double b;

            double steady;
            
            mutable std::default_random_engine          generator;
            mutable std::normal_distribution<double>    distribution;
    };

    // Initial Values Constructor
    template<int dim> InitialConditionsQ<dim>::InitialConditionsQ(double a,
                                                                  double b)
        : a(a)
        , b(b)
        , steady(a+b)
        , generator()
        , distribution(steady, 0.1)
    {}

    // Value function declaration
    template<int dim> double InitialConditionsQ<dim> 
        :: value (const Point<dim> &p,
                  const unsigned int i /*component*/) const
    {
        return this->distribution(this->generator);
    }

    template<int dim> class InitialConditionsR : public Function<dim>
    {
        public:
            InitialConditionsR(double a, double b);
            virtual double value(const Point<dim> & p,
                                 const unsigned int component = 0) const override;
        private:
            double a;
            double b;
            
            double steady;
            
            mutable std::default_random_engine          generator;
            mutable std::normal_distribution<double>    distribution;
    };
    // Initial Values Constructor
    template<int dim> InitialConditionsR<dim>::InitialConditionsR(double a,
                                                                  double b)
        : a(a)
        , b(b)
        , steady(b / pow(a + b,2))
        , generator()
        , distribution(steady, 0.1)
    {}

    // Value function declaration
    template<int dim> double InitialConditionsR<dim> 
        :: value (const Point<dim> &p,
                  const unsigned int i /*component*/) const
    {
        return this->distribution(this->generator);
    }

    // Setup system
    template<int dim> void ReactionDiffusionEquation<dim> :: setup_system(
        const std::unordered_map<std::string, double> params,
        const double         totalSimulationTime
    )
    {
        
        std::cout << "Passing parameters:" << std::endl;
        for(auto it=params.begin(); it!=params.end(); it++){
            std::cout   << "    " << it->first
                        << ": " << it->second
                        << std::endl;
        }
        this->a     = params.at("a");
        this->b     = params.at("b");
        this->gamma = params.at("gamma");
        this->D     = params.at("D");
        
        this->totalSimulationTime = totalSimulationTime;

        std::cout   << "Current parameter set:\n\n"
                    << "a: " << this->a << std::endl
                    << "b: " << this->b << std::endl
                    << "gamma: " << this->gamma << std::endl
                    << "D : " << this->D
                    << std::endl;

        std::cout   << "\nBuilding mesh" << std::endl;

        GridGenerator::hyper_cube(
            this->triangulation,
            0, 2*M_PI/params.at("k"),
            true
        );

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matchedPairsXX;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matchedPairsYY;
        // Flag adjacent faces in the triangulation due to periodicity
        GridTools::collect_periodic_faces(this->triangulation,
                                          0, 1, 0, matchedPairsXX);
        triangulation.add_periodicity(matchedPairsXX);
        GridTools::collect_periodic_faces(this->triangulation,
                                          2, 3, 1, matchedPairsYY);
        triangulation.add_periodicity(matchedPairsYY);
        triangulation.refine_global(8);

        

        std::cout   << "Mesh generated...\n"
                    << "Active Cells: " << triangulation.n_active_cells()
                    << std::endl;

        std::cout   << "\nIndexing degrees of freedom..."
                    << std::endl;

        this->dofHandler.distribute_dofs(fe);

        std::cout   << "Number of degrees of freedom: "
                    << dofHandler.n_dofs()
                    << std::endl;

        // Sparsity pattern is built from the 'relationships' between DOFs.
        // It can therefore be built without any additional information

        std::cout   << "\nBuilding sparsity pattern..."
                    << std::endl;
        std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>
        > periodicity_vectorX;

        std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>
        > periodicity_vectorY;

        GridTools::collect_periodic_faces(this->dofHandler,
                                          0,1,0,periodicity_vectorX);
        GridTools::collect_periodic_faces(this->dofHandler,
                                          2,3,1,periodicity_vectorY);

        DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorX,
                                                        this->constraints);
        DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorY,
                                                        this->constraints);

        constraints.close();

        DynamicSparsityPattern dsp(
            dofHandler.n_dofs(),
            dofHandler.n_dofs()
        );
        DoFTools::make_sparsity_pattern(dofHandler,
                                        dsp,
                                        this->constraints);
        sparsityPattern.copy_from(dsp);
        sparsityPattern.compress();
        
        std::cout   << "Reinitializing matrices based on new pattern..."
                    << std::endl;

        massMatrix.reinit(sparsityPattern);
        laplaceMatrix.reinit(sparsityPattern);
        matrixQ.reinit(sparsityPattern);
        matrixR.reinit(sparsityPattern);
        
        std::cout   << "Filling entries for mass and laplace matrix..."
                    << std::endl;

        MatrixCreator::create_mass_matrix(
            dofHandler,
            QGauss<dim>(fe.degree+1),
            massMatrix
        );

        MatrixCreator::create_laplace_matrix(
            dofHandler,
            QGauss<dim>(fe.degree+1),
            laplaceMatrix
        );

        // Initialize the vectors 
        std::cout   << "Initializing the various vectors..."
                    << std::endl;

        this->solutionQ.reinit(dofHandler.n_dofs());
        this->solutionR.reinit(dofHandler.n_dofs());
        this->oldSolutionQ.reinit(dofHandler.n_dofs());
        this->oldSolutionR.reinit(dofHandler.n_dofs());
        this->systemRightHandSideQ.reinit(dofHandler.n_dofs());
        this->systemRightHandSideR.reinit(dofHandler.n_dofs());

        std::cout   << "Calculating initial values and storing..."
                    << std::endl;

    }

    template<int dim> void ReactionDiffusionEquation<dim> :: solveQ()
    {
        SolverControl               solverControl(
                                        1000,
                                        1e-8 * systemRightHandSideQ.l2_norm()
                                    );
        SolverCG<Vector<double>>    cg(solverControl);
        
        this->constraints.condense(
            this->matrixQ,
            this->systemRightHandSideQ);
        cg.solve(
            this->matrixQ,
            this->solutionQ,
            this->systemRightHandSideQ,
            PreconditionIdentity()
        );
        this->constraints.distribute(solutionQ);

        std::cout   << "    Q solved: "
                    << solverControl.last_step()
                    << " CG iterations."
                    << std::endl;
    };
    
    template<int dim> void ReactionDiffusionEquation<dim> :: solveR()
    {
        SolverControl               solverControl(
                                        1000,
                                        1e-8 * systemRightHandSideR.l2_norm()
                                    );
        SolverCG<Vector<double>>    cg(solverControl);

        this->constraints.condense(
            this->matrixR,
            this->systemRightHandSideR);
        cg.solve(
            this->matrixR,
            this->solutionR,
            this->systemRightHandSideR,
            PreconditionIdentity()
        );

        this->constraints.distribute(solutionR);

        std::cout   << "    R solved: "
                    << solverControl.last_step()
                    << " CG iterations."
                    << std::endl;
    };

    template<int dim> void ReactionDiffusionEquation<dim> 
    :: outputResults() const {
        DataOut<dim> dataOut;

        dataOut.attach_dof_handler(this->dofHandler);
        dataOut.add_data_vector(this->solutionQ, "Q");
        dataOut.add_data_vector(this->solutionR, "R");

        dataOut.build_patches();

        const std::string filename = ("data/solution-" 
                                     + std::to_string(this->timestepNumber) 
                                     + ".vtu");

        DataOutBase::VtkFlags vtk_flags;
        vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
        dataOut.set_flags(vtk_flags);

        std::ofstream output(filename);
        dataOut.write_vtu(output);

    };

    // Run simulation
    template<int dim> void ReactionDiffusionEquation<dim> :: run (
        const std::unordered_map<std::string, double>   params,
        const double                                    totalSimulationTime
    )
    {   
        this->setup_system(params, totalSimulationTime);
        
        QGauss<dim>     quadratureFormula(this->fe.degree+1);

        Vector<double>  cellRightHandSideQ(fe.n_dofs_per_cell());
        Vector<double>  cellRightHandSideR(fe.n_dofs_per_cell());

        std::vector<double>  solutionValuesQ(quadratureFormula.size());
        std::vector<double>  solutionValuesR(quadratureFormula.size());

        FEValues<dim>   feValues(this->fe,
                                 quadratureFormula,
                                 update_values | update_JxW_values);

        VectorTools::project(
            this->dofHandler,
            this->constraints,
            QGauss<dim>(fe.degree + 1),
            InitialConditionsQ<dim>(this->a, this->b),
            this->oldSolutionQ
        );

        VectorTools::project(
            this->dofHandler,
            this->constraints,
            QGauss<dim>(fe.degree +1),
            InitialConditionsR<dim>(this->a, this->b),
            this->oldSolutionR
        );

        VectorTools::project(
            this->dofHandler,
            this->constraints,
            QGauss<dim>(fe.degree +1),
            InitialConditionsQ<dim>(this->a, this->b),
            this->solutionQ
        );

        VectorTools::project(
            this->dofHandler,
            this->constraints,
            QGauss<dim>(fe.degree +1),
            InitialConditionsR<dim>(this->a, this->b),
            this->solutionR
        );
        
        std::vector<types::global_dof_index> 
            local_dof_indices(fe.n_dofs_per_cell());

        for(; this->time < this->totalSimulationTime; ++this->timestepNumber)
        { 
        
            matrixQ.copy_from(massMatrix);
            matrixQ.add(this->timeStep, laplaceMatrix);

            matrixR.copy_from(massMatrix);
            matrixR.add(this->D * this->timeStep, laplaceMatrix);

            massMatrix.vmult(systemRightHandSideQ, this->oldSolutionQ);
            massMatrix.vmult(systemRightHandSideR, this->oldSolutionR);

            for(const auto &cell : dofHandler.active_cell_iterators())
            {
                feValues.reinit(cell);

                feValues.get_function_values(oldSolutionQ,
                                             solutionValuesQ);
                feValues.get_function_values(oldSolutionR,
                                             solutionValuesR);
                cell->get_dof_indices(local_dof_indices);

                for(const unsigned int qIndex : feValues.quadrature_point_indices())
                {
                    
                    double Qx = solutionValuesQ[qIndex];
                    double Rx = solutionValuesR[qIndex];

                    if (!(Qx >= 0))
                    {
                        std::cout   << std::endl;
                        std::cerr   << "Concentration of Q is not non-negative"
                                    << std::endl;

                        assert(Qx >= 0);
                    } 
                    else if (!(Rx >= 0))
                    {
                        std::cout   << std::endl;
                        std::cerr   << "Concentration of R is not non-negative"
                                    << std::endl;

                        assert(Rx >= 0);
                    }


                    for(const unsigned int i : feValues.dof_indices())
                    {
                        
                        systemRightHandSideQ(local_dof_indices[i]) += feValues.shape_value(i, qIndex) * 
                                (   this->gamma * (this->a - Qx + pow(Qx,2) * Rx)
                                ) * feValues.JxW(qIndex) * this->timeStep;
                        systemRightHandSideR(local_dof_indices[i]) += feValues.shape_value(i, qIndex) *
                                (   this->gamma * (this->b - pow(Qx,2) * Rx)
                                ) * feValues.JxW(qIndex) * this->timeStep;

                    }
                    
                }
            }

            this->time += this->timeStep;
            std::cout   << "Time step "
                        << this->timestepNumber
                        << " at time = "
                        << this->time
                        << std::endl;

            solveQ();
            solveR();

            double maxQ = *std::max_element(this->solutionQ.begin(),
                                            this->solutionQ.end());
            double minQ = *std::min_element(this->solutionQ.begin(),
                                            this->solutionQ.end());

        
            double maxR = *std::max_element(this->solutionR.begin(),
                                            this->solutionR.end());
            double minR = *std::min_element(this->solutionR.begin(),
                                            this->solutionR.end());

            std::cout   << "    Q Range: (" << minQ << ", " <<  maxQ << ")" 
                        << std::endl
                        << "    R Range: (" << minR << ", " <<  maxR << ")"
                        << std::endl << std::endl;
            
            if(this->timestepNumber % 10 == 0){
                outputResults();
            }
            
            oldSolutionQ = solutionQ;
            oldSolutionR = solutionR;
        }
    };
}

int main(){

    std::cout   << "Running" << std::endl
                << std::endl;

    std::unordered_map<std::string, double>     params;

    params["a"]     = 0.2;
    params["b"]     = 2.0;
    params["gamma"] = 2.0;
    params["D"]     = calcCritDiff(params.at("a"), params.at("b")) + 10.;

    params["k"] =  calcCritWavenumber(params.at("a"),
                                      params.at("b"),
                                      params.at("gamma"));

    double totalSimulationTime = 100;

    reactionDiffusion::ReactionDiffusionEquation<2> reactionDiffusion;

    reactionDiffusion.run(params, totalSimulationTime);

    std::cout   << "Completed..." << std::endl;

    return 0;
}
