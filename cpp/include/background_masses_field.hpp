#ifndef BACKGROUND_MASSES_FIELD
#define BACKGROUND_MASSES_FIELD
#include "../include/others.hpp"
class masses_field
{
public:
    masses_field( unsigned int part_num, double* masses, double* positions );
    acceleration           acc_at( double position[ 3 ] );  // force at single position
    vector< acceleration > accs_at( double*      positions,
                                    unsigned int pos_num );  // forces at multi-position
    // TODO: the pot_at functions
    double           pot_at( double position[ 3 ] );  // potential at single position
    vector< double > pots_at( double*      positions,
                              unsigned int pos_num );  // potentials at multi-position
    double           vc_at( double position[ 3 ] );    // circular velocity at single position
    vector< double > vcs_at( double*      positions,
                             unsigned int pos_num );  // circular velocity at multi-position

private:
    unsigned int partNum = 0;
    // The stack for particle informations: Nx4 1D array, N=partNum, 4=(mass, position)
    // double* dataStack = nullptr;
    dynamic_array< double > dataStack;
    void                    newtonian_possion_solver_a( double pos[ 3 ], double acc[ 3 ] );
    double                  newtonian_possion_solver_pot( double pos[ 3 ] );
    double                  G = 43007.1;  // In standard convention of Gadget4
};
#endif
