#include "../include/background_masses_field.hpp"
#include <cstdio>
// #include <iostream>

int main( int argc, char** argv )
{
    double       test_mass     = 1;  // 1e10 Msun
    double       test_pos[ 3 ] = { 0, 0, 0 };
    masses_field single_particle_field( 1, &test_mass, test_pos );
    // double       target_positions[ 12 ] = { -1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1 };
    // double target_positions[ 12 ] = { -1, 0, 0, 2, 0, 0, 3, 4, 0, 3, 4, 8.66 };
    double target_positions[ 12 ] = { -1, 1, 3, 0.5, 2, 2, -10, 2, 0, 0, -1.14, 1.618 };

    printf( "Accelerations:\n" );
    printf( "Call from the acc_at api:\n" );
    for ( int i = 0; i < 4; ++i )
        single_particle_field.acc_at( target_positions + i * 3 ).print();

    printf( "Call from the accs_at api:\n" );
    auto accs = single_particle_field.accs_at( target_positions, 4 );
    for ( auto acc : accs )
        acc.print();
    printf( "Radial components of the  accelerations:\n" );
    int i = 0;
    for ( auto acc : accs )
    {
        printf( "%lf, ", acc.comp_R( target_positions + ( i++ ) * 3 ) );
    }
    printf( "\n" );

    printf( "Potentials:\n" );
    printf( "Call from the pot_at api:\n" );
    for ( int i = 0; i < 4; ++i )
    {
        printf( "%lf, ", single_particle_field.pot_at( target_positions + i * 3 ) );
    }
    printf( "\n" );

    printf( "Call from the pots_at api:\n" );
    auto pots = single_particle_field.pots_at( target_positions, 4 );
    for ( int i = 0; i < 4; ++i )
    {
        printf( "%lf, ", pots.at( i ) );
    }
    printf( "\n" );

    return 0;
}
