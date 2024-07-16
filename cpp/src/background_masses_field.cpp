#include "../include/background_masses_field.hpp"
#include <cmath>
#include <cstring>
#include <thread>
#define MIN_THREAD 4
using std::vector;

masses_field::masses_field( unsigned int part_num, double* masses, double* positions )
    : dataStack( dynamic_array< double >( part_num * 4 ) )
// The stack for particle informations: Nx4 1D array, N=partNum, 4=(mass, position)
{
    this->partNum = part_num;
    for ( int i = 0; i < ( int )part_num; ++i )
    {
        this->dataStack.data_ptr[ 4 * i + 0 ] = masses[ i ];
        this->dataStack.data_ptr[ 4 * i + 1 ] = positions[ 3 * i + 0 ];
        this->dataStack.data_ptr[ 4 * i + 2 ] = positions[ 3 * i + 1 ];
        this->dataStack.data_ptr[ 4 * i + 3 ] = positions[ 3 * i + 2 ];
    }
}

void masses_field::newtonian_possion_solver_a( double pos[ 3 ], double acc[ 3 ] )
{
    acc[ 0 ] = acc[ 1 ] = acc[ 2 ] = 0;
    double delta_x = 0, delta_y = 0, delta_z = 0;  // differences of the position vectors
    double r3 = 0;                                 // cubic of radius
    for ( int i = 0; i < ( int )this->partNum; ++i )
    {
        delta_x = this->dataStack.data_ptr[ 4 * i + 1 ] - pos[ 0 ];
        delta_y = this->dataStack.data_ptr[ 4 * i + 2 ] - pos[ 1 ];
        delta_z = this->dataStack.data_ptr[ 4 * i + 3 ] - pos[ 2 ];
        r3      = std::pow( delta_x * delta_x + delta_y * delta_y + delta_z * delta_z, 1.5 );
        acc[ 0 ] += this->G * this->dataStack.data_ptr[ 4 * i + 0 ] * delta_x / r3;
        acc[ 1 ] += this->G * this->dataStack.data_ptr[ 4 * i + 0 ] * delta_y / r3;
        acc[ 2 ] += this->G * this->dataStack.data_ptr[ 4 * i + 0 ] * delta_z / r3;
    }
}

double masses_field::newtonian_possion_solver_pot( double pos[ 3 ] )
{
    double pot     = 0;
    double delta_x = 0, delta_y = 0, delta_z = 0;  // differences of the position vectors
    double r = 0;                                  // cubic of radius
    for ( int i = 0; i < ( int )this->partNum; ++i )
    {
        delta_x = this->dataStack.data_ptr[ 4 * i + 1 ] - pos[ 0 ];
        delta_y = this->dataStack.data_ptr[ 4 * i + 2 ] - pos[ 1 ];
        delta_z = this->dataStack.data_ptr[ 4 * i + 3 ] - pos[ 2 ];
        r       = std::pow( delta_x * delta_x + delta_y * delta_y + delta_z * delta_z, 0.5 );
        pot += -this->G * this->dataStack.data_ptr[ 4 * i + 0 ] / r;
    }
    return pot;
}

acceleration masses_field::acc_at( double position[ 3 ] )
{
    double tmp_acc[ 3 ];  // the temporary array of acceleration
    this->newtonian_possion_solver_a( position, tmp_acc );
    acceleration acc( tmp_acc );
    return acc;
}

double masses_field::pot_at( double position[ 3 ] )
{
    return this->newtonian_possion_solver_pot( position );
}

double masses_field::vc_at( double position[ 3 ] )
{
    double acc[ 3 ];
    this->newtonian_possion_solver_a( position, acc );
    // VC = sqrt(-vec{an} \dot vec{r} )
    return std::sqrt(
        -( position[ 0 ] * acc[ 0 ] + position[ 1 ] * acc[ 1 ] + position[ 2 ] * acc[ 2 ] ) );
}

vector< acceleration > masses_field::accs_at( double* positions, unsigned int pos_num )
// force at muti-position
{
    unsigned int used_thread = ( unsigned int )( std::thread::hardware_concurrency() / 4 );
    // at least MIN_THREAD threads
    used_thread                       = used_thread >= MIN_THREAD ? used_thread : MIN_THREAD;
    used_thread                       = used_thread >= pos_num ? used_thread : pos_num;
    unsigned int basic_num_per_thread = pos_num / used_thread;
    unsigned int extra_num            = pos_num % used_thread;
    // virtual container of the position ids
    vector< vector< unsigned int > > container( used_thread,
                                                vector< unsigned int >( basic_num_per_thread, 0 ) );
    // the basic ones
    for ( int i = 0; i < ( int )basic_num_per_thread; ++i )
    {
        for ( int j = 0; j < ( int )used_thread; ++j )
        {
            container.at( j ).at( i ) = i * used_thread + j;
        }
    }
    // the remains
    for ( int i = 0; i < ( int )extra_num; ++i )
    {
        container.at( i ).push_back( used_thread * basic_num_per_thread + i );
    }

    // dynamic memory of the acceleration array
    dynamic_array< double > accs( pos_num * 3 );
    // the thread function
    // NOTE: must call the dynamic array by reference, otherwise, it will repeatedly call the
    // deconstructer
    auto single_thread = [ this, &positions, &accs ]( vector< unsigned int > ids ) {
        for ( auto id : ids )
        {
            this->newtonian_possion_solver_a( positions + 3 * id, accs.data_ptr + 3 * id );
        }
    };

    vector< std::thread > threads;
    for ( int i = 0; i < used_thread; ++i )
    {
        threads.push_back( std::thread( single_thread, container.at( i ) ) );
    }
    for ( int i = 0; i < used_thread; ++i )
    {
        threads[ i ].join();
    }

    vector< acceleration > accs_vector;
    for ( int i = 0; i < ( int )pos_num; ++i )
    {
        accs_vector.push_back( acceleration( accs.data_ptr + 3 * i ) );
    }

    return accs_vector;
}

vector< double > masses_field::pots_at( double* positions, unsigned int pos_num )
{
    unsigned int used_thread = ( unsigned int )( std::thread::hardware_concurrency() / 4 );
    // at least MIN_THREAD threads
    used_thread                       = used_thread >= MIN_THREAD ? used_thread : MIN_THREAD;
    used_thread                       = used_thread <= pos_num ? used_thread : pos_num;
    unsigned int basic_num_per_thread = pos_num / used_thread;
    unsigned int extra_num            = pos_num % used_thread;
    // virtual container of the position ids
    vector< vector< unsigned int > > container( used_thread,
                                                vector< unsigned int >( basic_num_per_thread, 0 ) );
    // the basic ones
    for ( int i = 0; i < ( int )basic_num_per_thread; ++i )
    {
        for ( int j = 0; j < ( int )used_thread; ++j )
        {
            container.at( j ).at( i ) = i * used_thread + j;
        }
    }
    // the remains
    for ( int i = 0; i < ( int )extra_num; ++i )
    {
        container.at( i ).push_back( used_thread * basic_num_per_thread + i );
    }

    // dynamic memory of the acceleration array
    dynamic_array< double > pots( pos_num );
    // the thread function
    // NOTE: must call the dynamic array by reference, otherwise, it will repeatedly call the
    // deconstructer
    auto single_thread = [ this, &positions, &pots ]( vector< unsigned int > ids ) {
        for ( auto id : ids )
        {
            pots.data_ptr[ id ] = this->newtonian_possion_solver_pot( positions + 3 * id );
        }
    };

    vector< std::thread > threads;
    for ( int i = 0; i < used_thread; ++i )
    {
        threads.push_back( std::thread( single_thread, container.at( i ) ) );
    }
    for ( int i = 0; i < used_thread; ++i )
    {
        threads[ i ].join();
    }

    vector< double > pots_vector;
    for ( int i = 0; i < ( int )pos_num; ++i )
    {
        pots_vector.push_back( pots.data_ptr[ i ] );
    }

    return pots_vector;
}

vector< double > masses_field::vcs_at( double* positions, unsigned int pos_num )
{
    unsigned int used_thread = ( unsigned int )( std::thread::hardware_concurrency() / 4 );
    // at least MIN_THREAD threads
    used_thread                       = used_thread >= MIN_THREAD ? used_thread : MIN_THREAD;
    used_thread                       = used_thread <= pos_num ? used_thread : pos_num;
    unsigned int basic_num_per_thread = pos_num / used_thread;
    unsigned int extra_num            = pos_num % used_thread;
    // virtual container of the position ids
    vector< vector< unsigned int > > container( used_thread,
                                                vector< unsigned int >( basic_num_per_thread, 0 ) );
    // the basic ones
    for ( int i = 0; i < ( int )basic_num_per_thread; ++i )
    {
        for ( int j = 0; j < ( int )used_thread; ++j )
        {
            container.at( j ).at( i ) = i * used_thread + j;
        }
    }
    // the remains
    for ( int i = 0; i < ( int )extra_num; ++i )
    {
        container.at( i ).push_back( used_thread * basic_num_per_thread + i );
    }

    // dynamic memory of the acceleration array
    dynamic_array< double > vcs( pos_num );
    // the thread function
    // NOTE: must call the dynamic array by reference, otherwise, it will repeatedly call the
    // deconstructer
    auto single_thread = [ this, &positions, &vcs ]( vector< unsigned int > ids ) {
        double acc[ 3 ];
        for ( auto id : ids )
        {
            this->newtonian_possion_solver_a( positions + 3 * id, acc );
            // VC = sqrt(-vec{an} \dot vec{r} )
            vcs.data_ptr[ id ] = std::sqrt( -( *( positions + 3 * id + 0 ) * acc[ 0 ]
                                               + *( positions + 3 * id + 1 ) * acc[ 1 ]
                                               + *( positions + 3 * id + 2 ) * acc[ 2 ] ) );
        }
    };

    vector< std::thread > threads;
    for ( int i = 0; i < used_thread; ++i )
    {
        threads.push_back( std::thread( single_thread, container.at( i ) ) );
    }
    for ( int i = 0; i < used_thread; ++i )
    {
        threads[ i ].join();
    }

    vector< double > vcs_vector;
    for ( int i = 0; i < ( int )pos_num; ++i )
    {
        vcs_vector.push_back( vcs.data_ptr[ i ] );
    }

    return vcs_vector;
}
