#include "../include/rc_ana.hpp"
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
using std::string;

int main( int argc, char* argv[] )
{
    // Convention:
    /*
    argv[n] -> Parameter
    1 -> Rmin
    2 -> Rmax
    3 -> RbinNum
    4 -> PhiBinNum
    5 -> snapshot file address
    6 -> output file address
    */
    if ( argc != 7 )
    {
        if ( argc == 2 && string( "--help" ) == string( argv[ 1 ] ) )
        {
            printf( "mprc [Rmin] [Rmax] [RbiNum] [PhiBinNum] [snapshot address] [output file "
                    "address].\n" );
            printf( "Or: mprc --help to see this message.\n" );
            exit( 0 );
        }
        else
        {
            printf( "Get %d arguments.\n", argc - 1 );
            fprintf( stderr, "Expecte 6 arguments, or see --help for usage.\n" );
            exit( 123 );
        }
    }

    double       Rmin              = std::stod( argv[ 1 ] );
    double       Rmax              = std::stod( argv[ 2 ] );
    unsigned int RbinNum           = ( unsigned int )std::stoul( argv[ 3 ] );
    unsigned int PhiBinNum         = ( unsigned int )std::stoul( argv[ 4 ] );
    string       snapshot_filename = argv[ 5 ];
    string       log_filename      = argv[ 6 ];

    analyzer_of_snapshot analyzer( Rmin, Rmax, RbinNum, PhiBinNum, snapshot_filename,
                                   log_filename );
    analyzer.read_ana_write();
    return 0;
}
