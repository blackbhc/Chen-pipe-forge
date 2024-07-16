#include "../include/rc_ana.hpp"
#include "../include/background_masses_field.hpp"
#include <math.h>
#include <vector>

analyzer_of_snapshot::analyzer_of_snapshot( double Rmin, double Rmax, unsigned int RbinNum,
                                            unsigned int PhiBinNum, string in_filename,
                                            string out_filename )
    : coordinates( dynamic_array< double >( RbinNum * PhiBinNum * 3 ) ), rMin( Rmin ), rMax( Rmax ),
      rBinNum( RbinNum ), phiBinNum( PhiBinNum ), pos_num( RbinNum * PhiBinNum ),
      h5io( in_filename, out_filename )
{
    dynamic_array< double > rs( this->rBinNum );
    dynamic_array< double > phis( this->phiBinNum );

    double deltaR   = ( Rmax - Rmin ) / this->rBinNum;
    double deltaPhi = 2 * M_PI / this->phiBinNum;

    for ( int i = 0; i < ( int )this->rBinNum; ++i )
        rs.data_ptr[ i ] = this->rMin + deltaR * i;

    for ( int i = 0; i < ( int )this->phiBinNum; ++i )
        phis.data_ptr[ i ] = deltaPhi * i;

    for ( int i = 0; i < ( int )this->rBinNum; ++i )
    {
        for ( int j = 0; j < ( int )this->phiBinNum; ++j )
        {
            coordinates.data_ptr[ ( i * this->phiBinNum + j ) * 3 + 0 ] =
                rs.data_ptr[ i ] * cos( phis.data_ptr[ j ] );
            coordinates.data_ptr[ ( i * this->phiBinNum + j ) * 3 + 1 ] =
                rs.data_ptr[ i ] * sin( phis.data_ptr[ j ] );
        }
    }
}

void analyzer_of_snapshot::read_ana_write()
{
    // read snapshot
    vector< unsigned int > partNums;
    vector< string >       datasetNames;
    vector< double* >      coordinates;  // coordinates in the snapshot
    vector< double* >      masses;
    this->h5io.read_datasets( partNums, coordinates, masses, datasetNames );

    // write the coordinates
    this->h5io.write_coordinates( this->rBinNum, this->phiBinNum, this->coordinates.data_ptr );

    int compNum = ( int )partNums.size();
    for ( int i = 0; i < compNum; ++i )
    {
        // analyze snapshot
        masses_field cur_field( partNums.at( i ), masses.at( i ), coordinates.at( i ) );
        // create dynamic arrays of the log datas
        vector< acceleration > accs =
            cur_field.accs_at( this->coordinates.data_ptr, this->pos_num );

        vector< double > pots = cur_field.pots_at( this->coordinates.data_ptr, this->pos_num );
        vector< double > rvs  = pots;
        dynamic_array< double > accs_array( this->pos_num * 3 );
        for ( int i = 0; i < ( int )this->pos_num; ++i )
        {
            rvs.at( i ) = accs.at( i ).comp_R( this->coordinates.data_ptr + i );
            accs.at( i ).assign_to( accs_array.data_ptr + 3 * i );
        }
        // write other results
        this->h5io.write_rc_infos( this->rBinNum, this->phiBinNum, accs_array.data_ptr, rvs.data(),
                                   pots.data(), datasetNames.at( i ) );
    }

    for ( auto ptr : coordinates )
        delete[] ptr;
    for ( auto ptr : masses )
        delete[] ptr;
}
