#include "../include/snapH5io.hpp"
#include "../include/others.hpp"
#include "H5Apublic.h"
#include "H5Dpublic.h"
#include "H5Gpublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5public.h"
#include <string>
#define LONGER_THAN_NumPart_ThisFile 100
using std::string;

snapH5io::snapH5io( std::string inFile, std::string outFile )
{
    this->inFile  = H5Fopen( inFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
    this->outFile = H5Fcreate( outFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

    int* partNums = new int[ LONGER_THAN_NumPart_ThisFile ];
    memset( partNums, 0, sizeof( int ) * LONGER_THAN_NumPart_ThisFile );

    hid_t header  = H5Gopen( this->inFile, "/Header", H5P_DEFAULT );
    hid_t attr_id = H5Aopen( header, "NumPart_ThisFile", H5P_DEFAULT );
    H5Aread( attr_id, H5T_NATIVE_INT, partNums );

    for ( int i = 0; i < LONGER_THAN_NumPart_ThisFile; i++ )
        if ( partNums[ i ] != 0 )
        {
            string partTypeName                    = "PartType" + std::to_string( i );
            this->parttype2partNum[ partTypeName ] = partNums[ i ];
        }

    H5Aclose( attr_id );
    H5Gclose( header );
}

snapH5io::~snapH5io()
{
    H5Fclose( this->inFile );
    if ( this->outFile != -1 )
        H5Fclose( this->outFile );
}

void snapH5io::read_datasets( vector< unsigned int >& partNums, vector< double* >& coordinates,
                              vector< double* >& masses, vector< string >& datasetNames )
{
    // check whether the given vectors are uninitialized
    if ( partNums.size() != 0 || coordinates.size() != 0 || masses.size() != 0
         || datasetNames.size() != 0 )
        throw "The vectors used for dataset reading should be uninitialized.";

    // read the coordinates and masses
    int i = 0;
    for ( auto pair = this->parttype2partNum.begin(); pair != this->parttype2partNum.end(); ++pair )
    {
        datasetNames.push_back( pair->first );
        partNums.push_back( pair->second );
        hid_t group_id_of_comp = H5Gopen2( this->inFile, pair->first.c_str(), H5P_DEFAULT );

        hid_t   mass_set_id = H5Dopen2( group_id_of_comp, "Masses", H5P_DEFAULT );
        double* mass_ptr    = new double[ pair->second ];
        memset( mass_ptr, 0, sizeof( double ) * pair->second );
        H5Dread( mass_set_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mass_ptr );
        H5Dclose( mass_set_id );
        masses.push_back( mass_ptr );

        hid_t   coord_set_id = H5Dopen2( group_id_of_comp, "Coordinates", H5P_DEFAULT );
        double* coord_ptr    = new double[ pair->second * 3 ];
        memset( coord_ptr, 0, sizeof( double ) * pair->second * 3 );
        H5Dread( coord_set_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coord_ptr );
        H5Dclose( coord_set_id );
        coordinates.push_back( coord_ptr );

        H5Gclose( group_id_of_comp );
    }
}

void snapH5io::write_rc_infos( unsigned int RbinNum, unsigned int PhiBinNum, double* accs,
                               double* rvs, double* pots, string datasetName )
{
    hid_t group_id_of_comp =
        H5Gcreate2( this->outFile, datasetName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    hsize_t dims_1d[ 1 ] = { ( hsize_t )RbinNum * PhiBinNum };     // for potentials et al.
    hsize_t dims_2d[ 2 ] = { ( hsize_t )RbinNum * PhiBinNum, 3 };  // for accelerations

    // create the datasets in the analysis result file
    // Datasets: RC, RS with the same size, 1D array
    hid_t potSet =
        H5Dcreate2( group_id_of_comp, "Potentials", H5T_NATIVE_DOUBLE,
                    H5Screate_simple( 1, dims_1d, NULL ), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hid_t rvSet =
        H5Dcreate2( group_id_of_comp, "Rotation Velocities", H5T_NATIVE_DOUBLE,
                    H5Screate_simple( 1, dims_1d, NULL ), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hid_t accSet =
        H5Dcreate( group_id_of_comp, "Accelerations", H5T_NATIVE_DOUBLE,
                   H5Screate_simple( 2, dims_2d, NULL ), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    H5Dwrite( potSet, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, pots );
    H5Dwrite( rvSet, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rvs );
    H5Dwrite( accSet, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, accs );
    H5Dclose( potSet );
    H5Dclose( rvSet );
    H5Dclose( accSet );
    H5Gclose( group_id_of_comp );
}

void snapH5io::write_coordinates( unsigned int RbinNum, unsigned int PhiBinNum, double Rmin,
                                  double Rmax, double* poses )
{
    hsize_t dims_0d[ 1 ] = { ( hsize_t )1 };                       // for attributes
    hsize_t dims_2d[ 2 ] = { ( hsize_t )RbinNum * PhiBinNum, 3 };  // for coordinates


    // create the datasets in the analysis result file
    // Datasets: RC, RS with the same size, 1D array
    hid_t coordinate_setid =
        H5Dcreate( this->outFile, "Coordinates", H5T_NATIVE_DOUBLE,
                   H5Screate_simple( 2, dims_2d, NULL ), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    H5Dwrite( coordinate_setid, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, poses );

    hid_t space_id = H5Screate( H5S_SCALAR );
    hid_t attr_id = H5Acreate2( coordinate_setid, "RbinNum", H5T_NATIVE_UINT, space_id, H5P_DEFAULT,
                                H5P_DEFAULT );
    H5Awrite( attr_id, H5T_NATIVE_UINT, &RbinNum );
    H5Aclose( attr_id );

    attr_id = H5Acreate2( coordinate_setid, "PhiBinNum", H5T_NATIVE_UINT, space_id, H5P_DEFAULT,
                          H5P_DEFAULT );
    H5Awrite( attr_id, H5T_NATIVE_UINT, &PhiBinNum );
    H5Aclose( attr_id );

    attr_id = H5Acreate2( coordinate_setid, "Rmin", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT,
                          H5P_DEFAULT );
    H5Awrite( attr_id, H5T_NATIVE_DOUBLE, &Rmin );
    H5Aclose( attr_id );

    attr_id = H5Acreate2( coordinate_setid, "Rmax", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT,
                          H5P_DEFAULT );
    H5Awrite( attr_id, H5T_NATIVE_DOUBLE, &Rmax );
    H5Aclose( attr_id );

    H5Sclose( space_id );
    H5Dclose( coordinate_setid );
}
