#ifndef OTHERS_HH
#define OTHERS_HH
#include <cstdio>
#include <cstring>
#include <vector>
using std::vector;

#define test_reach printf( "Call from [File:%s, Line:%d].\n", __FILE__, __LINE__ )

class vec3  // the 1x3 vector
{
public:
    double data[ 3 ] = { 0 };
    bool   type_is_same( vec3& another_vec3 );
    vec3( double data[ 3 ] );
    double norm();
    void   normalize();
    double dot( vec3& another );
    vec3   cross( vec3& another );
    void   print()
    {
        printf( "%lf, %lf, %lf\n", this->data[ 0 ], this->data[ 1 ], this->data[ 2 ] );
    };
    void assign_to( double* array_ptr );
};

class acceleration : public vec3
{
public:
    acceleration( double data[ 3 ] ) : vec3( data ){};
    // calculate the components of the force by inner products w.r.t. the related unit vectors
    double comp_R( double pos[ 3 ] );  // cylindrically radial direction
};

template < typename Type > class dynamic_array
{
public:
    Type* data_ptr = nullptr;
    dynamic_array( unsigned int len )
    {
        this->data_ptr = new Type[ len ];
        memset( this->data_ptr, 0, sizeof( Type ) * len );
    }

    ~dynamic_array()
    {
        delete[] data_ptr;
        data_ptr = nullptr;
    }
};

#endif
